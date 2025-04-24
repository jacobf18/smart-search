import re
import threading

from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from logger_config import logger
from vllm_client_local import VllmClient, get_vllm_model_id
from search.search_utils import search_by_http
from data_utils import format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from agent.agent_utils import RagPath
from utils import batch_truncate


def _normalize_subquery(subquery: str) -> str:
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery


class CoRagAgent:

    def __init__(
            self, vllm_client: VllmClient, corpus: Dataset, e5_ip: str, vllm_ip: str
    ):
        self.vllm_client = vllm_client
        self.corpus = corpus
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(get_vllm_model_id(host=vllm_ip))
        self.lock = threading.Lock()
        self.e5_ip = e5_ip
        self.vllm_ip = vllm_ip

    def sample_path(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            **kwargs
    ) -> RagPath:
        """Sample a path for the given query and task description.
        This function generates subqueries and their corresponding answers iteratively 
        until the maximum path length is reached or the maximum number of LLM calls is exceeded.

        Args:
            query (str): Query to be answered
            task_desc (str): Task description to be used in the prompt
            max_path_length (int, optional): Maximum path length. Defaults to 3.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.
            temperature (float, optional): LLM temperature. Used for randomness. Defaults to 0.7.

        Returns:
            RagPath: A path object containing the query, subqueries, answers, and document IDs.
        """
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        past_doc_ids: List[List[str]] = kwargs.pop('past_doc_ids', [])
        scores: List[int] = []
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)

        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 4 * (max_path_length - len(past_subqueries))
        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            num_llm_calls += 1
            messages: List[Dict] = get_generate_subquery_prompt(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                task_desc=task_desc,
            )
            self._truncate_long_messages(messages, max_length=max_message_length)

            subquery: str = self.vllm_client.call_chat(messages=messages, temperature=subquery_temp, **kwargs)
            subquery = _normalize_subquery(subquery)

            if subquery in past_subqueries:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)
            
            new_path = RagPath(
                query=query,
                past_subqueries=past_subqueries,
                past_subanswers=past_subanswers,
                past_doc_ids=past_doc_ids
            )
            
            # Score the new path
            score = self._eval_state_without_answer(
                path=new_path,
                num_rollouts=3,
                task_desc=task_desc,
                max_path_length=3,
                temperature=0.7,
                max_message_length=max_message_length
            )
            scores.append(score)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
            scores=scores
        )

    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate the final answer for the given query and task description.
        This function uses the subqueries and their corresponding answers to generate a final answer.

        Args:
            corag_sample (RagPath): A path object containing the query, subqueries, answers, and document IDs.
            task_desc (str): Task description to be used in the prompt
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.
            documents (Optional[List[str]], optional): Document texts for each intermediate subquery. Defaults to None.

        Returns:
            str: The final answer generated by the LLM.
        """
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        return self.vllm_client.call_chat(messages=messages, **kwargs)

    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        """Truncate long messages to fit within the maximum length.
        This function modifies the messages in place to ensure that their content does not exceed the specified maximum length.

        Args:
            messages (List[Dict]): List of messages to be truncated.
            max_length (int): Maximum length in tokens for each message.
        """
        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]

    def sample_subqueries(self, query: str, task_desc: str, n: int = 10, max_message_length: int = 4096, **kwargs) -> List[str]:
        """Sample subqueries for the given query and task description.
        This function generates subqueries based on the provided query and task description.

        Args:
            query (str): Query to be answered
            task_desc (str): Task description to be used in the prompt
            n (int, optional): Number of paths to sample. Defaults to 10.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.

        Returns:
            List[str]: A list of sampled subqueries.
        """
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        completion: ChatCompletion = self.vllm_client.call_chat(messages=messages, return_str=False, n=int(1.5 * n), **kwargs)
        subqueries: List[str] = [_normalize_subquery(c.message.content) for c in completion.choices]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List]:
        """Get the subanswer and document IDs for the given subquery.
        This function uses the E5 retriever to find relevant documents and then generates an answer based on those documents.

        Args:
            subquery (str): Subquery to be answered
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.

        Returns:
            Tuple[str, List]: _description_
        """
        retriever_results: List[Dict] = search_by_http(query=subquery, host=self.e5_ip)
        doc_ids: List[str] = [res['doc_id'] for res in retriever_results]
        documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]

        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer: str = self.vllm_client.call_chat(messages=messages, temperature=0., max_tokens=128)
        return subanswer, doc_ids

    def tree_search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        """Tree search for the given query and task description.
        This function generates subqueries and their corresponding answers iteratively.

        Args:
            query (str): Query to be answered
            task_desc (str): Task description to be used in the prompt
            max_path_length (int, optional): Maximum path length. Defaults to 3.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.
            temperature (float, optional): Temperature for the LLM. Used for randomness. Defaults to 0.7.
            expand_size (int, optional): Number of subqueries to sample at each step. Defaults to 4.
            num_rollouts (int, optional): Depth of rollouts to estimate scores for each subquery. Defaults to 2.
            beam_size (int, optional): Number of subqueries to keep in the queue. Defaults to 1.

        Returns:
            RagPath: A path object containing the query, subqueries, answers, and document IDs.
        """
        return self._search(
            query=query, task_desc=task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            **kwargs
    ) -> RagPath:
        """Best of N sampling for the given query and task description.

        Args:
            query (str): Query to be answered
            task_desc (str): Task description to be used in the prompt
            max_path_length (int, optional): Maximum path length. Defaults to 3.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.
            temperature (float, optional): LLM temperature. Used to inject randomness. Defaults to 0.7.
            n (int, optional): Number of paths to sample.  to 4.

        Returns:
            RagPath: _description_
        """
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path: RagPath = self.sample_path(
                query=query, task_desc=task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        """Tree search for the given query and task description.
        This function generates subqueries and their corresponding answers iteratively.

        Args:
            query (str): Query to be answered
            task_desc (str): Task description to be used in the prompt
            max_path_length (int, optional): Maximum path length. Defaults to 3.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.
            temperature (float, optional): LLM temperature. Used to inject randomness. Defaults to 0.7.
            expand_size (int, optional): Number of subquery expansions. Defaults to 4.
            num_rollouts (int, optional): Additional depth to rollout for estimating path scores. Defaults to 2.
            beam_size (int, optional): Number of nodes to retain. Defaults to 1.

        Returns:
            RagPath: A path object containing the query, subqueries, answers, and document IDs.
        """
        candidates: List[RagPath] = [RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])]
        for step in range(max_path_length):
            new_candidates: List[RagPath] = []
            for candidate in candidates:
                new_subqueries: List[str] = self.sample_subqueries(
                    query=query, task_desc=task_desc,
                    past_subqueries=deepcopy(candidate.past_subqueries),
                    past_subanswers=deepcopy(candidate.past_subanswers),
                    n=expand_size, temperature=temperature, max_message_length=max_message_length
                )
                for subquery in new_subqueries:
                    new_candidate: RagPath = deepcopy(candidate)
                    new_candidate.past_subqueries.append(subquery)
                    subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidates.append(new_candidate)

            if len(new_candidates) > beam_size:
                scores: List[float] = []
                for path in new_candidates:
                    score = self._eval_state_without_answer(
                        path, num_rollouts=num_rollouts,
                        task_desc=task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        """Evaluate the current path by generating an answer for the subquery and calculating the log probability of the answer.

        Args:
            current_path (RagPath): Current path to be evaluated
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.

        Returns:
            float: Conditional log probability of the answer generated for the subquery of "No relevant information found".
        """
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: ChatCompletion = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=1,
            extra_body={
                "prompt_logprobs": 1
            }
        )
        answer_logprobs: List[float] = parse_answer_logprobs(response)

        return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        """Evaluate the current path without using the answer.
        This function uses rollouts to estimate the score of the current path.

        Args:
            path (RagPath): Current path to be evaluated
            num_rollouts (int): Number of rollouts to perform
            task_desc (str): Task description to be used in the prompt
            max_path_length (int, optional): Maximum path length. Defaults to 3.
            temperature (float, optional): LLM temperature. Used to inject randomness. Defaults to 0.7.
            max_message_length (int, optional): Maximum message length in tokens. Defaults to 4096.

        Returns:
            float: Estimated score of the current path based on rollouts.
        """
        assert len(path.past_subqueries) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries) + 2), # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        # TODO: should we use the min score instead?
        return sum(scores) / len(scores)
