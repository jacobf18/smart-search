import heapq
import math
from copy import deepcopy
from typing import Optional, List, Dict

from agent.agent_utils import RagPath
from openai.types.chat import ChatCompletion
from prompts import (
    get_generate_subquery_prompt,
    get_generate_final_answer_prompt
)

from agent.corag_agent import CoRagAgent
from agent.corag_agent import _normalize_subquery

from vllm_client import VllmClient
from datasets import Dataset

class TreeNode:
    def __init__(self, path: RagPath, logprob: float, parent: Optional["TreeNode"] = None):
        self.path = deepcopy(path)
        self.logprob = logprob  # cumulative log probability of subqueries
        self.depth = len(path.past_subqueries)
        self.parent = parent
        self.levin_cost = 0.0

    def __lt__(self, other):
        return self.levin_cost < other.levin_cost
    
class CoRagAgentWithPHS(CoRagAgent):
    
    def __init__(self, vllm_client: VllmClient, corpus: Dataset, e5_ip: str, vllm_ip: str, 
                 confidence_threshold: float = 0.5):
        """Initializes the CoRagAgentWithPHS class.
        This class is a specialized version of the CoRagAgent that implements a tree search algorithm
        called 'Policy-guided heuristic search' (PHS) to quickly find a good path for answering a query.

        Args:
            vllm_client (VllmClient): VLLM client for answering queries.
            corpus (Dataset): Dataset containing the documents to be searched.
            e5_ip (str): IP address of the E5 server.
            vllm_ip (str): IP address of the VLLM server.
            confidence_threshold (float, optional): Confidence threshold to determine if completed. Defaults to 0.5.
        """
        super().__init__(vllm_client, corpus, e5_ip, vllm_ip)
        self.confidence_threshold = confidence_threshold

    def tree_search(
        self, query: str, 
        task_desc: str,
        max_path_length: int = 3,
        max_message_length: int = 4096,
        temperature: float = 0.7,
        expand_size: int = 4,
        **kwargs
    ) -> RagPath:
        root_path = RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])
        root_node = TreeNode(path=root_path, logprob=0.0)

        open_list = []
        heapq.heappush(open_list, root_node)

        while open_list:
            node = heapq.heappop(open_list)
            current_path = node.path

            if self._is_solution(current_path, task_desc, max_message_length):
                return current_path

            messages = get_generate_subquery_prompt(
                query=query,
                past_subqueries=current_path.past_subqueries,
                past_subanswers=current_path.past_subanswers,
                task_desc=task_desc
            )
            self._truncate_long_messages(messages, max_length=max_message_length)

            completion: ChatCompletion = self.vllm_client.call_chat(
                messages=messages,
                return_str=False,
                n=expand_size,
                extra_body={"prompt_logprobs": 1},
                temperature=temperature,
                **kwargs
            )

            for choice in completion.choices:
                subquery = _normalize_subquery(choice.message.content)
                if subquery in current_path.past_subqueries:
                    continue

                token_logprobs = choice.logprobs.token_logprobs or []
                sub_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)

                subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                    subquery=subquery, max_message_length=max_message_length
                )

                new_path = RagPath(
                    query=query,
                    past_subqueries=current_path.past_subqueries + [subquery],
                    past_subanswers=current_path.past_subanswers + [subanswer],
                    past_doc_ids=current_path.past_doc_ids + [doc_ids]
                )

                running_cost = len(new_path.past_subqueries) # number of nodes (g(n) in the paper)
                policy_logprob = node.logprob + sub_logprob # cumulative log probability (log(pi(n)) in the paper)
                heuristic_cost = self._estimate_heuristic_llm(new_path, task_desc, max_message_length, **kwargs) # h(n) in the paper
                levin_cost = math.log(running_cost + heuristic_cost + 1e-5) - policy_logprob

                new_node = TreeNode(path=new_path, logprob=policy_logprob, parent=node)
                new_node.levin_cost = levin_cost
                heapq.heappush(open_list, new_node)

        # use this only if nothing worked idk?? can use logger instead
        return self.sample_path(
                    query=query,
                    task_desc=task_desc,
                    max_path_length=max_path_length,
                    max_message_length=max_message_length,
                    temperature=temperature,
                    **kwargs
                )

    def _is_solution(self, path: RagPath, task_desc: str, max_message_length: int) -> bool:
        log_prob = self._eval_single_path(
            path,
            task_desc=task_desc,
            max_message_length=max_message_length
        )
        # This is the log probability of the string 'No relevant information found'
        # So, if the log probability of the path is less than this, we consider it a solution.
        # This is a heuristic, and the threshold can be adjusted based on the model's behavior.
        return log_prob < math.log(self.confidence_threshold)

    def _estimate_heuristic_llm(self, path: RagPath, task_desc: str, max_message_length: int, **kwargs) -> int:
        """Heuristic function to estimate the number of remaining subqueries.
        This function uses the LLM to predict how many subqueries are needed to fully answer the original query.

        Args:
            path (RagPath): RagPath to the current node in the search tree
            task_desc (str): Task description
            max_message_length (int): Maximum message length for the LLM
            **kwargs: Additional arguments for the LLM call

        Returns:
            int: Estimated number of remaining subqueries
        """
        # Ask the LLM how many subqueries it expects are remaining
        messages: List[Dict] = self.get_generate_intermediate_answer_prompt(
            subquery=path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(path.past_subqueries, path.past_subanswers)],
        )
        messages.append({'role': 'user', 'content': 'How many more subqueries are needed to fully answer the original query. Respond with a single integer.'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: ChatCompletion = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=5,
            **kwargs
        )

        text = response.choices[0].message.content.strip()

        # idk how to ensure its an int, this was an attempt
        try:
            est_remaining = int(text.split()[0])
        except Exception:
            est_remaining = max(1, 3 - len(path.past_subqueries))  # fallback

        return max(0, est_remaining)