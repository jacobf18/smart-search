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

    def tree_search_with_phs(
        self, query: str, 
        task_desc: str,
        max_path_length: int = 3,
        max_message_length: int = 4096,
        temperature: float = 0.7,
        expand_size: int = 4,
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> RagPath:
        root_path = RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])
        root_node = TreeNode(path=root_path, logprob=0.0)

        open_list = []
        heapq.heappush(open_list, root_node)

        while open_list:
            node = heapq.heappop(open_list)
            current_path = node.path

            if self._is_solution(current_path, task_desc, confidence_threshold, max_message_length):
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

                g = len(new_path.past_subqueries)
                p = node.logprob + sub_logprob
                h = self._estimate_heuristic_llm(new_path, task_desc, max_message_length, **kwargs)
                levin_cost = math.log(g + h + 1e-5) - p

                new_node = TreeNode(path=new_path, logprob=p, parent=node)
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

    def _is_solution(self, path: RagPath, task_desc: str, confidence_threshold: float, max_message_length: int) -> bool:
        messages = get_generate_final_answer_prompt(
            query=path.query,
            past_subqueries=path.past_subqueries,
            past_subanswers=path.past_subanswers,
            task_desc=task_desc
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: ChatCompletion = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=1,
            extra_body={"prompt_logprobs": 1}
        )

        logprobs = response.choices[0].logprobs.token_logprobs or []
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else float('-inf')
        return math.exp(avg_logprob) >= confidence_threshold

    def _estimate_heuristic_llm(self, path: RagPath, task_desc: str, max_message_length: int, **kwargs) -> float:
        # Ask the LLM how many subqueries it expects are remaining
        messages = [
            {
                "role": "system",
                "content": "Given the following subqueries and answers so far, estimate how many more subqueries are needed to fully answer the original query. Respond with a single integer."
            },
            {
                "role": "user",
                "content": f"Query: {path.query}\n" +
                           "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(path.past_subqueries, path.past_subanswers)))
            }
        ]
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