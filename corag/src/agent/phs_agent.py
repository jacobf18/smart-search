import heapq
import math
from copy import deepcopy
from turtle import pen
from typing import Optional, List, Dict

from agent.agent_utils import RagPath
from openai.types.chat import ChatCompletion
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt

from agent.corag_agent import CoRagAgent
from agent.corag_agent import _normalize_subquery

from vllm_client_local import VllmClient
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import numpy as np

class TreeNode:
    def __init__(self, path: RagPath, logprob: float, parent: Optional["TreeNode"] = None):
        self.path = deepcopy(path)
        self.logprob = logprob  # cumulative log probability of subqueries
        self.depth = len(path.past_subqueries)
        self.parent = parent
        self.children = []
        self.levin_cost = 0.0
        self.penalty = np.inf

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
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = 0.95
        
    def filter_similar_subqueries(self, choices):
        filtered_indices = []
        subqueries = [_normalize_subquery(choice.message.content) for choice in choices]
        embeddings = self.embedding_model.encode(subqueries, show_progress_bar=False)
        
        # cluster the embeddings by cosine similarity of the embeddings
        for i in range(len(embeddings)):
            is_similar = False
            for j in range(i):
                sim = cosine_similarity(
                    embeddings[i].reshape(1,-1), 
                    embeddings[j].reshape(1,-1)
                    # dim=0
                ).item()
                if sim > self.similarity_threshold:
                    is_similar = True
                    break
            if not is_similar:
                filtered_indices.append(i)
        return [choices[i] for i in filtered_indices]
    
    def tree_to_dict(self, node):
        # Create a shallow copy of the node's __dict__
        node_dict = node.__dict__.copy()
        
        # Recursively convert children
        if 'children' in node_dict and isinstance(node_dict['children'], list):
            node_dict['children'] = [self.tree_to_dict(child) for child in node_dict['children']]
        
        return node_dict

    def tree_search(
        self, query: str, 
        task_desc: str,
        max_path_length: int = 3,
        max_message_length: int = 4096,
        temperature: float = 0.7,
        expand_size: int = 4,
        max_tree_size = 100,
        **kwargs
    ) -> RagPath:
        root_path = RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[])
        root_node = TreeNode(path=root_path, logprob=0.0)
        all_nodes = []
        open_list = []
        heapq.heappush(open_list, root_node)
        explored_num = 0
        while open_list and explored_num < max_tree_size:
            # print(f"Explored nodes: {explored_num}, Open list size: {len(open_list)}")
            explored_num += 1
            node = heapq.heappop(open_list)
            current_path = node.path
            
            # if self._is_solution(current_path, task_desc, max_message_length):
            #     return current_path

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
                # extra_body={"prompt_logprobs": 1},
                logprobs=True,
                temperature=temperature,
                **kwargs
            )
            
            filtered_choices = self.filter_similar_subqueries(completion.choices)
            
            new_paths = []
            scores = []

            for choice in filtered_choices:
                subquery = _normalize_subquery(choice.message.content)
                # if subquery in current_path.past_subqueries:
                #     continue
                
                subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                    subquery=subquery, max_message_length=max_message_length
                )
                
                # token_logprobs = [c.logprob for c in choice.logprobs.content]
                
                # sub_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
                
                new_path = RagPath(
                    query=query,
                    past_subqueries=current_path.past_subqueries + [subquery],
                    past_subanswers=current_path.past_subanswers + [subanswer],
                    past_doc_ids=current_path.past_doc_ids + [doc_ids]
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
                
                new_paths.append(new_path)
                scores.append(score)
                
            logprobs = np.log(softmax(-np.array(scores)))
            
            for new_path, logprob, score in zip(new_paths, logprobs, scores):
                running_cost = len(new_path.past_subqueries) # number of nodes (g(n) in the paper)
                # policy_logprob = node.logprob + sub_logprob # cumulative log probability (log(pi(n)) in the paper)
                policy_logprob = logprob + node.logprob
                # heuristic_cost = self._estimate_heuristic_llm(new_path, task_desc, max_message_length, **kwargs) # h(n) in the paper
                # heuristic_cost = max_path_length - len(new_path.past_subqueries) # h(n) in the paper
                
                heuristic_cost = 0
                levin_cost = math.log(running_cost + heuristic_cost + 1e-5) - policy_logprob

                new_node = TreeNode(path=new_path, logprob=policy_logprob, parent=node)
                node.children.append(new_node)
                all_nodes.append(new_node)
                new_node.levin_cost = levin_cost
                heapq.heappush(open_list, new_node)
                new_node.penalty = score
                
        # Loop over all leaf nodes and choose the best one according to _eval_single_path
        best_node = None
        best_score = float('inf')
        for node in all_nodes:
            if len(node.children) > 0:
                continue
            path = node.path
            
            if len(path.past_subqueries) > len(path.past_subanswers):
                # If the path is not complete, complete it
                subanswer, doc_ids = self._get_subanswer_and_doc_ids(
                    subquery=path.past_subqueries[-1], max_message_length=max_message_length
                )
                
                path.past_subanswers.append(subanswer)
                path.past_doc_ids.append(doc_ids)
            
            score = self._eval_single_path(path, max_message_length=max_message_length)
            if score < best_score:
                best_score = score
                best_node = node
        
        # Create a dictionary representation of the entire tree
        # tree_dict = self.tree_to_dict(root_node)

        if best_node is not None:
            return best_node.path
        # If no solution was found, return the root path
        # This is a fallback and should not happen in normal circumstances.
        # logger.warning(f"Did not find a solution within {max_tree_size} nodes. Returning the root path.")
        # This should never happen, but just in case
        return self.root_path

    def _is_solution(self, path: RagPath, task_desc: str, max_message_length: int) -> bool:
        log_prob = self._eval_single_path(
            path,
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
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(path.past_subqueries, path.past_subanswers)],
        )
        # messages.append({'role': 'user', 'content': 'How many more subqueries are needed to fully answer the original query. Respond with a single integer.'})
        messages.append({'role': 'user', 
                         'content': f'What additional subqueries should be asked to fully answer the original query: "{path.query}". Separate subqueries with question marks "?".'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: ChatCompletion = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            # max_tokens=5,
            **kwargs
        )

        text = response.choices[0].message.content.strip()
        
        est_remaining = max(1,len(text.split("?")))

        # response: ChatCompletion = self.vllm_client.call_chat(
        #     messages=messages,
        #     return_str=False,
        #     max_tokens=5,
        #     **kwargs
        # )

        # # idk how to ensure its an int, this was an attempt
        # try:
        #     est_remaining = int(text.split()[0])
        # except Exception:
        #     est_remaining = max(1, 3 - len(path.past_subqueries))  # fallback

        return max(0, est_remaining)