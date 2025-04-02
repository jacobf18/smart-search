import glob
from re import split
import torch

from typing import List, Dict, Tuple
from datasets import Dataset

from search.simple_encoder import SimpleEncoder
from data_utils import load_corpus
from logger_config import logger


def _get_all_shards_path(index_dir: str) -> List[str]:
    path_list = glob.glob('{}/*-shard-*.pt'.format(index_dir))
    assert len(path_list) > 0

    def _parse_shard_idx(p: str) -> int:
        return int(p.split('-shard-')[1].split('.')[0])

    path_list = sorted(path_list, key=lambda path: _parse_shard_idx(path))
    logger.info('Embeddings path list: {}'.format(path_list))
    return path_list


class E5Searcher:

    def __init__(
            self, index_dir: str,
            model_name_or_path: str = 'intfloat/e5-large-v2',
            verbose: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.verbose = verbose

        n_gpus: int = torch.cuda.device_count()
        self.gpu_ids: List[int] = list(range(n_gpus))

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path=self.model_name_or_path,
            max_length=64,
        )
        self.encoder.to(self.gpu_ids[-1])

        shard_paths = _get_all_shards_path(self.index_dir)
        logger.info(f'Found {len(shard_paths)} shards in {self.index_dir}')
        
        indices1 = [0]

        cur = 0
        for i in range(0, 20):
            if (i + 1) % 5 == 0:
                cur += 459_760
            else:
                cur += 1_000_000
            indices1.append(cur)

        # Create 2-tuples of the form (start, end)
        ranges1 = [(indices1[i], indices1[i + 1], indices1[i + 1] - indices1[i]) for i in range(0, 20)]

        indices2 = [0]

        cur = 0
        for i in range(0, 20):
            if (i + 1) % 5 == 0:
                cur += 459_759
            else:
                cur += 1_000_000
            indices2.append(cur)

        # Create 2-tuples of the form (start, end)
        ranges2 = [(indices2[i], indices2[i + 1], indices2[i + 1] - indices2[i]) for i in range(0, 20)]
        
        # split1 = torch.zeros((indices[19], 1024), dtype=torch.float16, device='cuda:0')
        # split2 = torch.zeros((indices[-1], 1024), dtype=torch.float16, device='cuda:1')
        
        split_embeddings = [
            torch.zeros((indices1[-1] + 1, 1024), dtype=torch.float16, device='cuda:0'),
            torch.zeros((indices2[-1], 1024), dtype=torch.float16, device='cuda:1')
        ]
        
        # turn off gradients
        for i in range(2):
            split_embeddings[i].requires_grad = False
        
        for i in range(20):
            logger.info(f'Loading shard {i}')
            shard_tensor = torch.load(shard_paths[i], map_location=torch.device("cuda:0"), weights_only=True)
            split_embeddings[0][ranges1[i][0]:ranges1[i][1]] = shard_tensor
            
            del shard_tensor
            torch.cuda.empty_cache()
        
        for i in range(20, 39):
            logger.info(f'Loading shard {i}')
            
            shard_tensor = torch.load(shard_paths[i], map_location=torch.device("cuda:1"), weights_only=True)
            
            split_embeddings[1][ranges2[i-20][0]:ranges2[i-20][1]] = shard_tensor
            
            del shard_tensor
            torch.cuda.empty_cache()
        
        split_embeddings[1][ranges2[-1][0]:] = torch.load(shard_paths[39], map_location=torch.device("cuda:0"), weights_only=True)
        
        # # all_embeddings: torch.Tensor = torch.cat(
        #     # [torch.load(p, weights_only=True, map_location=lambda storage, loc: storage) for p in shard_paths], dim=0
        # # )
        # logger.info(f'Load {all_embeddings.shape[0]} embeddings from {self.index_dir}')

        # split_embeddings = torch.chunk(all_embeddings, len(self.gpu_ids))
        self.embeddings: List[torch.Tensor] = split_embeddings

        self.corpus: Dataset = load_corpus()

    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        query_embed: torch.Tensor = self.encoder.encode_queries(queries).to(dtype=self.embeddings[0].dtype)

        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(queries)):
            results: List[Dict] = []
            for score, idx in zip(batch_sorted_score[query_idx], batch_sorted_indices[query_idx]):
                results.append({
                    'doc_id': int(idx.item()),
                    'score': score.item(),
                })

                if self.verbose:
                    results[-1].update(self.corpus[int(idx.item())])
            results_list.append(results)

        return results_list

    def _compute_topk(self, query_embed: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_score_list: List[torch.Tensor] = []
        batch_sorted_indices_list: List[torch.Tensor] = []

        idx_offset = 0
        for i in range(len(self.embeddings)):
            query_embed = query_embed.to(self.embeddings[i].device)
            score = torch.mm(query_embed, self.embeddings[i].t())
            sorted_score, sorted_indices = torch.topk(score, k=k, dim=-1, largest=True)

            sorted_indices += idx_offset
            batch_score_list.append(sorted_score.cpu())
            batch_sorted_indices_list.append(sorted_indices.cpu())
            idx_offset += self.embeddings[i].shape[0]

        batch_score = torch.cat(batch_score_list, dim=1)
        batch_sorted_indices = torch.cat(batch_sorted_indices_list, dim=1)
        # only keep the top k results based on batch_score
        batch_score, top_indices = torch.topk(batch_score, k=k, dim=-1, largest=True)
        batch_sorted_indices = torch.gather(batch_sorted_indices, dim=1, index=top_indices)

        return batch_score, batch_sorted_indices
