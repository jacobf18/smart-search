from typing import List, Dict, Union

from inference import qa_utils
from logger_config import logger

def compute_em_and_f1(labels: List[List[str]], preds: List[str]) -> Dict[str, float]:
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    em, f1, em_scores, f1_scores = qa_utils.qa_metrics(labels, preds)
    return {'em': round(em, 3), 'f1': round(f1, 3)}

def compute_em_and_f1_multi(labels: List[List[str]], preds: List[List[str]]) -> Dict[str, float]:
    """Compute EM and F1 when each prediction is a list of multiple generated answers.
    We take the max over the set of predictions per sample.
    """
    em_scores, f1_scores = [], []
    for label_list, pred_list in zip(labels, preds):
        norm_label_list = [qa_utils.normalize_squad(t) for t in label_list]
        norm_preds = [qa_utils.normalize_squad(p) for p in pred_list]
        max_em, max_f1 = 0.0, 0.0
        for p in norm_preds:
            em, f1 = qa_utils.qa_metrics([norm_label_list], [p])[0:2]
            max_em = max(max_em, em)
            max_f1 = max(max_f1, f1)
        em_scores.append(max_em)
        f1_scores.append(max_f1)
    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)
    return {'em': round(em, 3), 'f1': round(f1, 3)}

def compute_metrics_dict(
    labels: Union[List[str], List[List[str]]],
    preds: Union[List[str], List[List[str]]],
    eval_metrics: str,
    is_multiple_predictions: bool = False
) -> Dict[str, float]:
    metric_names: List[str] = eval_metrics.split(',')
    metric_dict: Dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in ['em_and_f1', 'dpr']:
            if is_multiple_predictions:
                metric_dict.update(compute_em_and_f1_multi(labels, preds))
            else:
                metric_dict.update(compute_em_and_f1(labels, preds))
        elif metric_name == 'kilt':
            logger.warning('KILT metric requires run separate script')
        else:
            raise ValueError(f'Invalid metric: {metric_name}')

    return metric_dict
