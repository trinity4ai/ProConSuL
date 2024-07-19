from typing import Any, Dict, List

from omegaconf import DictConfig

from proconsul.evaluation.auto_metric import AutoMetric


def calc_auto_metrics(
        data: Dict[str, Dict[str, Any]],
        eval_config: DictConfig,
        metrics: List[AutoMetric],
) -> Dict[str, Dict[str, float]]:
    """ data is expected to have specific fields, e.g. 'doc_to_eval'. """
    auto_metrics = dict()
    doc_to_eval_key = eval_config.doc_to_eval_key

    for key, point in data.items():
        auto_metrics[key] = dict()
        for metric in metrics:
            summ = point[doc_to_eval_key]
            auto_metrics[key][metric.get_name()] = metric.compute(summ, point)

    return auto_metrics
