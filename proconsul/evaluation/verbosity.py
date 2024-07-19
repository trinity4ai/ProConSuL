from typing import Dict

from proconsul.evaluation.auto_metric import AutoMetric


class Verbosity(AutoMetric):

    def __init__(self, name: str = "Verbosity"):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def compute(self, doc: str, other_columns: Dict) -> float:
        l = len(doc)
        len_score = (l > 250) * ((l < 550) * (l - 250) / 300 + (l > 550))

        exact_rep_score = 0
        for substr_len in [30, 50]:
            substrs = [doc[i: i + substr_len] for i in range(len(doc) - substr_len)]
            substrs = [s for s in substrs if len(s.split()) > 3]
            if len(substrs) != len(set(substrs)):
                exact_rep_score = max(exact_rep_score, 0.4 if substr_len == 30 else 1)
        return min(1.0, len_score + exact_rep_score)
