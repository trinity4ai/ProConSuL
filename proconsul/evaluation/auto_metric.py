import abc
from typing import Dict


class AutoMetric(abc.ABC):
    def get_name(self) -> str:
        raise NotImplementedError('abstract method')

    def compute(self, doc: str, other_columns: Dict) -> float:
        raise NotImplementedError('abstract method')
