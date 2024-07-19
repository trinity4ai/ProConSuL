from abc import ABC
from datasets import Dataset

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


class DatasetExtension(ABC):
    def __init__(self, dataset: Dataset):
        self.base_dataset = dataset
        self.dict_dataset = {row[DatasetScheme.ID_KEY]: row for row in dataset}


class MutableDataset(DatasetExtension):
    _generated_docs = dict()
    _generated_contexts = dict()

    def set_doc(self, node_id: str, doc: str):
        self._generated_docs[node_id] = doc

    def get_doc(self, node_id: str):
        if node_id in self._generated_docs:
            return self._generated_docs[node_id]
        else:
            return ''

    def set_context(self, node_id: str, context: str):
        self._generated_contexts[node_id] = context

    def get_context(self, node_id: str):
        if node_id in self._generated_contexts:
            return self._generated_contexts[node_id]
        else:
            return ''
