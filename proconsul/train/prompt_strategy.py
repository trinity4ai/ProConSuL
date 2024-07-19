from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional

from datasets import Dataset

from proconsul.data_processing.utils_data_processing import get_context
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.dataset_extension import DatasetExtension, MutableDataset


class PromptStrategy(ABC):
    def __init__(self, instruction: str) -> None:
        self._instruction = instruction

    @abstractmethod
    def get_prompt(
            self,
            data_point: Dict,
            test: bool,
            dataset: Optional[Union[List[Dict], Dataset, DatasetExtension]] = None
    ) -> str:
        raise NotImplementedError('Abstract method')


class BasePromptStrategy(PromptStrategy):
    def __init__(
            self,
            instruction: str,
    ) -> None:
        super().__init__(instruction)

    def get_prompt(self,
                   data_point: Dict,
                   test: bool,
                   dataset: Optional[Union[List[Dict], Dataset]] = None) -> str:
        code = data_point[DatasetScheme.CODE_KEY]
        doc = '' if test else data_point[DatasetScheme.DOC_KEY]
        return self._instruction.format(code=code, doc=doc)


class PromptStrategyRecursive(PromptStrategy):
    def __init__(self, instruction: str, context_separator: str, context_scheme: str) -> None:
        super().__init__(instruction)
        self._context_separator = context_separator
        self._context_scheme = context_scheme

    def get_prompt(
            self,
            data_point: Dict,
            test: bool,
            data: Optional[Union[List[Dict], Dataset, DatasetExtension]] = None,  # Must be MutableDataset
    ) -> str:
        if not isinstance(data, MutableDataset):
            raise TypeError('Must have a mutable dataset for recursive generation!')
        docstrings = {
            node: data.get_doc(node) for node in data_point[DatasetScheme.TO_KEY] if node in data.dict_dataset
        }
        context = get_context(
            data.dict_dataset, data_point[DatasetScheme.ID_KEY], data_point, docstrings,
            self._context_separator, self._context_scheme, True, True
        )
        data.set_context(data_point[DatasetScheme.ID_KEY], context)
        code = data_point[DatasetScheme.CODE_KEY]
        doc = '' if test else data_point[DatasetScheme.DOC_KEY]
        return self._instruction.format(context=context, code=code, doc=doc).strip()


class PromptStrategyWithContext(PromptStrategy):
    CONTEXT_KEY = DatasetScheme.CONTEXT_KEY

    def __init__(self, instruction: str, **kwargs) -> None:
        super().__init__(instruction)
        if 'target_doc_key' in kwargs:
            self.target_doc_key = kwargs['target_doc_key']
        else:
            self.target_doc_key = DatasetScheme.DOC_KEY

    def get_prompt(
            self,
            data_point: Dict,
            test: bool,
            dataset: Optional[Union[List[Dict], Dataset]] = None
    ) -> str:
        context = data_point[self.CONTEXT_KEY]
        code = data_point[DatasetScheme.CODE_KEY]
        if test:
            instr = self._instruction.format(lever='', num_doc_words=20, context=context, code=code, doc='').strip()
            return instr
        else:
            doc = data_point[self.target_doc_key]
            instr = self._instruction.format(lever='',
                                             num_doc_words=len(doc.split()),
                                             context=context,
                                             code=code,
                                             doc=doc).strip()
            return instr


class PromptStrategyForSynthesis(PromptStrategy):
    def __init__(self, instruction: str, **kwargs) -> None:
        super().__init__(instruction)

    def get_prompt(
            self,
            data_point: Dict,
            test: bool = None,
            dataset: Optional[Union[List[Dict], Dataset]] = None
    ) -> str:
        code = data_point[DatasetScheme.CODE_KEY]
        doc = data_point[DatasetScheme.DOC_KEY]
        instr = self._instruction.format(code=code, doc=doc)
        return instr
