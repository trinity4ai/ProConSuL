from typing import Any, Dict, List, Union

from transformers import BatchEncoding, PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.dataset_extension import MutableDataset
from proconsul.train.prompt_strategy import PromptStrategy
from proconsul.train.utils import generate_and_tokenize_prompt


class CollatorForCompletionWithEos(DataCollatorForCompletionOnlyLM):
    EOS_TOKEN_ID = 2  # Works for Llama, might not work for any other model

    def torch_call(self, examples):
        for i in range(len(examples)):
            if DatasetScheme.LABELS_KEY in examples[i]:
                del examples[i][DatasetScheme.LABELS_KEY]
        batch = super().torch_call(examples)
        for i in range(len(batch[DatasetScheme.LABELS_KEY])):
            batch[DatasetScheme.LABELS_KEY][i][-1] = self.EOS_TOKEN_ID
        return batch


class RecursiveDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(
            self,
            data: MutableDataset,
            tokenizer: PreTrainedTokenizer,
            prompt_strategy: PromptStrategy,
            max_length: int,
            response_template: Union[str, List[int]],
            instruction_template: Union[str, List[int]] = None,
            mlm: bool = False,
            ignore_index: int = -100,
            **kwargs,
    ) -> None:
        super().__init__(
            response_template, instruction_template, mlm=mlm, ignore_index=ignore_index, tokenizer=tokenizer, **kwargs
        )
        self._data = data
        self._prompt_strategy = prompt_strategy
        self._max_length = max_length

    def torch_call(self, examples: List[Dict[str, Any] | BatchEncoding]) -> Dict[str, Any]:
        prompts = [
            generate_and_tokenize_prompt(
                data_point, self.tokenizer, self._max_length, self._prompt_strategy, test=True, dataset=self._data
            ) for data_point in examples
        ]
        ids = [data_point[DatasetScheme.ID_KEY] for data_point in examples]
        return super().torch_call(prompts) | {DatasetScheme.DATASET_IDS_KEY: ids}
