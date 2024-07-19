from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra.utils
from datasets import Dataset
from omegaconf import DictConfig
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.prompt_strategy import PromptStrategy
from proconsul.train.utils import get_prediction_output, process_dataset, load_model
from proconsul.train.dataset_extension import MutableDataset
from proconsul.train.custom_samplers import RecursiveSampler, DynamicBatchSampler
from proconsul.train.custom_collators import RecursiveDataCollator

from abc import ABC


def sort_dataset(dataset: Dataset) -> Dataset:
    lengths = list(map(lambda x: -len(x), dataset[DatasetScheme.INPUT_IDS_KEY]))
    dataset = dataset.add_column("length", lengths)
    dataset = dataset.sort("length")
    return dataset.remove_columns("length")


class InferenceEngine(ABC):
    def generate(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
        response_template: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError("InferenceEngine.generate is not implemented")

    def generate_single(
        self,
        prompt: str,
    ) -> str:
        raise NotImplementedError("InferenceEngine.generate_single is not implemented")


class HFBaseEngine(InferenceEngine):
    def __init__(
        self,
        generation_config_dict: DictConfig,
        checkpoint: str,
        bits_and_bytes_config_dict: DictConfig,
        batch_size: int,
        adapter_checkpoint: Optional[str] = None,
        test: bool = False,
    ):
        self.model, self.tokenizer = load_model(
            checkpoint=checkpoint, bits_and_bytes_config_dict=bits_and_bytes_config_dict, test=test
        )
        if adapter_checkpoint is not None:
            self.model.load_adapter(adapter_checkpoint)
        self.model.config.pad_token_id = -100
        self.model.eval()

        # init generation_config
        self.generation_config = hydra.utils.instantiate(generation_config_dict)
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id

        self.batch_size = batch_size

    def generate(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
        response_template: str,
    ) -> Dict[str, Any]:
        data, dataloader = self._prepare_func(
            data=data,
            max_length=max_length,
            prompt_strategy=prompt_strategy,
            response_template=response_template,
            batch_size=self.batch_size,
        )

        dict_dataset, output_docs = self._gen_func(data, dataloader)
        return get_prediction_output(dict_dataset, output_docs)

    def _gen_func(self, data: Dataset, dataloader: DataLoader) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        output_docs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output_docs += self._batch_predict(batch)
        output_docs = {row[DatasetScheme.ID_KEY]: pred for row, pred in zip(data, output_docs[: len(data)])}
        return {row[DatasetScheme.ID_KEY]: row for row in data}, output_docs

    def _prepare_func(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
        response_template: str,
        batch_size: int,
    ) -> Tuple[Dataset, DataLoader]:
        data = process_dataset(data, self.tokenizer, max_length, prompt_strategy, test=True)
        data = sort_dataset(data)

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=self.tokenizer, pad_to_multiple_of=8
        )
        dataloader = DataLoader(
            data.select_columns([DatasetScheme.INPUT_IDS_KEY, DatasetScheme.ATTENTION_MASK_KEY]),
            batch_sampler=DynamicBatchSampler(data, batch_size, self.generation_config.max_new_tokens),
            collate_fn=data_collator,
        )
        return data, dataloader

    def _batch_predict(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[str]:
        input_ids = batch[DatasetScheme.INPUT_IDS_KEY]
        input_ids = input_ids.to(self.model.device)
        attention_mask = batch[DatasetScheme.ATTENTION_MASK_KEY]
        batch_result = (
            self.model.generate(input_ids, generation_config=self.generation_config, attention_mask=attention_mask)
            .data.detach()
            .clone()
        )
        batch_result[batch_result == -100] = self.tokenizer.eos_token_id
        return self.tokenizer.batch_decode(list(batch_result.cpu().numpy()), skip_special_tokens=True)

    def generate_single(
        self,
        prompt: str,
    ) -> str:
        old_add_eos_token = self.tokenizer.add_eos_token
        self.tokenizer.add_eos_token = False

        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(**model_input, max_new_tokens=512)[0]
            res = self.tokenizer.decode(generated)

        self.tokenizer.add_eos_token = old_add_eos_token
        return res


class HFRecursiveEngine(HFBaseEngine):
    def _gen_func(
        self, data: MutableDataset, dataloader: DataLoader
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        output_docs = dict()
        output_contexts = dict()
        prompt_lens = dict()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                docs = self._batch_predict(batch)
                for i, doc, node_id in zip(range(len(docs)), docs, batch[DatasetScheme.DATASET_IDS_KEY]):
                    prompt_len = int(batch[DatasetScheme.PROMPT_LEN_KEY][i])
                    truncated_doc = doc[prompt_len:]
                    output_docs[node_id] = doc
                    prompt_lens[node_id] = prompt_len
                    output_contexts[node_id] = data.get_context(node_id)
                    data.set_doc(node_id, truncated_doc)
        data.dict_dataset = {
            k: ({DatasetScheme.PROMPT_LEN_KEY: prompt_lens[k], DatasetScheme.GEN_CONTEXT_KEY: output_contexts[k]} | v)
            for k, v in data.dict_dataset.items()
        }
        return data.dict_dataset, output_docs

    def _prepare_func(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
        response_template: str,
        batch_size: int,
    ) -> Tuple[Dataset, DataLoader]:
        data = MutableDataset(data)
        sampler = RecursiveSampler(data.base_dataset, batch_size)

        collator = RecursiveDataCollator(
            data, self.tokenizer, prompt_strategy, max_length, response_template, pad_to_multiple_of=8
        )
        dataloader = DataLoader(data.base_dataset, batch_sampler=sampler, collate_fn=collator)

        return data, dataloader


class VllmEngine(InferenceEngine):
    def __init__(
        self,
        generation_config_dict: DictConfig,
        checkpoint: str,
        bits_and_bytes_config_dict: DictConfig,
        max_length: int,
        adapter_checkpoint: Optional[str] = None,
        test: bool = False,
    ):
        logits_processors = []
        if "suppress_tokens" in generation_config_dict:
            logits_processors.append(self.get_supress_token_processor(generation_config_dict.suppress_tokens))
            generation_config_dict = generation_config_dict.copy()

            del generation_config_dict.suppress_tokens

        if test:
            add_eos_token = False
        else:
            add_eos_token = True

        self.sampling_params = SamplingParams(**generation_config_dict, logits_processors=logits_processors)

        self.model = LLM(
            checkpoint,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=max_length,
            tensor_parallel_size=2,
            # enable_lora=enable_lora, FIXME Necessary to use LoRA with VLLM
            # skip_tokenizer_init=True, FIXME we don't need this tokenizer
            seed=42,
        )

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=add_eos_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
        response_template: str,
    ) -> Dict[str, Any]:
        data = self._prepare_data(
            data=data,
            max_length=max_length,
            prompt_strategy=prompt_strategy,
        )
        output = self.model.generate(
            prompts=None,
            sampling_params=self.sampling_params,
            prompt_token_ids=[row[DatasetScheme.INPUT_IDS_KEY] for row in data],
        )
        output_token_ids = [
            row[DatasetScheme.INPUT_IDS_KEY] + pred.outputs[0].token_ids for row, pred in zip(data, output)
        ]
        output_token_ids = [
            [token if token != -100 else self.tokenizer.eos_token_id for token in output_token_ids_row]
            for output_token_ids_row in output_token_ids
        ]
        output_whole_text = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)
        output_docs = {row[DatasetScheme.ID_KEY]: pred for row, pred in zip(data, output_whole_text[: len(data)])}
        return get_prediction_output({row[DatasetScheme.ID_KEY]: row for row in data}, output_docs)

    def generate_single(
        self,
        prompt: str,
    ) -> str:
        return self.model.generate([prompt], self.sampling_params)[0].outputs[0].text

    @staticmethod
    def get_supress_token_processor(to_supress: List[int]) -> Callable:
        def _processor(token_ids, logits):
            for token_id in to_supress:
                logits[token_id] = -9999.999
            return logits

        return _processor

    def _prepare_data(
        self,
        data: Dataset,
        max_length: int,
        prompt_strategy: PromptStrategy,
    ) -> Dataset:
        data = process_dataset(data, self.tokenizer, max_length, prompt_strategy, test=True)
        data = sort_dataset(data)
        return data
