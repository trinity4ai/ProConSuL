from traceback import format_exc
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra.utils
import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.dataset_extension import DatasetExtension
from proconsul.train.prompt_strategy import PromptStrategy


def tokenize(
        prompt: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        set_labels: bool = False,
) -> BatchEncoding:
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if set_labels:
        result[DatasetScheme.LABELS_KEY] = result[DatasetScheme.INPUT_IDS_KEY].copy()
    result[DatasetScheme.PROMPT_LEN_KEY] = len(tokenizer.decode(
        result[DatasetScheme.INPUT_IDS_KEY], skip_special_tokens=True
    ))
    return result


def generate_and_tokenize_prompt(
        data_point: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        prompt_strategy: PromptStrategy,
        test: bool = False,
        set_labels: bool = False,
        dataset: Optional[Union[Dataset, DatasetExtension, List[Dict]]] = None,
) -> BatchEncoding:
    old_add_eos_token = tokenizer.add_eos_token
    tokenizer.add_eos_token = not test
    prompt = prompt_strategy.get_prompt(data_point, test, dataset)
    tokenized = tokenize(prompt, tokenizer, max_length, set_labels)
    tokenizer.add_eos_token = old_add_eos_token
    return tokenized


def process_dataset(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        prompt_strategy: PromptStrategy,
        test: bool = False,
        set_labels: bool = False,
        prompt_data: Optional[Union[Dataset, DatasetExtension, List[Dict]]] = None,
) -> Dataset:
    return dataset.map(
        generate_and_tokenize_prompt,
        batched=False,
        keep_in_memory=False,  # Disable caching, should be False for big datasets
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': max_length,
            'prompt_strategy': prompt_strategy,
            'test': test,
            'set_labels': set_labels,
            'dataset': prompt_data,
        },
        desc='Generating and tokenizing prompts',
    )


def load_model(
        checkpoint: str,
        bits_and_bytes_config_dict: DictConfig,
        test: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    free_memory = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f'{free_memory - 2}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    if test:
        device_map = 'auto'
        add_eos_token = False
    else:
        device_map = None
        add_eos_token = True

    quantization_config = (
        hydra.utils.instantiate(bits_and_bytes_config_dict)
        if bits_and_bytes_config_dict.get("load_in_8bit", False)
        or bits_and_bytes_config_dict.get("load_in_4bit", False)
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        max_memory=max_memory,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=add_eos_token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return model, tokenizer


def get_prediction_output(data: Dict[str, Dict[str, Any]], output_docs: Dict[str, str]) -> Dict[str, Any]:
    output = {}
    for node_id, row in data.items():
        pred = output_docs[node_id]
        prompt_len = row[DatasetScheme.PROMPT_LEN_KEY]
        data_point = {
            DatasetScheme.WHOLE_TEXT_KEY: pred,
            DatasetScheme.PROMPT_KEY: pred[:prompt_len],
            DatasetScheme.GENSUM_KEY: pred[prompt_len:],
        }
        for key, value in row.items():
            if key not in [
                DatasetScheme.INPUT_IDS_KEY,
                DatasetScheme.ATTENTION_MASK_KEY,
                DatasetScheme.LABELS_KEY,
                DatasetScheme.ID_KEY,
            ]:
                data_point[key] = value
        output[row[DatasetScheme.ID_KEY]] = data_point
    return output


def check_model(
        model: Union[PeftModel, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        prompt_strategy_dict: DictConfig,
        data: Optional[Union[Dataset, DatasetExtension, List[Dict]]] = None,
        test: bool = False,
) -> None:
    print('Checking the model:')
    print('-----------------------------------------------------------------------------------------------------')

    data_point = {
        'id': 'RepoId/ID12345',
        'code': """static bool getAArch64PBV(QualType QT, ASTContext &C) {
    QT = QT.getCanonicalType();
    unsigned Size = C.getTypeSize(QT);

    // Only scalars and complex within 16 bytes wide set PVB to true.
    if (Size != 8 && Size != 16 && Size != 32 && Size != 64 && Size != 128)
        return false;

    if (QT->isFloatingType())
        return true;

    if (QT->isIntegerType())
        return true;

    if (QT->isPointerType())
        return true;

    // TODO: Add support for complex types (section 3.1.2, item 2).

    return false;
}""",
        'context': '',
        'doc': 'Well,',
    }
    prompt_strategy = hydra.utils.instantiate(prompt_strategy_dict)

    old_add_eos_token = tokenizer.add_eos_token
    tokenizer.add_eos_token = False

    try:
        eval_prompt = prompt_strategy.get_prompt(data_point, test, data)
        model_input = tokenizer(eval_prompt, return_tensors='pt').to('cuda')
        model.eval()
        with torch.no_grad():
            generated = model.generate(**model_input, max_new_tokens=512)[0]
            print(tokenizer.decode(generated))
    except Exception:
        print('Warning: Model check failed with exception:')
        print(format_exc())

    print('-----------------------------------------------------------------------------------------------------')

    tokenizer.add_eos_token = old_add_eos_token
