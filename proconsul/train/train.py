import datetime
import json
import os.path
from timeit import default_timer as timer
from typing import Optional, Tuple

import hydra.utils
from omegaconf import DictConfig
from peft import get_peft_model, PeftModel
from transformers import PreTrainedTokenizer, Seq2SeqTrainer

from proconsul.train.utils import check_model, load_model, process_dataset
from proconsul.train.custom_collators import CollatorForCompletionWithEos
from proconsul.train.dataset_utils import get_datasets


def train_model(
        checkpoint: str,
        run_name: str,
        data_path: str,
        response_template: str,
        max_length: int,
        lora_config_dict: DictConfig,
        generation_config_dict: DictConfig,
        training_config_dict: DictConfig,
        bits_and_bytes_config_dict: DictConfig,
        prompt_strategy_dict: DictConfig,
        save_best_model: bool = True,
        max_data_points: Optional[int] = None,
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    lora_config = hydra.utils.instantiate(lora_config_dict)

    generation_config = hydra.utils.instantiate(generation_config_dict)

    training_args = hydra.utils.instantiate(training_config_dict)
    training_args.generation_config = generation_config
    training_args.run_name = f'{run_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    model, tokenizer = load_model(checkpoint, bits_and_bytes_config_dict)
    model.config.pad_token_id = -100

    model.train()
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    prompt_strategy = hydra.utils.instantiate(prompt_strategy_dict)

    with open(data_path, 'r') as file:
        json_data = json.load(file)
    all_datasets = get_datasets(json_data, max_data_points)

    train_dataset = all_datasets['train']
    val_dataset = all_datasets['validation'] if 'validation' in all_datasets else None

    train_data = process_dataset(train_dataset, tokenizer, max_length, prompt_strategy)
    val_data = process_dataset(val_dataset, tokenizer, max_length, prompt_strategy) if val_dataset else None

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=CollatorForCompletionWithEos(
            response_template=response_template,
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors='pt',
        ),
    )

    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    model.config.use_cache = False

    print('Started training')
    start_time = timer()
    trainer.train()
    print(f'Finished training in {timer() - start_time}s')

    if save_best_model:
        print('Saving best checkpoint')
        trainer.save_model(os.path.join(training_args.output_dir, 'best'))
        print('Finished saving best checkpoint')

    return model, tokenizer


# Run with > deepspeed --num_gpus=2 --no_local_rank --module proconsul.train.train
def run_train(config: DictConfig) -> None:
    output_dir = os.path.abspath(config.train.training_args.output_dir)
    if os.path.exists(output_dir):
        raise RuntimeError(f'{output_dir} OUTPUT DIR ALREADY EXISTS, CHANGE NAME!')

    if hasattr(config.train, "max_data_points"):
        max_data_points = config.train.max_data_points
    else:
        max_data_points = None

    model, tokenizer = train_model(
        checkpoint=config.train.model_checkpoint,
        run_name=config.run_name,
        data_path=config.path_after_processing,
        response_template=config.train.response_template,
        max_length=config.train.tokenizer_max_length,
        lora_config_dict=config.train.lora,
        generation_config_dict=config.train.generation,
        training_config_dict=config.train.training_args,
        bits_and_bytes_config_dict=config.train.bits_and_bytes,
        prompt_strategy_dict=config.train.prompt,
        save_best_model=config.train.save_best_model,
        max_data_points=max_data_points,
    )

    check_model(model, tokenizer, config.train.prompt)
