import datetime
import os.path
from timeit import default_timer as timer
from typing import Optional, Tuple
import json

import click
import hydra.utils
from omegaconf import DictConfig
from peft import get_peft_model, PeftModel
from transformers import PreTrainedTokenizer
from trl import ORPOTrainer, DPOTrainer
from datasets import Dataset

from proconsul.cli_utils import config_dir_option, config_name_option, create_hydra_config
from proconsul.train.utils import load_model


def train_model(
        checkpoint: str,
        run_name: str,
        data_path: str,
        lora_config_dict: DictConfig,
        generation_config_dict: DictConfig,
        training_config_dict: DictConfig,
        alignment_method: str,
        bits_and_bytes_config_dict: DictConfig,
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

    with open(data_path, 'r') as file:
        alignment_dataset_dict = json.load(file)
    if max_data_points is not None:
        alignment_dataset_dict = {k: v[:max_data_points] for k, v in alignment_dataset_dict.items()}
    train_dataset = Dataset.from_dict(alignment_dataset_dict)

    val_dataset = None

    if alignment_method == 'ORPO':
        trainer_type = ORPOTrainer
    elif alignment_method == 'DPO':
        trainer_type = DPOTrainer
    else:
        raise AssertionError(f'Unknown alignment method name "{alignment_method}"')
    trainer = trainer_type(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
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


def run_train(config: DictConfig) -> None:
    assert any([config.train.training_args._target_ == f'trl.{m}Config'
                for m in ['ORPO', 'DPO']]), 'Unexpected _target_ of training_args config'
    preferred_alignment_method = config.train.training_args._target_[4:-6]

    output_dir = os.path.abspath(config.train.training_args.output_dir)
    if os.path.exists(output_dir):
        raise RuntimeError(f'{output_dir} OUTPUT DIR ALREADY EXISTS, CHANGE NAME!')

    if hasattr(config.train, "max_data_points"):
        max_data_points = config.train.max_data_points
    else:
        max_data_points = None

    train_model(
        checkpoint=config.train.model_checkpoint,
        run_name=config.run_name,
        data_path=config.path_after_processing,
        lora_config_dict=config.train.lora,
        generation_config_dict=config.train.generation,
        training_config_dict=config.train.training_args,
        alignment_method=preferred_alignment_method,
        bits_and_bytes_config_dict=config.train.bits_and_bytes,
        save_best_model=config.train.save_best_model,
        max_data_points=max_data_points,
    )


@click.command()
@config_dir_option
@config_name_option
def main(config_dir: str, config_name: str) -> None:
    config = create_hydra_config(config_dir, config_name)
    run_train(config)


if __name__ == '__main__':
    main()
