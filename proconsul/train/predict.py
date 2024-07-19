import json
from typing import Any, Dict

from datasets import Dataset
from omegaconf import DictConfig

from proconsul.common_utils import verify_output
from proconsul.train.predict_utils import save_predict, PredictStrategy
from proconsul.data_processing.utils_data_processing import get_shared_filter_funcs, passes_filter


class Predict(PredictStrategy):
    def predict(self, all_datasets: Dict[str, Dataset], response_template: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        split_key = "test"
        assert split_key in all_datasets, "No data under test key"
        dataset = all_datasets[split_key]

        test_filters = get_shared_filter_funcs()
        dataset = dataset.filter(
            lambda data: passes_filter(data, test_filters)[0],
            desc="Test data filtering",
        )
        print(f"Number of valid points: {len(dataset)}")

        full_output = dict()
        output = self.gen_model.generate(
            data=dataset,
            max_length=self.max_length,
            prompt_strategy=self.prompt_strategy,
            response_template=response_template,
        )

        full_output |= {split_key: output}

        return full_output


def _run_predict(config: DictConfig, use_recursive: bool) -> None:
    full_output_path = verify_output(config.path_after_predict_full)
    short_output_path = verify_output(config.path_after_predict)

    with open(config.path_after_processing, "r") as file:
        json_data = json.load(file)

    predict_res = Predict(
        predict_config=config.predict, model_checkpoint=config.train.model_checkpoint, use_recursive=use_recursive
    ).run_predict_template(
        json_data=json_data,
        config=config,
    )
    json_data = {key: predict_res[key] if key in predict_res else val for key, val in json_data.items()}

    save_predict(full_output_path, short_output_path, json_data)


def run_predict(config: DictConfig) -> None:
    _run_predict(config=config, use_recursive=False)


def run_recursive_predict(config: DictConfig) -> None:
    _run_predict(config=config, use_recursive=True)
