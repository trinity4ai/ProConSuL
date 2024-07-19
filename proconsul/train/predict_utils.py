import json
from traceback import format_exc
from typing import Any, Dict, List, Optional, Union

import hydra.utils
from datasets import Dataset
from omegaconf import DictConfig

from proconsul.common_utils import read_test_files
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.dataset_utils import get_datasets
from proconsul.train.dataset_extension import DatasetExtension
from proconsul.train.gen_models import InferenceEngine, VllmEngine, HFBaseEngine, HFRecursiveEngine

from abc import ABC


def check(
    gen_model: InferenceEngine,
    prompt_strategy_dict: DictConfig,
    data: Optional[Union[Dataset, DatasetExtension, List[Dict]]] = None,
    test: bool = False,
) -> None:
    print("Checking the model:")
    print("-----------------------------------------------------------------------------------------------------")

    data_point = {
        "id": "RepoId/ID12345",
        "code": """static bool getAArch64PBV(QualType QT, ASTContext &C) {
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
        "context": "",
        "doc": "Well,",
    }
    prompt_strategy = hydra.utils.instantiate(prompt_strategy_dict)

    try:
        eval_prompt = prompt_strategy.get_prompt(data_point, test, data)
        print(gen_model.generate_single(eval_prompt))
    except Exception:
        print("Warning: Model check failed with exception:")
        print(format_exc())

    print("-----------------------------------------------------------------------------------------------------")


class PredictStrategy(ABC):
    def __init__(self, predict_config: DictConfig, model_checkpoint: str, use_recursive: bool = False) -> None:
        super().__init__()
        self.predict_config = predict_config
        self.model_checkpoint = model_checkpoint
        self.prompt_strategy = hydra.utils.instantiate(self.predict_config.prompt)
        self.use_recursive = use_recursive
        self.max_length = self.predict_config.tokenizer_max_length

        self.gen_model = self._get_model()

    def predict(self, all_datasets: Dict[str, Dataset], response_template: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("PredictStrategy.predict is not implemented")

    def _get_model(self) -> InferenceEngine:
        if not self.use_recursive and self.predict_config.use_vllm:
            gen_model = VllmEngine(
                generation_config_dict=self.predict_config.generation_vllm,
                checkpoint=self.model_checkpoint,
                bits_and_bytes_config_dict=self.predict_config.bits_and_bytes,
                adapter_checkpoint=self.predict_config.adapter_checkpoint,
                max_length=self.max_length,
                test=True,
            )
        elif self.use_recursive:
            gen_model = HFRecursiveEngine(
                generation_config_dict=self.predict_config.generation,
                checkpoint=self.model_checkpoint,
                bits_and_bytes_config_dict=self.predict_config.bits_and_bytes,
                batch_size=self.predict_config.batch_size,
                adapter_checkpoint=self.predict_config.adapter_checkpoint,
                test=True,
            )
        else:
            gen_model = HFBaseEngine(
                generation_config_dict=self.predict_config.generation,
                checkpoint=self.model_checkpoint,
                bits_and_bytes_config_dict=self.predict_config.bits_and_bytes,
                batch_size=self.predict_config.batch_size,
                adapter_checkpoint=self.predict_config.adapter_checkpoint,
                test=True,
            )
        return gen_model

    def run_predict_template(
        self, json_data: Dict[str, Dict[str, Any]], config: DictConfig, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        all_datasets = get_datasets(json_data=json_data, max_points=self.predict_config.get("max_data_points", None))

        check(self.gen_model, config.train.prompt, test=True)

        return self.predict(
            all_datasets=all_datasets,
            response_template=config.train.response_template,
            **kwargs,
        )


def save_predict(full_output_path: str, short_output_path: str, full_output: Dict):
    with open(full_output_path, "w") as file:
        json.dump(full_output, file, indent=2)
    print(f"Full results saved to {full_output_path}")

    short_output = {split: dict() for split in full_output if split != "train"}
    test_points = read_test_files()
    test_ids = set(test_points.keys())
    for split in full_output.keys():
        if split == "train":
            continue
        for node_id, data_point in full_output[split].items():
            if node_id not in test_ids:
                continue
            assert (
                data_point[DatasetScheme.NAME_KEY] == test_points[node_id][DatasetScheme.NAME_KEY]
            ), f"Incorrect function name for id: {node_id}"
            short_output[split][node_id] = data_point

    with open(short_output_path, "w") as file:
        json.dump(short_output, file, indent=2)
    print(f"Results for test set saved to {short_output_path}")
