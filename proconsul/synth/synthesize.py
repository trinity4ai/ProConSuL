import json
from typing import Any, Dict
from tqdm import tqdm

from datasets import Dataset
from omegaconf import DictConfig
from hydra.utils import instantiate

from proconsul.common_utils import verify_output
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.train.predict_utils import PredictStrategy
from proconsul.datasets_extraction.nearest_neighbours import deduplicate_train

from proconsul.synth.utils_synthetics import (
    BaseSynthExperiment,
    pre_synth_filters,
    pre_synth_clean,
)
from proconsul.data_processing.utils_data_processing import passes_filter, add_cxt


class Synthesize(PredictStrategy):
    def predict(
        self,
        all_datasets: Dict[str, Dataset],
        response_template: str,
        use_doc: bool = False,
        use_context: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        split_key = "train"
        assert "train" in all_datasets, "No data under train key"
        dataset = all_datasets["train"]

        print(f"Number of data before pre synth filtering: {len(dataset)}")

        # pre synth clean
        info = {
            "all_ids": set(
                ids
                for ids, is_not_func in zip(dataset[DatasetScheme.ID_KEY], dataset[DatasetScheme.NON_FUNCTION_KEY])
                if not is_not_func
            )
        }
        dataset = dataset.map(
            lambda data: pre_synth_clean(data, info, use_doc, use_context),
            desc="Pre synthesize cleaning",
        )

        # pre synth filters
        synth_filters = pre_synth_filters(use_doc, use_context)
        dataset = dataset.filter(
            lambda data: passes_filter(data, synth_filters)[0],
            desc="Pre synthesize filtering",
        )

        print(f"Number of data after pre synth filtering: {len(dataset)}")

        # synth generation
        output = self.gen_model.generate(
            data=dataset,
            max_length=self.max_length,
            prompt_strategy=self.prompt_strategy,
            response_template=response_template,
        )

        for ids in output.keys():
            output[ids][DatasetScheme.SYNTHETIC_DOC_KEY] = output[ids][DatasetScheme.GENSUM_KEY]
            del output[ids][DatasetScheme.GENSUM_KEY]

        print(f"Number of data after synth generation: {len(output)}")

        return {split_key: output}


def finalize_synth(
    path_after_synth: str,
    experiment: BaseSynthExperiment,
    use_doc: bool = False,
    use_context: bool = False,
) -> Dict[str, Dict[str, Any]]:
    with open(path_after_synth, "r") as output_file:
        all_splits = json.load(output_file)

    split_key = "train"
    assert split_key in all_splits, "No data under train key"
    output = all_splits[split_key]

    print(f"Number of data before pre cxt filters: {len(output)}")

    # pre cxt clean
    output = {
        ids: experiment.synth_pre_context_clean(data, {}, use_doc, use_context)
        for ids, data in tqdm(output.items(), desc="Pre cxt clean")
    }
    # pre cxt filters
    post_synth_filters = experiment.synth_pre_context_filters(use_doc, use_context)
    output = {
        ids: data
        for ids, data in tqdm(output.items(), desc="Pre cxt filtering")
        if passes_filter(data, post_synth_filters)[0]
    }

    print(f"Number of data after pre cxt filters: {len(output)}")

    # add context
    output = add_cxt(output, doc_field=DatasetScheme.SYNTHETIC_DOC_KEY)

    print(f"Number of data before target filters: {len(output)}")

    # target clean
    output = {
        ids: experiment.synth_target_clean(data, {}, use_doc, use_context)
        for ids, data in tqdm(output.items(), desc="Target clean")
    }
    # target filters
    target_filters = experiment.synth_target_filters(use_doc, use_context)
    output = {
        ids: data
        for ids, data in tqdm(output.items(), desc="Target filtering")
        if passes_filter(data, target_filters)[0] and all(to in output for to in data[DatasetScheme.TO_KEY])
    }

    print(f"Number of data after target filters: {len(output)}")

    for ids, data in output.items():
        output[ids] |= {DatasetScheme.DOC_KEY: data[DatasetScheme.SYNTHETIC_DOC_KEY]}

        for key in [
            DatasetScheme.SYNTHETIC_DOC_KEY,
            DatasetScheme.PROMPT_KEY,
            DatasetScheme.PROMPT_LEN_KEY,
            DatasetScheme.WHOLE_TEXT_KEY,
        ]:
            del output[ids][key]

    # replace old "train" split
    all_splits[split_key] = output

    return all_splits


def run_synthesize(config: DictConfig) -> None:
    path_after_synth = verify_output(config.path_after_synth)
    with open(config.path_after_processing, "r") as file:
        json_data = json.load(file)

    predict_res = Synthesize(
        predict_config=config.synth, model_checkpoint=config.train.model_checkpoint
    ).run_predict_template(
        json_data=json_data,
        config=config,
        use_doc=config.synth.use_doc,
        use_context=config.synth.use_context,
    )

    json_data = {key: predict_res[key] if key in predict_res else val for key, val in json_data.items()}
    with open(path_after_synth, "w") as output_file:
        json.dump(json_data, output_file, indent=2)
        print(f"Results saved to {path_after_synth}")


def run_finalize_synth(config: DictConfig) -> None:
    experiment = instantiate(config.synth.experiment)

    path_after_synth_final = verify_output(config.path_after_synth_final)
    output = finalize_synth(
        path_after_synth=config.path_after_synth,
        experiment=experiment,
        use_doc=config.synth.use_doc,
        use_context=config.synth.use_context,
    )
    # make train dedup
    output = deduplicate_train(dataset=output, data_processing_config=config.data_processing)
    print(f"Number of data after train dedup: {len(output['train'])}")

    with open(path_after_synth_final, "w") as output_file:
        json.dump(output, output_file, indent=2)
        print(f"Results saved to {path_after_synth_final}")
