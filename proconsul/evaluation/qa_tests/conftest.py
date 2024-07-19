import os
from importlib.resources import files
import json
from pathlib import Path
from typing import Dict

import pytest

from proconsul.evaluation.qa_tests import statistics
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


aggregated_function_info = {}


def pytest_addoption(parser):
    parser.addoption("--predict-results", action="append",
                     help="Path to JSON-file with prediction results. "
                          "You can add several paths, functions of test_datasets will be searched through all of them.")
    parser.addoption("--output-directory", action="store", default="output",
                     help="Path to directory with output results, default is 'output'")


def json_into_dict(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    # TODO Support all splits, not just test
    if 'test' in data:
        data = data['test']
    if type(data) is list:
        return {v[DatasetScheme.ID_KEY]: v for v in data}
    elif type(data) is dict:
        return data
    else:
        raise ValueError("Unknown data format in {}".format(json_file))


def combine_test_dataset_with_docstring(test_dataset: Dict, model_predictions: Dict) -> None:
    global aggregated_function_info
    for key in test_dataset.keys():
        found_keys = [k for k in model_predictions.keys()
                      if (k in key and
                          test_dataset[key][DatasetScheme.NAME_KEY] == model_predictions[k][DatasetScheme.NAME_KEY])]
        if len(found_keys) == 0:
            raise ValueError("Model predictions does not contain this id - {}".format(key))
        if len(found_keys) != 1:
            raise ValueError("Model predictions contain multiple entries for this id - {}\n"
                             "Entries found: {}".format(key, found_keys))
        # update sufficiency_claims, illegal_facts, ToDos, etc.
        aggregated_function_info[key] = model_predictions[found_keys[0]] | test_dataset[key]
        aggregated_function_info[key][DatasetScheme.ID_KEY] = key


def pytest_generate_tests(metafunc):
    predict_results_paths = metafunc.config.option.predict_results
    if len(aggregated_function_info) == 0:
        test_datasets_dir = files("proconsul.evaluation") / "test_datasets"
        test_datasets_path = test_datasets_dir.glob(f'**/random_test_dataset.json')
        test_dataset_dict = {}
        for file_path in test_datasets_path:
            test_dataset_dict.update(json_into_dict(Path(file_path)))
        predict_results_dict = {}
        for predict_path in predict_results_paths:
            predict_results_dict.update(json_into_dict(Path(predict_path)))
        combine_test_dataset_with_docstring(test_dataset_dict, predict_results_dict)
    metafunc.parametrize("function_info", list(aggregated_function_info.values()), indirect=True)


@pytest.fixture
def function_info(request) -> Dict:
    yield request.param


def filter_results_for_validation(all_results: Dict) -> Dict:
    filtered_output = {}
    for key in all_results.keys():
        filtered_output[key] = {k: v for k, v in all_results[key].items()
                                if k in {DatasetScheme.NAME_KEY, DatasetScheme.PATH_KEY, DatasetScheme.DOC_KEY,
                                         DatasetScheme.GENSUM_KEY,
                                         DatasetScheme.SUFFICIENCY_CLAIMS_KEY,
                                         DatasetScheme.SUFFICIENCY_CLAIMS_SCORE_KEY,
                                         DatasetScheme.SUFFICIENCY_CLAIMS_ANSWERS_KEY,
                                         DatasetScheme.ILLEGAL_FACTS_KEY,
                                         DatasetScheme.ILLEGAL_FACTS_SCORE_KEY,
                                         DatasetScheme.ILLEGAL_FACTS_ANSWERS_KEY}}
    return filtered_output


@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    output_path = Path(session.config.option.output_directory)
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "scores.csv", "w", encoding="utf-8") as f1:
        f1.write(statistics.all_scores_to_csv())
    with open(output_path / "qa_tests_results.json", "w", encoding="utf-8") as f2:
        json.dump(aggregated_function_info, f2, indent=2)
