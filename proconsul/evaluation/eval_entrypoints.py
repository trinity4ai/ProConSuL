import json
import os
from omegaconf import DictConfig

from proconsul.evaluation.evaluation_utils import calc_auto_metrics
from proconsul.evaluation.ask_fact_halluc_gpt import get_gpt_fact_halluc_responses
from proconsul.evaluation.triviality import Triviality
from proconsul.evaluation.verbosity import Verbosity


def evaluate_auto_metrics(config_dict: DictConfig) -> None:
    data_path = config_dict.path_after_predict
    save_path = config_dict.path_after_autometrics_calc
    with open(data_path, 'r') as file:
        all_data = json.load(file)
    assert 'train' not in all_data, 'Annoying, huh? lmao'

    auto_metrics = [Verbosity(config_dict.evaluation.auto_labels['Verbosity']),
                    Triviality(config_dict.evaluation.auto_labels['Triviality'])]

    for split, data in all_data.items():
        metric_values = calc_auto_metrics(data, config_dict.evaluation, auto_metrics)
        for key, p in data.items():
            p |= metric_values[key]

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    with open(save_path, 'w') as file:
        json.dump(all_data, file, indent=2)


def add_gpt_fact_halluc_responses(config_dict: DictConfig) -> None:
    data_path = config_dict.path_after_predict
    with open(data_path, 'r') as file:
        all_data = json.load(file)
    assert 'train' not in all_data, 'Annoying, huh? lmao'
    data_len = sum(len(data) for data in all_data.values())
    assert data_len < config_dict.evaluation.gpt_onetime_query_limit, f'''Your config doesn\'t let you \
spend that much money on API ({data_len} points vs {config_dict.evaluation.gpt_onetime_query_limit})'''

    for split, data in all_data.items():
        messages = get_gpt_fact_halluc_responses(data, config_dict.evaluation)
        for key, p in data.items():
            p |= messages[key]

    os.makedirs(os.path.split(config_dict.path_after_llm_response)[0], exist_ok=True)
    with open(config_dict.path_after_llm_response, 'w') as file:
        json.dump(all_data, file, indent=2)


def merge_all_metrics(config_dict: DictConfig) -> None:
    assert os.path.exists(config_dict.path_after_autometrics_calc), f'''No autometrics \
file corresponding to "{config_dict.file_name}"'''
    with open(config_dict.path_after_autometrics_calc, 'r') as file:
        data_auto = json.load(file)
    assert os.path.exists(config_dict.path_after_annotation), f'''No annotation \
file corresponding to "{config_dict.file_name}"'''
    with open(config_dict.path_after_annotation, 'r') as file:
        data_semi = json.load(file)
    for split, data in data_semi.items():
        assert split in data_auto and set(data.keys()) == set(data_auto[split].keys()), '''Auto and Semi-auto datapoints
do not match together. If you compute auto after semi-auto, no need for this function.'''

    all_data = dict()
    for split, data in data_semi.items():
        all_data[split] = {key: data_point | data_auto[split][key] for key, data_point in data.items()}

    os.makedirs(os.path.split(config_dict.path_after_metrics_calc)[0], exist_ok=True)
    with open(config_dict.path_after_metrics_calc, 'w') as file:
        json.dump(all_data, file, indent=2)
