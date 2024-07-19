import json
from typing import Any, Dict

from omegaconf import DictConfig
from tqdm import tqdm

from proconsul.common_utils import verify_output
from proconsul.data_processing.utils_data_processing import (
    add_cxt,
    clean_dataset_strings,
    get_shared_filter_funcs,
    get_train_filter_funcs,
    passes_filter,
)


def clean_and_filter(dataset: Dict[str, Dict[str, Any]], test: bool) -> Dict[str, Dict[str, Any]]:
    clean_dataset_strings(dataset)
    filtered = dict()
    shared_filter_funcs = get_shared_filter_funcs()
    train_filter_funcs = get_train_filter_funcs()
    filter_fall_pool = []
    shared_fall_counter = 0
    for key, datapoint in tqdm(dataset.items()):
        b, _ = passes_filter(datapoint, shared_filter_funcs)
        if not b:
            shared_fall_counter += 1
            continue
        if not test:
            b, n = passes_filter(datapoint, train_filter_funcs)
            filter_fall_pool += [n]
            if not b:
                continue
        filtered[key] = datapoint
    filter_stats = []
    for i in range(len(train_filter_funcs)):
        filter_stats += [(i, filter_fall_pool.count(i))]
    print(shared_fall_counter)
    print(filter_stats)
    return filtered


def get_filtered_from_names(dataset: Dict[str, Dict[str, Any]], test: bool) -> Dict[str, Dict[str, Any]]:
    dataset_with_cxt = add_cxt(dataset)
    filtered_dataset = clean_and_filter(dataset_with_cxt, test=test)
    return filtered_dataset


def get_data_cxt(config: DictConfig) -> None:
    output_file = verify_output(config.path_after_processing)

    with open(config.path_after_extraction, 'r') as file:
        data = json.load(file)

    test_filtered = get_filtered_from_names(data['test'], test=True)
    train_filtered = get_filtered_from_names(data['train'], test=False)

    processed_data = {'train': train_filtered, 'test': test_filtered}

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
