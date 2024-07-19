import json
import logging
from typing import List

from omegaconf import DictConfig
from tqdm import tqdm

from proconsul.common_utils import print_time, read_test_files, verify_output
from proconsul.data_processing.rec_sum_utils.get_subset import compute_subset_of_dataset
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.datasets_extraction.merge import merge_json
from proconsul.datasets_extraction.nearest_neighbours import find_duplicates

LOG = logging.getLogger(__name__)


def merge_and_dedup(
        input_dir: str,
        output_file: str,
        info_path: str,
        train_repos: List[str],
        test_repos: List[str],
        test_paths: List[str]
) -> None:
    train_repos = set(repo.lower() for repo in train_repos)
    test_repos = set(repo.lower() for repo in test_repos)

    output_file = verify_output(output_file)

    merge_json(input_dir, output_file, info_path)
    with print_time("Load merged file"):
        with open(output_file, 'r') as file:
            merged_data = json.load(file)

    split_data = dict()
    for key, value in merged_data.items():
        if key.split('/')[0].lower() in test_repos:
            split_data[f'test/{key}'] = value
        elif any(test_path in value[DatasetScheme.PATH_KEY] for test_path in test_paths):
            split_data[f'test/{key}'] = value
        elif key.split('/')[0].lower() in train_repos:
            split_data[f'train/{key}'] = value

    with print_time("Write file with splits"):
        with open(output_file, 'w') as file:
            json.dump(split_data, file)

    with print_time("Find dedups"):
        find_duplicates(output_file, output_file, usevendor=True, batch=10_000)

    with print_time("Load file with dedups"):
        with open(output_file, 'r') as file:
            data_with_sims = json.load(file)

    result_data = {'train': dict(), 'test': dict()}
    new_keys = ['doc_nearests', 'code_nearests', 'doc+code_nearests']
    for key, value in tqdm(data_with_sims.items(), desc="splitting on train and test w/o leaks..."):
        split, node_id = key.split('/', maxsplit=1)
        if split == 'test':
            result_data['test'][node_id] = value
            continue

        good_key = True
        for sim_type in new_keys:
            if sim_type not in value.keys():
                continue
            for sim_key, _ in value[sim_type]:
                if sim_key.split('/')[0] == 'test':
                    good_key = False

        if good_key:
            result_data['train'][node_id] = {k: v for k, v in value.items() if k not in new_keys}

    test_points = read_test_files()
    for uid, v in test_points.items():
        assert result_data['test'][uid][DatasetScheme.NAME_KEY] == v[DatasetScheme.NAME_KEY], f'Incorrect function name for id: {uid}'
    test_ids = set(test_points.keys())

    result_data['test'] = compute_subset_of_dataset(result_data['test'], test_ids, None)
    result_data['test'] = {key: {k: v for k, v in value.items() if k not in new_keys}
                           for key, value in result_data['test'].items()}

    with print_time("Write file with final result"):
        with open(output_file, 'w') as file:
            json.dump(result_data, file, indent=2)


def run_merge_and_dedup(config: DictConfig):
    merge_and_dedup(
        config.datasets_extraction.dataroot,
        config.path_after_extraction,
        config.datasets_extraction.repos_info_path,
        config.datasets_extraction.train_repo_lang_names,
        config.datasets_extraction.test_repo_lang_names,
        config.datasets_extraction.test_paths,
    )
