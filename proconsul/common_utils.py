import json
import os.path
import time
from typing import Dict

from importlib.resources import files


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def read_test_files() -> Dict:
    test_datasets_dir = files("proconsul.evaluation") / "test_datasets"
    test_datasets_path = test_datasets_dir.glob(f'**/random_test_dataset.json')
    test_dataset_dict = {}
    for file_path in test_datasets_path:
        with open(file_path, "r") as file:
            test_dataset_dict = test_dataset_dict | json.load(file)['test']
    return test_dataset_dict


def verify_output(output_file: str) -> str:
    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        raise RuntimeError(f'{output_file} OUTPUT FILE ALREADY EXISTS, CHANGE NAME!')
    return output_file


def is_enclosed(text: str, left: str = "{", right: str = "}", min_num: int = 0) -> bool:
    count = 0
    is_min_num_reached = min_num == 0

    for t in text:
        if t == left:
            count += 1
            if count == min_num:
                is_min_num_reached = True
        elif t == right:
            count -= 1
            if count < 0:
                return False
    return count == 0 and is_min_num_reached
