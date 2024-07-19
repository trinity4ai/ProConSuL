import json
import random
import re
from string import ascii_letters
from typing import Any, Dict, List, Optional, Set

from omegaconf import DictConfig

from proconsul.common_utils import verify_output
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


def generate_name(used_names: Set[str], min_len: int, max_len: int) -> str:
    name_len = random.randint(min_len, max_len)
    new_name = ''
    for _ in range(name_len):
        new_name += random.choice(ascii_letters)

    if new_name in used_names:
        return generate_name(used_names, min_len, max_len)
    return new_name


def generate_names(names: List[str], min_len: int, max_len: int, seed: Optional[int]) -> Dict[str, str]:
    if seed is not None:
        random.seed(seed)
    used_names = set()
    new_names = dict()
    for name in names:
        new_name = generate_name(used_names, min_len, max_len)
        used_names.add(new_name)
        new_names[name] = new_name
    return new_names


def obfuscate_data(
        data: Dict[str, Dict[str, Any]],
        all_data: Dict[str, Dict[str, Any]],
        min_len: int,
        max_len: int,
        rate: float,
        seed: Optional[int],
) -> Dict[str, Dict]:
    all_names = [data_point[DatasetScheme.NAME_KEY].split(':')[-1] for data_point in all_data.values()]
    new_names = generate_names(all_names, min_len, max_len, seed)

    for cur_node, data_point in data.items():
        used_names = [all_data[node][DatasetScheme.NAME_KEY].split(':')[-1]
                      for node in data_point[DatasetScheme.TO_KEY] if node in all_data and node != cur_node]
        for name in used_names:
            # Check that name satisfies C/C++ naming rules (to filter out operators)
            if re.fullmatch(r'^[a-zA-Z0-9_]+$', name) is None:
                continue
            p = re.compile(f'(?<![a-zA-Z0-9_]){re.escape(name)}(?![a-zA-Z0-9_])')
            full_code = data_point[DatasetScheme.CODE_KEY]
            if '{' not in full_code:
                continue
            code_signature, code_body = full_code.split('{', 1)
            # Check that there actually exists something to obfuscate
            if re.search(p, data_point[DatasetScheme.CONTEXT_KEY]) is None or re.search(p, code_body) is None:
                continue
            if random.random() < rate:
                data_point[DatasetScheme.CONTEXT_KEY] = re.sub(p, new_names[name], data_point[DatasetScheme.CONTEXT_KEY])
                data_point[DatasetScheme.CODE_KEY] = code_signature + '{' + re.sub(p, new_names[name], code_body)

    return data


def obfuscate(config: DictConfig) -> None:
    output_file = verify_output(config.path_after_obfuscation)

    with open(config.path_after_processing, 'r') as file:
        data = json.load(file)

    with open(config.path_after_extraction, 'r') as file:
        all_data = json.load(file)

    min_len = config.data_processing.get('min_len', 3)
    max_len = config.data_processing.get('max_len', 7)
    rate = config.data_processing.get('rate', 0.5)
    seed = config.data_processing.get('seed', None)
    data['train'] = obfuscate_data(data['train'], all_data['train'], min_len, max_len, rate, seed)

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)
