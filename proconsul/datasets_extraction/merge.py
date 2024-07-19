import json
import logging
import os
from glob import glob

import pandas as pd
from tqdm import tqdm

from proconsul.common_utils import print_time

LOG = logging.getLogger(__name__)


def merge_json(input_dir, output_file, csv_file):
    final_data = {}
    pattern = os.path.join(input_dir, '**', '0', 'raw_dataset.json')
    info = pd.read_csv(csv_file)

    for file_path in tqdm(glob(pattern, recursive=True), desc='merging all json files to one...'):
        subdirectory_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        row = info[info["name"] == subdirectory_name]
        if row.shape[0] == 0:
            print(f'WARNING: no info found about repository {subdirectory_name}')
            continue
        if row.shape[0] != 1:
            raise ValueError(f"More than one row for repository {subdirectory_name}:\n{row}")
        if not row["ready"].tolist()[0]:
            # not ready
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                modified_key = f"{subdirectory_name}/{key}"
                # Initialize a new dictionary for each key, ensuring all fields are copied
                modified_values = {}
                for inner_key, inner_value in value.items():
                    if inner_key == "to":
                        # Modify the "to" field specifically
                        modified_values[inner_key] = [f"{subdirectory_name}/{v}" for v in inner_value]
                    else:
                        # Copy all other fields directly
                        modified_values[inner_key] = inner_value
                final_data[modified_key] = modified_values

    with print_time("Save the merge result"):
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
