from typing import Any, Dict, Optional

from datasets import Dataset

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


def verify_keys(data_point: Dict) -> None:
    keys = [
        DatasetScheme.ID_KEY,
        DatasetScheme.CODE_KEY,
        DatasetScheme.DOC_KEY,
        DatasetScheme.NAME_KEY,
        DatasetScheme.TO_KEY,
    ]
    for key in keys:
        if key not in data_point:
            raise ValueError(f'DATASET HAS NO "{key}" KEY!')


def get_datasets(json_data: Dict[str, Dict[str, Any]], max_points: Optional[int] = None) -> Dict[str, Dataset]:
    datasets = dict()
    for split, data in json_data.items():
        dataset = Dataset.from_list([{DatasetScheme.ID_KEY: k} | v for k, v in data.items()])
        if len(dataset) == 0:
            continue
        datasets[split] = dataset
        verify_keys(datasets[split][0])
        if max_points is not None:
            datasets[split] = datasets[split].select(range(max_points))

    return datasets
