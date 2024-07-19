import json
from typing import Any, Dict, Optional, Set

from omegaconf import DictConfig

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


def get_all_children(
    dataset: Dict[str, Dict[str, Any]],
    subset: Set[str],
    root_id: str,
    depth: Optional[int] = None,
) -> None:
    subset.add(root_id)

    if depth is not None and depth <= 0:
        return

    connected_ids = dataset[root_id][DatasetScheme.TO_KEY]
    for c in connected_ids:
        if c in subset or c not in dataset:
            continue

        get_all_children(
            dataset,
            subset,
            root_id=c,
            depth=depth - 1 if depth is not None else None
        )


def compute_subset_of_dataset(
    dataset: Dict[str, Dict[str, Any]],
    chosen_ids: Set[str],
    depth: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    subset = set()

    if depth is None or depth > 0:
        for d_id in chosen_ids:
            if d_id not in dataset or d_id in subset:
                continue

            get_all_children(
                dataset,
                subset,
                root_id=d_id,
                depth=depth,
            )

    data_subset = {key: val for key, val in dataset.items() if key in subset}
    return data_subset


def get_subset(
    dataset_path: str,
    out_path: str,
    chosen_ids: Set[str],
    depth: Optional[int] = None,
) -> None:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    data_subset = compute_subset_of_dataset(dataset, chosen_ids, depth)

    with open(out_path, "w") as f:
        json.dump(data_subset, f, indent=2)


def run_get_subset(config: DictConfig):
    config = config.data_processing.rec_sum_utils
    get_subset(
        config["dataset_path"],
        config["out_path"],
        set(config["chosen_ids"]),
        config.get("depth", None),
    )
