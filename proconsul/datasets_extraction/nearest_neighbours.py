import logging
from collections import defaultdict
import itertools
import json
import re
import time
from typing import Any, Optional, Dict, List

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from proconsul.common_utils import print_time, verify_output
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme

LOG = logging.getLogger(__name__)


class JaccardSimIndex:
    def __init__(self, **kwargs):
        pass

    def build_index(self, **kwargs):
        raise NotImplementedError('Implement Jaccard similarity index')

    def query_many(self, **kwargs):
        raise NotImplementedError('Implement Jaccard similarity index')


def print_histogram(data, bins=10, max_width=40):
    hist, bin_edges = np.histogram(data, bins=bins)

    max_count = max(hist)
    normalized_hist = [int((count / max_count) * max_width) for count in hist]

    print("Distribution Histogram:")
    for count, edge in zip(normalized_hist, bin_edges[:-1]):
        print(f"{edge:5.2f} | {'*' * count}")


def batch_generator(*args, batch_size: int = None):
    if not all(len(lst) == len(args[0]) for lst in args):
        raise ValueError("All input lists must have the same length.")

    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if batch_size > len(args[0]):
            raise ValueError("batch_size must not exceed the length of the input lists.")
    else:
        yield tuple(lst[:] for lst in args)
        return

    total_size = len(args[0])
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        yield tuple(lst[start_idx:end_idx] for lst in args)


def find_similar_items(docs, func_ids, threshold: float = 0.9, batch_size: int = None):
    LOG.info("Finding exact matches...")
    doc2func_ids = defaultdict(list)
    for doc, func_id in tqdm(zip(docs, func_ids), desc="Building dict..."):
        doc2func_ids[doc].append(func_id)

    # find exact matches
    new_docs = []
    new_func_ids = []
    func_id2exact_matches_func_id = {}
    result_pairs = set()
    for doc, f_ids in tqdm(doc2func_ids.items(), desc="Compute exact matches..."):
        new_docs.append(doc)
        new_func_ids.append(f_ids[0])  # select only first element from exact matches
        if len(f_ids) > 1:
            assert len(f_ids) < 1_000, f"Enormous number of exact match clones: {len(f_ids)}"
            func_id2exact_matches_func_id[f_ids[0]] = f_ids[1:]
            for pair in tqdm(itertools.combinations(f_ids, 2), disable=len(f_ids) < 1000, desc="Iterating combinations..."):
                result_pairs.add((*sorted(pair, reverse=True), 1))  # populate result with exact matches

    # near duplicates search
    with print_time("Building index for near duplicates"):
        index = JaccardSimIndex(show_progress=True, similarity_threshold=threshold)
        index.build_index(index_dataset=new_docs, entities_names=new_func_ids)

    for batch_docs, batch_func_ids in tqdm(batch_generator(docs, func_ids, batch_size=batch_size), desc="searching near duplicates..."):
        pairs = index.query_many(query_dataset=batch_docs, query_entities_names=batch_func_ids)
        for p in pairs:
            if p[0] != p[1]:
                result_pairs.add((*sorted([p[0], p[1]], reverse=True), p[2]))
                if p[0] in func_id2exact_matches_func_id:
                    for f_id in func_id2exact_matches_func_id[p[0]]:
                        if f_id != p[1]:
                            result_pairs.add((*sorted([f_id, p[1]], reverse=True), p[2]))
                if p[1] in func_id2exact_matches_func_id:
                    for f_id in func_id2exact_matches_func_id[p[1]]:
                        if f_id != p[0]:
                            result_pairs.add((*sorted([f_id, p[0]], reverse=True), p[2]))

    return result_pairs


def calculate_jaccard_similarities(input_dataset, field="doc", threshold=0.9, batch_size=None):
    # Filter out None documents
    def check_fields(content: Dict, field: str):
        splits = field.split('+')
        # 'doc' or 'code' cases
        if len(splits) == 1:
            return content[field] is not None and content[field].strip() != ""
        # 'doc+code' cases
        elif len(splits) == 2:
            return check_fields(content, splits[0]) and check_fields(content, splits[1])
        else:
            raise ValueError(f"Unexpected field {field} - expected next variants like: `doc`, `code`, `doc+code`")

    def get_doc(content: Dict, field: str):
        splits = field.split('+')
        # 'doc' or 'code' cases
        if len(splits) == 1:
            return content[field]
        # 'doc+code' cases
        elif len(splits) == 2:
            return content[splits[0]] + "\n" + content[splits[1]]
        else:
            raise ValueError(f"Unexpected field {field} - expected next variants like: `doc`, `code`, `doc+code`")

    docs = []
    func_ids = []
    for index, (func_id, content) in tqdm(enumerate(input_dataset.items()), total=len(input_dataset),
                                          desc="Preparing docs and func_ids..", leave=False):
        if check_fields(content, field):
            curr_doc = tuple(s for s in re.split(r'[^a-zA-Z0-9]+', get_doc(content, field).lower()) if s.strip())
            if curr_doc:
                docs.append(curr_doc)
                func_ids.append(func_id)
    print(f"{field}: Number of items to process: {len(docs)}")
    # Find similar docs in dataset
    result_pairs = find_similar_items(docs, func_ids, threshold=threshold, batch_size=batch_size)
    # Statistics
    print(f"{field}: Number of similar pairs: {len(result_pairs)}")
    print(f"{field}: Distribution of similarities scores:")
    res_similarities = [s[2] for s in result_pairs]
    print_histogram(res_similarities)
    print(f"{field}: Similarities scores equals to 1 in {sum(1 for s in res_similarities if s == 1)} out of "
          f"{len(result_pairs)} pairs")
    # Change to format {func_id: set([func_id1, func_id2, ...]),...}
    similar_items = defaultdict(set)
    for func_id1, func_id2, similarity in result_pairs:
        similar_items[func_id1].add((func_id2, similarity))
        similar_items[func_id2].add((func_id1, similarity))
    print(f"{field}: Number of items that have near duplicates {len(similar_items)} out of {len(docs)}")
    for func_id in similar_items:
        yield func_id, similar_items[func_id]


_NEAREST_SUFFIX = '_nearests'


def compute_near_duplicates(data: Dict, fields: List[str], batch_size: int, threshold: float):
    for field in fields:
        field_name = field + _NEAREST_SUFFIX
        start = time.time()
        res = calculate_jaccard_similarities(data, field=field, threshold=threshold, batch_size=batch_size)
        for func_id, similar_func_ids in res:
            data[func_id][field_name] = list(similar_func_ids)
        end = time.time()
        duration_seconds = end - start
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{field}: Pipeline duration: {int(hours)}:{int(minutes):02d}:{seconds:06.3f}")


def find_duplicates(
        input: str,
        output: str,
        threshold: float = 0.9,
        debug: bool = False,
        batch: Optional[int] = 10_000,
        usevendor: bool = False
):
    with open(input, 'r') as f:
        input_dataset = json.load(f)
    if not usevendor:
        input_dataset = {k: v for k, v in input_dataset.items() if "vendor" not in v or v["vendor"] is False}
    if debug:
        input_dataset = {k: v for k, v in list(input_dataset.items())[:100]}

    compute_near_duplicates(input_dataset, ['doc', 'code', 'doc+code'], batch, threshold)

    with open(output, 'w') as f:
        json.dump(input_dataset, f, indent=2)


def deduplicate_train(
    dataset: Dict[str, Dict[str, Any]], data_processing_config: DictConfig
) -> Dict[str, Dict[str, Any]]:
    threshold = data_processing_config.get("threshold", 0.9)
    batch_size = data_processing_config.get("batch", 10_000)
    target_doc_key = data_processing_config.get("target_doc_key", DatasetScheme.DOC_KEY)

    clone_fields = [target_doc_key, DatasetScheme.CODE_KEY, f'{target_doc_key}+{DatasetScheme.CODE_KEY}']
    compute_near_duplicates(dataset['train'], clone_fields, batch_size, threshold)

    similarity_fields = [f + _NEAREST_SUFFIX for f in clone_fields]
    added_keys = set()
    removed_num = 0
    train_data = dict()
    for key, value in tqdm(dataset['train'].items(), desc="remove duplicates..."):
        good_key = True
        for sim_type in similarity_fields:
            if sim_type not in value.keys():
                continue
            for sim_key, _ in value[sim_type]:
                if sim_key in added_keys:
                    good_key = False

        if good_key:
            train_data[key] = {k: v for k, v in value.items() if k not in similarity_fields}
            added_keys.add(key)
        else:
            removed_num += 1
    print(f"Removed {removed_num} out of {len(dataset['train'])} "
          f"{removed_num/len(dataset['train']):.2f} from train dataset")
    dataset['train'] = train_data

    return dataset


def run_deduplicate_train(config: DictConfig) -> None:
    with open(config.path_after_processing, "r") as file:
        dataset = json.load(file)
    output_file = verify_output(config.path_after_dedup_train)

    dataset = deduplicate_train(dataset=dataset, data_processing_config=config.data_processing)
    
    with open(output_file, 'w') as file:
        json.dump(dataset, file, indent=2)
