from itertools import chain, repeat
from typing import Iterator, List, Tuple

import networkx as nx
from datasets import Dataset
from torch.utils.data import Sampler

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


class RecursiveSampler(Sampler):
    def __init__(
            self,
            dataset: Dataset,
            max_batch_size: int,
            cycle_repeats: int = 2,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._max_batch_size = max_batch_size
        _project_graph = self._get_project_graph(dataset)
        _node_order = self._get_ordering(_project_graph, cycle_repeats)
        self._batches = self._get_batches(_node_order)

    @staticmethod
    def _get_project_graph(data: Dataset) -> nx.DiGraph:
        node_to_id = dict()
        for i, node in enumerate(data_point[DatasetScheme.ID_KEY] for data_point in data):
            node_to_id[node] = i
        proj_graph = nx.DiGraph()
        for row in data:
            node_from = row[DatasetScheme.ID_KEY]
            proj_graph.add_node(node_to_id[node_from])
            for node_to in filter(lambda node: node in node_to_id, row[DatasetScheme.TO_KEY]):
                proj_graph.add_edge(node_to_id[node_from], node_to_id[node_to])
        return proj_graph

    @staticmethod
    def _get_ordering(project_graph: nx.DiGraph, cycle_repeats: int = 2) -> List[Tuple[int, int]]:
        project_condensed = nx.condensation(project_graph)
        generations = list(nx.topological_generations(project_condensed))
        sorted_nodes = []
        generation_height = 0
        for generation in reversed(generations):
            for component in generation:
                nodes = project_condensed.nodes[component]['members']
                if len(nodes) > 1:
                    repeated_nodes = enumerate(chain.from_iterable(repeat(nodes, cycle_repeats)))
                    sorted_nodes += list(map(lambda i_node: (i_node[1], generation_height + i_node[0]), repeated_nodes))
                    generation_height += len(nodes) * cycle_repeats - 1
                else:
                    sorted_nodes += list(map(lambda node: (node, generation_height), nodes))
            generation_height += 1
        return sorted_nodes

    def _get_batches(self, node_order: List[Tuple[int, int]]) -> List[List[int]]:
        batches = []
        cur_node = 0
        while cur_node < len(node_order):
            cur_height = node_order[cur_node][1]
            cur_batch = []
            while len(cur_batch) < self._max_batch_size and \
                    cur_node < len(node_order) and \
                    cur_height == node_order[cur_node][1]:
                cur_batch.append(node_order[cur_node][0])
                cur_node += 1
            batches.append(cur_batch)
        return batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches:
            yield batch


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        gen_max_len: int,
        max_batch_mul: int = 32,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        indices_lengths = [(i, l) for i, l in enumerate(dataset[DatasetScheme.PROMPT_LEN_KEY])]
        self._lengths = [l for _, l in indices_lengths]
        self._indices = [i for i, _ in indices_lengths]

        self._batches = self._get_batches(
            ref_len=self._lengths[0], gen_max_len=gen_max_len, max_batch_mul=max_batch_mul
        )

    def _get_batches(self, ref_len: int, gen_max_len: int, max_batch_mul: int) -> List[List[int]]:
        batches = []
        i = 0
        while i < len(self._lengths):
            batch_mul = max(1, min(max_batch_mul, ref_len // (self._lengths[i] + gen_max_len)))
            cur_batch_size = batch_mul * self._batch_size
            batches.append(self._indices[i : i + cur_batch_size])

            i += cur_batch_size
        return batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches:
            yield batch
