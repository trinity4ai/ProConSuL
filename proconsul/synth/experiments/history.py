import re
from typing import Any, Callable, Dict, List, Tuple

from proconsul.evaluation.triviality import Triviality
from proconsul.evaluation.verbosity import Verbosity
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.synth.utils_synthetics import BaseSynthExperiment, strip_text

import numpy as np


def has_substr_copy(doc: str, ref: str, length: int) -> int:
    if len(doc) < length or len(ref) < length:
        return 0
    doc_substr = set([doc[i: i + length] for i in range(len(doc) - length)])
    ref_substr = set([ref[i: i + length] for i in range(len(ref) - length)])
    if len(doc_substr.intersection(ref_substr)) > 0:
        return 1
    return 0


def has_leaked_entities(doc: str, code: str, cxt: str, gt_doc: str) -> int:
    """ Searches doc for code entities present in gt_doc only. """
    doc = re.sub(r'[^\w]', ' ', doc)
    ent = [e for e in doc.split() if '_' in e or '()' in e or e[1:].lower() != e[1:]]
    ent = [e for e in ent if len(e) > 3]
    for e in ent:
        if e in gt_doc and e.lower() not in code.lower() + cxt.lower():
            return 1
    return 0


class HistoryExperiment(BaseSynthExperiment):
    def __init__(self, drop_if_no_callees_ratio: float = 1.0) -> None:
        super().__init__()
        self.drop_if_no_callees_ratio = drop_if_no_callees_ratio

    def synth_pre_context_clean(
        self, data: Dict[str, Any], info: Dict[str, Any], use_doc: bool = False, use_context: bool = False
    ) -> Dict[str, Any]:
        data = super().synth_pre_context_clean(data, info, use_doc, use_context)
        data[DatasetScheme.SYNTHETIC_DOC_KEY] = "\n".join(
            [strip_text(doc_part, sub_patterns = r"[\/\\\:\*\s\"\']*") for doc_part in data[DatasetScheme.SYNTHETIC_DOC_KEY].split("\n")]
        )
        return data

    def synth_pre_context_filters(self, use_doc:bool=False, use_context:bool=False) -> List[Tuple[str, Callable]]:
        base_filters = super().synth_pre_context_filters(use_doc, use_context)
        triv, verb = Triviality().compute, Verbosity().compute
        return base_filters + [
            ('synth doc < 180', lambda data: len(data[DatasetScheme.SYNTHETIC_DOC_KEY]) < 180),
            ('synth doc has more than 2 and less than 70 words', lambda data: 70 > len(data[DatasetScheme.SYNTHETIC_DOC_KEY].split()) > 2),
            ('synth doc does not start with name', lambda data: 
                not data[DatasetScheme.SYNTHETIC_DOC_KEY].lower().startswith(data[DatasetScheme.NAME_KEY].lower().split(':')[-1]) 
                and not data[DatasetScheme.SYNTHETIC_DOC_KEY].lower().startswith(data[DatasetScheme.NAME_KEY].lower())),
            ('synth doc does not have code', lambda data: 
                all(pattern not in data[DatasetScheme.SYNTHETIC_DOC_KEY] for pattern in ['{', '}'])),
            ('synth doc is not doxygen', lambda data: 
                all(pattern not in data[DatasetScheme.SYNTHETIC_DOC_KEY].lower() for pattern in ['@', '\\return', '\\param', '\\brief', '\\since'])),
            ('synth doc has no bad words', lambda data: 
                all(
                    pattern not in data[DatasetScheme.SYNTHETIC_DOC_KEY].lower() 
                    for pattern in ['see', 'fixme', 'todo', 'deprecated', 'function is used to', 'function is called', 'this is intended to']
                )
                and 'NOTE' not in data[DatasetScheme.SYNTHETIC_DOC_KEY]
            ),
            ('synth doc has no 4 repeated symbols', lambda data: not any([
                all(
                    [data[DatasetScheme.SYNTHETIC_DOC_KEY][j + i] == data[DatasetScheme.SYNTHETIC_DOC_KEY][i] for j in [1, 2, 3]]
                ) for i in range(len(data[DatasetScheme.SYNTHETIC_DOC_KEY]) - 3)
            ])),
            ('synth doc is not trivial', lambda data: 
                triv(data[DatasetScheme.SYNTHETIC_DOC_KEY], {'name': data[DatasetScheme.NAME_KEY], 'code': data[DatasetScheme.CODE_KEY]}) < 0.8),
            ('synth doc is not verbose', lambda data: 
                verb(data[DatasetScheme.SYNTHETIC_DOC_KEY], {'name': data[DatasetScheme.NAME_KEY], 'code': data[DatasetScheme.CODE_KEY]}) < 0.9),
            ('only english keyboard symbols in synth doc', lambda data: len(data[DatasetScheme.SYNTHETIC_DOC_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.SYNTHETIC_DOC_KEY]))),
        ]

    def synth_target_filters(self, use_doc:bool=False, use_context:bool=False) -> List[Tuple[str, Callable]]:
        base_filters = super().synth_target_filters(use_doc, use_context)
        rng = np.random.default_rng(0)
        base_filters += [
            ('only english keyboard symbols in context', lambda data: len(data[DatasetScheme.CONTEXT_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CONTEXT_KEY]))),
            ('doc + code + cxt < 5000', lambda data:
                len(data[DatasetScheme.SYNTHETIC_DOC_KEY] + data[DatasetScheme.CODE_KEY] + data[DatasetScheme.CONTEXT_KEY]) < 5000),
            ('doc has no substring copy from code', lambda data:
                has_substr_copy(data[DatasetScheme.SYNTHETIC_DOC_KEY], data[DatasetScheme.CODE_KEY], 35)==0
            ),
            ('doc has no substring copy from context', lambda data:
                has_substr_copy(data[DatasetScheme.SYNTHETIC_DOC_KEY], data[DatasetScheme.CONTEXT_KEY], 60)==0
            ),
            ('rnd drop if no callees', lambda data: len(data[DatasetScheme.TO_KEY]) == 0 and rng.random() <= self.drop_if_no_callees_ratio),
        ]
        if use_doc:
            base_filters += [
                ('doc has no leaked entities present in gt_doc only', lambda data:
                    has_leaked_entities(
                        data[DatasetScheme.SYNTHETIC_DOC_KEY],
                        data[DatasetScheme.CODE_KEY],
                        data[DatasetScheme.CONTEXT_KEY],
                        data[DatasetScheme.DOC_KEY]
                    ) == 0
                ),
            ]
        return base_filters
