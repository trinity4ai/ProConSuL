import re
from typing import Any, Callable, Dict, List, Tuple
from abc import ABC

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.common_utils import is_enclosed


def strip_text(text: str, sub_patterns: str = r"[\/\\\:\*\s]*") -> str:
    pattern = r"|".join([r"^" + sub_patterns] + [sub_patterns + r"$"])
    return re.sub(pattern, "", text)


def pre_synth_clean(
    data: Dict[str, Any], info: Dict[str, Any], use_doc: bool = False, use_context: bool = False
) -> Dict[str, Any]:
    data[DatasetScheme.CODE_KEY] = (
        data[DatasetScheme.CODE_KEY].strip()
        if not data.get(DatasetScheme.MACRO_EXP_KEY, False)
        else data[DatasetScheme.MACRO_CODE_KEY].strip()
    )
    # remove ids in "to" not from dataset
    data[DatasetScheme.TO_KEY] = [to_id for to_id in data[DatasetScheme.TO_KEY] if to_id in info["all_ids"]]

    if use_doc:
        data[DatasetScheme.DOC_KEY] = "\n".join(
            [strip_text(doc_part) for doc_part in data[DatasetScheme.DOC_KEY].split("\n")]
        )
    return data


def pre_synth_filters(use_doc:bool=False, use_context:bool=False) -> List[Tuple[str, Callable]]:
    filters = [
        ('code > 50', lambda data: len(data[DatasetScheme.CODE_KEY]) > 50),
        ('code < 4000', lambda data: len(data[DatasetScheme.CODE_KEY]) < 4000),
        ('valid code or valid macro', lambda data:
            (data.get(DatasetScheme.MACRO_KEY, False) or is_enclosed(data[DatasetScheme.CODE_KEY], "{", "}", 1)) and is_enclosed(data[DatasetScheme.CODE_KEY], "(", ")", 1)
        ),
        ('macro or code must not have empty body', lambda data: data.get(DatasetScheme.MACRO_KEY, False) or not data[DatasetScheme.CODE_KEY].split('{')[1].strip().startswith('}')),
        ('only english keyboard symbols in code', lambda data: len(data[DatasetScheme.CODE_KEY])==0 or bool(
            re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CODE_KEY]))),
        ('is function', lambda data: not data.get(DatasetScheme.NON_FUNCTION_KEY, False)),
    ]
    
    if use_doc:
        filters += [
            ('doc < 800', lambda data: len(data[DatasetScheme.DOC_KEY]) < 800),
            ('no 4 repeated symbols', lambda data: not any([all([data[DatasetScheme.DOC_KEY][j + i] == data[DatasetScheme.DOC_KEY][i] for j in [1, 2, 3]]) for i in
                                                           range(len(data[DatasetScheme.DOC_KEY]) - 3)])),
            ('only english keyboard symbols in doc', lambda data: len(data[DatasetScheme.DOC_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.DOC_KEY]))),
        ]
    
    if use_context:
        filters += [
            ('only english keyboard symbols in context', lambda data: len(data[DatasetScheme.CONTEXT_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CONTEXT_KEY]))),
        ]
    
    if use_doc and use_context:
        filters += [
            ('doc + code + cxt < 5000', lambda data: len(data[DatasetScheme.DOC_KEY] + data[DatasetScheme.CODE_KEY] + data[DatasetScheme.CONTEXT_KEY]) < 5000),
        ]
    
    return filters


class BaseSynthExperiment(ABC):
    def synth_pre_context_clean(
        self, data: Dict[str, Any], info: Dict[str, Any], use_doc: bool = False, use_context: bool = False
    ) -> Dict[str, Any]:
        data[DatasetScheme.SYNTHETIC_DOC_KEY] = "\n".join([strip_text(doc_part) for doc_part in data[DatasetScheme.SYNTHETIC_DOC_KEY].split("\n")])
    
        short_name = data[DatasetScheme.NAME_KEY].split(':')[-1]
        if data[DatasetScheme.SYNTHETIC_DOC_KEY].startswith(short_name + ' - '):
            data[DatasetScheme.SYNTHETIC_DOC_KEY] = data[DatasetScheme.SYNTHETIC_DOC_KEY][len(short_name + ' - '):].strip()
        if data[DatasetScheme.SYNTHETIC_DOC_KEY].startswith(short_name + ': '):
            data[DatasetScheme.SYNTHETIC_DOC_KEY] = data[DatasetScheme.SYNTHETIC_DOC_KEY][len(short_name + ': '):].strip()
            
        return data

    def synth_pre_context_filters(self, use_doc: bool = False, use_context: bool = False) -> List[Tuple[str, Callable]]:
        return []

    def synth_target_clean(
        self, data: Dict[str, Any], info: Dict[str, Any], use_doc: bool = False, use_context: bool = False
    ) -> Dict[str, Any]:
        return data

    def synth_target_filters(self, use_doc: bool = False, use_context: bool = False) -> List[Tuple[str, Callable]]:
        return [
        ('not system',lambda data: not data.get(DatasetScheme.SYSTEM_KEY, False)),
        ('not macro',lambda data: not data.get(DatasetScheme.MACRO_KEY, False)),
        ('not macro exp',lambda data: not data.get(DatasetScheme.MACRO_EXP_KEY, False)),
        ('not vendor',lambda data: not data.get(DatasetScheme.VENDOR_KEY, False)),
        ('not declaration',lambda data: not data.get(DatasetScheme.DECL_KEY, False)),
    ]
