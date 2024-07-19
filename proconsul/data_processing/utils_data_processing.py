import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.common_utils import is_enclosed


def strip_text(text: str) -> str:
    sub_patterns = r"[\/\\\:\*\s]*"
    pattern = r"|".join([r"^" + sub_patterns] + [sub_patterns + r"$"])
    return re.sub(pattern, "", text)


def get_context(
        dataset: Dict[str, Dict[str, Any]],
        node_id: str,
        datapoint: Dict,
        docstrings: Optional[Dict[str, str]] = None,
        separator: str = '\n*\n',
        scheme: str = '{name}: {doc}',
        add_params: bool = False,
        strip: bool = False,
        doc_field: str = DatasetScheme.DOC_KEY,
) -> str:
    callees = [dataset[node] for node in datapoint[DatasetScheme.TO_KEY] if node in dataset
               and node != node_id]
    context_points = []
    for callee in callees:
        name = callee[DatasetScheme.NAME_KEY].split('::')[-1]
        doc = docstrings[callee[DatasetScheme.ID_KEY]] if docstrings is not None else callee[doc_field]
        if strip:
            doc = doc.strip()
        if not add_params:
            context_points.append(scheme.format(name=name, doc=doc))
            continue
        params = callee[DatasetScheme.CODE_KEY].split('(')
        if len(params) < 2:
            params = ''
        else:
            params = params[1].split(')')
            params = params[0] if len(params) > 0 else ''
        if strip:
            params = params.strip()
        context_points.append(scheme.format(name=name, doc=doc, params=params))
    if strip:
        context_points = list(map(str.strip, context_points))
    context = separator.join(context_points)
    if strip:
        context = context.strip()
    return context


def add_cxt(dataset: Dict[str, Dict[str, Any]], doc_field: str = DatasetScheme.DOC_KEY) -> Dict[str, Dict[str, Any]]:
    for node_id, data_point in tqdm(dataset.items()):
        if data_point[doc_field] is None:
            data_point[doc_field] = ''
        data_point[DatasetScheme.CONTEXT_KEY] = get_context(dataset, node_id, data_point, doc_field=doc_field)
    return dataset


def clean_dataset_strings(dataset: Dict[str, Dict[str, Any]]) -> None:
    for data_point in tqdm(dataset.values()):
        data_point['raw_doc'] = data_point[DatasetScheme.DOC_KEY]
        doc = strip_text(data_point[DatasetScheme.DOC_KEY])
        if '    ' not in doc and '\n\n\n\n' not in doc:
            doc = re.sub(r'([ ,\n])(\1+)', r'\1', doc)
        data_point[DatasetScheme.DOC_KEY] = doc
        data_point[DatasetScheme.NAME_KEY] = data_point[DatasetScheme.NAME_KEY].strip()


def passes_filter(datapoint: Dict[str, Any], filter_funcs: List[Tuple[str, Callable]]) -> Tuple[bool, int]:
    # macro code -> code if macro ???
    for i, (fdesc, ffun) in enumerate(filter_funcs):
        if not ffun(datapoint):
            return False, i
    return True, -1


def get_train_filter_funcs() -> List[Tuple[str, Callable]]:
    return [
        ('doc < 250', lambda data: len(data[DatasetScheme.DOC_KEY]) < 250),
        ('doc has more than 2 and less than 70 words', lambda data: 70 > len(data[DatasetScheme.DOC_KEY].split()) > 2),
        ('code > 50', lambda data: len(data[DatasetScheme.CODE_KEY]) > 50),
        ('code < 4000', lambda data: len(data[DatasetScheme.CODE_KEY]) < 4000),
        ('doc is not doxygen', lambda data: 
                all(pattern not in data[DatasetScheme.DOC_KEY].lower() for pattern in ['@', '\\return', '\\param', '\\brief', '\\since'])),
        ('macro or code must not have empty body', lambda data: data.get(DatasetScheme.MACRO_KEY, False) or not data[DatasetScheme.CODE_KEY].split('{')[1].strip().startswith('}')),
        ('no 4 repeated symbols', lambda data: not any([all([data[DatasetScheme.DOC_KEY][j + i] == data[DatasetScheme.DOC_KEY][i] for j in [1, 2, 3]]) for i in
                                                           range(len(data[DatasetScheme.DOC_KEY]) - 3)])),
        ('doc does not start with name', lambda data: 
            not data[DatasetScheme.DOC_KEY].lower().startswith(data[DatasetScheme.NAME_KEY].lower().split(':')[-1]) 
            and not data[DatasetScheme.DOC_KEY].lower().startswith(data[DatasetScheme.NAME_KEY].lower().split('_')[-1]) 
            and not data[DatasetScheme.DOC_KEY].lower().startswith(data[DatasetScheme.NAME_KEY].lower())),
        ('doc has no bad words', lambda data: 
            all(
                pattern not in data[DatasetScheme.DOC_KEY].lower()
                for pattern in ['see', 'fixme', 'todo', 'deprecated']
            )
            and 'NOTE' not in data[DatasetScheme.DOC_KEY]
        ),
        ('not system',lambda data: not data.get(DatasetScheme.SYSTEM_KEY, False)),
        ('not macro',lambda data: not data.get(DatasetScheme.MACRO_KEY, False)),
        ('not macro exp',lambda data: not data.get(DatasetScheme.MACRO_EXP_KEY, False)),
        ('not vendor',lambda data: not data.get(DatasetScheme.VENDOR_KEY, False)),
        ('not declaration',lambda data: not data.get(DatasetScheme.DECL_KEY, False)),
        ('only english keyboard symbols in code', lambda data: len(data[DatasetScheme.CODE_KEY])==0 or bool(
            re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CODE_KEY]))),
        ('only english keyboard symbols in doc', lambda data: len(data[DatasetScheme.DOC_KEY])==0 or bool(
            re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.DOC_KEY]))),
        ('only english keyboard symbols in context', lambda data: len(data[DatasetScheme.CONTEXT_KEY])==0 or bool(
            re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CONTEXT_KEY]))),
        ('doc + code + cxt < 5000', lambda data:
            len(data[DatasetScheme.DOC_KEY] + data[DatasetScheme.CODE_KEY] + data[DatasetScheme.CONTEXT_KEY]) < 5000),

    ]


def get_shared_filter_funcs() -> List[Tuple[str, Callable]]:
    return [
        ('valid code or valid macro', lambda data:
            (
                data.get(DatasetScheme.MACRO_KEY, False) 
                or is_enclosed(data[DatasetScheme.CODE_KEY], "{", "}", 1)
            ) 
            and is_enclosed(data[DatasetScheme.CODE_KEY], "(", ")", 1)
        ),
        ('code < 5000', lambda data: len(data[DatasetScheme.CODE_KEY]) < 5000),
        ('macro or code must not have empty body', lambda data: data.get(DatasetScheme.MACRO_KEY, False) or not data[DatasetScheme.CODE_KEY].split('{')[1].strip().startswith('}')),
        ('is function', lambda data: not data.get(DatasetScheme.NON_FUNCTION_KEY, False)),
        ('not declaration', lambda data: not data.get(DatasetScheme.DECL_KEY, False)),
    ]


def convert_data_to_array(in_filename: str, out_filename: str, key: str) -> None:
    with open(in_filename, 'r') as in_file:
        data = json.load(in_file)

    data_list = [{DatasetScheme.ID_KEY: k} | v for k, v in data.items()]
    out_data = {key: data_list}

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, 'w') as out_file:
        json.dump(out_data, out_file, indent=2)
