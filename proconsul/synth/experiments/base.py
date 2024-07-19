import re
from typing import Any, Callable, Dict, List, Set, Tuple

from proconsul.evaluation.triviality import Triviality
from proconsul.evaluation.verbosity import Verbosity
from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.synth.utils_synthetics import BaseSynthExperiment, strip_text

from thefuzz import fuzz


def is_likely_cpp_variable(word: str) -> bool:
    word = word.strip()
    if word == "":
        return False

    # skip abbreviations
    if any(c.isupper() for c in word):
        return False

    if word[0].isdigit():
        return False

    if word == "_" or word in ("IDs", "ASTs", "CPUs"):
        return False

    has_underscore = "_" in word
    has_uppercase = any(c.isupper() for c in word[1:])

    # Check for snake_case or camelCase
    if has_underscore or has_uppercase:
        return True
    return False


def custom_split(text: str) -> List[str]:
    # First, split by whitespace
    words = text.split()

    # Define a regular expression pattern for splitting by punctuation
    # and specific cases like '*Abbrev,' -> 'Abbrev'
    pattern = r"[\s\*\(\),;&\[\]\-\!\=\>\<\'\`\"\%\:\/\.]+"

    # Use a list comprehension with re.split() for further splitting
    split_words = [re.split(pattern, word) for word in words]

    # Flatten the list of lists
    flat_list = [item for sublist in split_words for item in sublist if item]
    # Remove trailing dots
    flat_list = [word[:-1] if word.endswith(".") else word for word in flat_list]

    return flat_list


def get_variables(text: str) -> Set[str]:
    if text.strip() == "":
        return set()
    return {word.split("::")[-1].split()[-1].lower() for word in custom_split(text) if is_likely_cpp_variable(word)}


def remove_comments(code: str):
    pattern = r"\/\*[\s\S]*\*\/|\/\/.*|\"+[\s\S]*\"+|\'+[\s\S]*\'+|\`+[\s\S]*\`+"
    return re.sub(pattern, "", code)


class SynthExperiment(BaseSynthExperiment):
    def synth_pre_context_clean(
        self, data: Dict[str, Any], info: Dict[str, Any], use_doc: bool = False, use_context: bool = False
    ) -> Dict[str, Any]:
        data = super().synth_pre_context_clean(data, info, use_doc, use_context)

        code_without_comments = remove_comments(data[DatasetScheme.CODE_KEY])
        if "{" in code_without_comments:
            name_with_args = strip_text(code_without_comments.split("{")[0])
            name_with_args_ind = data[DatasetScheme.SYNTHETIC_DOC_KEY].find(name_with_args)
            if name_with_args_ind!=-1:
                data[DatasetScheme.SYNTHETIC_DOC_KEY] = data[DatasetScheme.SYNTHETIC_DOC_KEY][:name_with_args_ind]+data[DatasetScheme.SYNTHETIC_DOC_KEY][name_with_args_ind+len(name_with_args):]
                
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
                triv(data[DatasetScheme.SYNTHETIC_DOC_KEY], {'name': data[DatasetScheme.NAME_KEY], 'code': data[DatasetScheme.CODE_KEY]}) < 0.1),
            ('synth doc is not verbose', lambda data: 
                verb(data[DatasetScheme.SYNTHETIC_DOC_KEY], {'name': data[DatasetScheme.NAME_KEY], 'code': data[DatasetScheme.CODE_KEY]}) < 0.5),
            ('only english keyboard symbols in synth doc', lambda data: len(data[DatasetScheme.SYNTHETIC_DOC_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.SYNTHETIC_DOC_KEY]))),
            ('synth doc does not copy parts of code', lambda data:
                fuzz.partial_ratio(
                    data[DatasetScheme.SYNTHETIC_DOC_KEY].lower(),
                    remove_comments(data[DatasetScheme.CODE_KEY]).lower()
                ) < 80
            ),
        ]
        
    def synth_target_filters(self, use_doc:bool=False, use_context:bool=False) -> List[Tuple[str, Callable]]:
        base_filters = super().synth_target_filters(use_doc, use_context)
        return base_filters + [
            ('only english keyboard symbols in context', lambda data: len(data[DatasetScheme.CONTEXT_KEY])==0 or bool(
                re.search(r'[a-zA-Z0-9\s\.\/\<\>\?\;\:\"\'\`\!\@\#\$\%\^\&\*\(\)\[\]\{\}\_\+\=\|\\\-\~\,]', data[DatasetScheme.CONTEXT_KEY]))),
            ('doc + code + cxt < 5000', lambda data:
                len(data[DatasetScheme.SYNTHETIC_DOC_KEY] + data[DatasetScheme.CODE_KEY] + data[DatasetScheme.CONTEXT_KEY]) < 5000),
            ('doc contains entities only from code/context', lambda data:
                len(
                    get_variables(data[DatasetScheme.SYNTHETIC_DOC_KEY]) - (get_variables(data[DatasetScheme.CODE_KEY]) | get_variables(data[DatasetScheme.CONTEXT_KEY]))
                ) == 0
            ),
        ]
