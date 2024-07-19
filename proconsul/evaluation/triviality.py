import itertools
import os
import copy
import re
from collections import defaultdict
from typing import List, Set, Tuple, Dict

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.evaluation.auto_metric import AutoMetric
from proconsul.evaluation.code_abbreviations import CODE_ABBREVIATIONS


class Triviality(AutoMetric):

    def __init__(self, name: str = "Triviality", trivial_len_limit: int = 150) -> None:
        super().__init__()

        self._name = name
        self.abbrs, self.common_doc_words, self.stopwords = self._get_common_wordsets()
        self.trivial_len_limit = trivial_len_limit

    def get_name(self) -> str:
        return self._name

    @staticmethod
    def _get_common_wordsets() -> Tuple[defaultdict, List, Set]:
        abbrs = CODE_ABBREVIATIONS
        abbrs = {d['word']: [d['abbrs'][i]['abbr'] for i in range(len(d['abbrs']))] for d in abbrs}
        abbrs = {w + 's': l + [abb + 's' for abb in l] for w, l in abbrs.items()} | abbrs
        abbrs['destination'] += ['dst']
        abbrs = defaultdict(lambda: [], abbrs)

        common_doc_words = ['return', 'get', 'given', 'input', 'output', 'value', 'information', 'to',
                            'current', 'param', 'parameter', 'example', 'true', 'false', 'see',
                            'function', 'name', 'pre', 'post', 'set', 'assign', 'assigned']
        common_doc_words += [w + 's' for w in common_doc_words]

        # move import here to allow library using if there are some troubles with nltk or spacy
        import spacy
        from nltk.corpus import stopwords as nltk_stop

        stopwords = list(spacy.load("en_core_web_sm").Defaults.stop_words | set(nltk_stop.words('english')))
        stopwords = {w for w in stopwords if (len(w) > 1 or w in ['a'])}
        return abbrs, common_doc_words, stopwords

    @staticmethod
    def _split_fun_name(name: str) -> List[str]:
        presplits = name.split('_')
        splits = []
        for presplit in presplits:
            split_pos = []
            for i in range(len(presplit)):
                if i == 0:
                    split_pos += [i]
                elif presplit[i].isupper() and presplit[i - 1].islower():
                    split_pos += [i]
                elif presplit[i].isupper() and i != len(presplit) - 1 and presplit[i + 1].islower():
                    split_pos += [i]

            splits += [presplit[split_pos[i]:split_pos[i + 1]] for i in range(len(split_pos) - 1)]

            if len(split_pos) > 0:
                splits += [presplit[split_pos[-1]:]]
        for spl in copy.deepcopy(splits):
            for gr in itertools.groupby(spl, str.isdigit):
                splits += [''.join(list(gr[1]))]
        return list({spl.lower() for spl in splits}) + [name.lower()]

    def compute(self, summ: str, other_columns: Dict) -> float:
        fun_name, code = other_columns[DatasetScheme.NAME_KEY], other_columns[DatasetScheme.CODE_KEY]
        if len(summ.strip()) > self.trivial_len_limit:
            return 0
        splitted_fun_name = set(self._split_fun_name(fun_name))
        regex = re.compile('[^a-zA-Z0-9]')
        summ = summ.replace('\'s', ' ')
        splitted_doc = {w.lower() for w in regex.sub(' ', summ).split()}.difference(set(self.stopwords))
        if len(splitted_doc) < 2:
            return 1
        pre_fun_name = ' '.join(code.split('(')[0].split(' ')[:-1])
        in_brackets = code.split(')')[0].split('(')[-1]
        args_types = pre_fun_name + ' ' + in_brackets
        presplitted_args = {s for s in regex.sub(' ', args_types).split() if len(s) > 2}
        splitted_args = set(itertools.chain(*[self._split_fun_name(s) for s in presplitted_args]))

        score = 0
        digit2str = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }
        for w in splitted_doc:
            if w in digit2str:
                w = digit2str[w]
            max_score = 0
            for spl in splitted_fun_name | splitted_args:
                if spl not in splitted_fun_name:
                    score_mult = 0.6
                else:
                    score_mult = 1
                cur_score = 0
                if spl in digit2str:
                    spl = digit2str[spl]
                if w == spl:
                    cur_score = 1
                elif spl in self.abbrs[w]:
                    cur_score = 0.9
                elif len(spl) >= 3 and w.startswith(spl):
                    cur_score = 0.8
                elif len(w) >= 4 and w in spl:
                    cur_score = 0.8
                elif len(os.path.commonprefix([w, spl])) >= 3:
                    cur_score = len(os.path.commonprefix([w, spl])) / max(len(spl), len(w))
                elif w in self.common_doc_words:
                    cur_score = 0.5
                elif len(w) >= 4 and w[:4] in spl:
                    cur_score = 0.5
                elif w.startswith(spl):
                    cur_score = 0.3
                else:
                    continue
                max_score = max(max_score, cur_score * score_mult)
            score += max_score
        score /= (len(splitted_doc) - 0.5)
        adjusted_score = (score > 0.6) * ((score - 0.6) / 0.25) * (score < 0.85) + (score >= 0.85)
        return adjusted_score
