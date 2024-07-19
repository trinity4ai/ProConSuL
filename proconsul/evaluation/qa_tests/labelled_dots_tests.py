import sys
from typing import List, Dict

import pytest

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme
from proconsul.evaluation.qa_tests import statistics
from proconsul.evaluation.qa_tests.questioning import ask_questions_on_docstring
from proconsul.evaluation.qa_tests.conftest import aggregated_function_info


@pytest.fixture
def generated_docstring(function_info: Dict, scope='session') -> str:
    return function_info[DatasetScheme.GENSUM_KEY]


@pytest.fixture
def function_id(function_info: Dict, scope='session') -> str:
    return function_info[DatasetScheme.ID_KEY]


@pytest.fixture
def gpt_question_start() -> str:
    return "Does the docstring mention"


@pytest.fixture
def sufficiency_claims(function_info: Dict, scope='session') -> List[str]:
    return function_info[DatasetScheme.SUFFICIENCY_CLAIMS_KEY]


@pytest.fixture
def illegal_facts(function_info: Dict, scope='session') -> List[str]:
    return function_info[DatasetScheme.ILLEGAL_FACTS_KEY]


@pytest.fixture
def random_questions(function_info, scope='session') -> List[str]:
    return function_info[DatasetScheme.GENSUM_KEY]


@pytest.fixture
def sufficiency_subs(function_info, scope='session') -> List[str]:
    return function_info[DatasetScheme.SUFFICIENCY_SUBS_KEY]


@pytest.fixture
def hallucination_subs(function_info, scope='session') -> List[str]:
    return function_info[DatasetScheme.HALLUCINATION_SUBS_KEY]


@pytest.fixture
def factuality_with_gpt(function_info: Dict, scope='session') -> int:
    return function_info["Factuality_with_gpt"]


@pytest.fixture
def hallucinations_with_gpt(function_info: Dict, scope='session') -> int:
    return function_info["Hallucinations_with_gpt"]


@pytest.fixture
def verbosity_auto(function_info: Dict, scope='session') -> float:
    return function_info["Verbosity_auto"]


@pytest.fixture
def triviality_auto(function_info: Dict, scope='session') -> float:
    return function_info["Triviality_auto"]


@pytest.fixture
def sufficiency_answers_score(function_info: Dict, scope='session') -> float:
    return function_info[DatasetScheme.SUFFICIENCY_CLAIMS_SCORE_KEY]


def test_docstring_on_sufficiency_claims(function_id: str, generated_docstring: str,
                                         sufficiency_claims: List[str], gpt_question_start: str):
    claims_answers = ask_questions_on_docstring(generated_docstring, sufficiency_claims, gpt_question_start)
    passed_questions = [q for q, a in claims_answers.items() if str(a).startswith("Yes")]
    passed_score = len(passed_questions) / len(sufficiency_claims)
    aggregated_function_info[function_id][DatasetScheme.SUFFICIENCY_CLAIMS_ANSWERS_KEY] = claims_answers
    aggregated_function_info[function_id][DatasetScheme.SUFFICIENCY_CLAIMS_SCORE_KEY] = passed_score
    statistics.add_result(DatasetScheme.SUFFICIENCY_CLAIMS_SCORE_KEY, len(passed_questions), len(sufficiency_claims))
    print(f"""\n
        - Docstring under test -\n{generated_docstring}\n\n
        - Found - OK -\n{set(passed_questions)}\n\n
        - Missing -\n{set(claims_answers).difference(set(passed_questions))}\n""")
    assert passed_score == 1.0


def test_docstring_on_illegal_facts(function_id: str, generated_docstring: str,
                                    illegal_facts: List[str], gpt_question_start: str):
    if len(illegal_facts) == 0:
        print(f"""\n - No illegal facts found for {function_id}\n\n""")
    else:
        claims_answers = ask_questions_on_docstring(generated_docstring, illegal_facts, gpt_question_start)
        passed_questions = [q for q, a in claims_answers.items() if str(a).startswith("No")]
        passed_score = len(passed_questions) / len(illegal_facts)
        aggregated_function_info[function_id][DatasetScheme.ILLEGAL_FACTS_ANSWERS_KEY] = claims_answers
        aggregated_function_info[function_id][DatasetScheme.ILLEGAL_FACTS_SCORE_KEY] = passed_score
        statistics.add_result(DatasetScheme.ILLEGAL_FACTS_SCORE_KEY, len(passed_questions), len(illegal_facts))
        print(f"""\n
            - Docstring under test -\n{generated_docstring}\n\n
            - Absent - OK -\n{set(passed_questions)}\n\n
            - Found - unexpectedly -\n{set(claims_answers).difference(set(passed_questions))}\n""")
        assert passed_score == 1.0


# rename to `save_all_scores` so that all scores are collected from JSON-file with predict_results
def save_all_scores(function_id: str, factuality_with_gpt: int, hallucinations_with_gpt: int,
                         verbosity_auto: float, triviality_auto: float, sufficiency_answers_score: float):
    statistics.add_result("factuality_with_gpt", factuality_with_gpt, 1)
    statistics.add_result("hallucinations_with_gpt", hallucinations_with_gpt, 1)
    statistics.add_result("verbosity_auto", verbosity_auto, 1)
    statistics.add_result("triviality_auto", triviality_auto, 1)
    #statistics.add_result(DatasetScheme.SUFFICIENCY_CLAIMS_SCORE_KEY, sufficiency_answers_score, 1)
    assert 1.0 == 1.0


def main(args):
    import subprocess

    return subprocess.call(['pytest', '--tb=short', str(__file__)] + list(args))


if __name__ == "__main__":
    errcode = main(sys.argv[1:])
    sys.exit(errcode)
