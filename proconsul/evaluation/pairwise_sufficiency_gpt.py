from typing import Tuple, List, Dict, Any
import json

from omegaconf import DictConfig
from openai import OpenAI

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme as DSch


def get_suff_prompts(code, doc1, doc2) -> Tuple[str, str, str]:
    sys = f'''Below you have a code snippet with 2 summaries delimited with summary_A and summary_B tags. \
Please tell which one of them is more comprehensive and complete, \
i.e. covers more crucial aspects of the code and gives a clearer description of what the function does, \
or if they are equally comprehensive. Please be as concise as possible, I don't have much time.'''
    user = f'''\
<code>
{code} </code>

<summary_A>
{doc1} </summary_A>

<summary_B>
{doc2} </summary_B>

Which one is more complete? Are they comparable?
Your answer:'''
    final_grade_message = f'''

Based on your thoughts give a final answer. Return a single character: "A" for summary_A, "B" for summary_B and "С" if they are comparable.
Your response (one letter):'''
    return sys, user, final_grade_message


def compare_pair_sufficiency_gpt(code: str, doc_a: str, doc_b: str) -> Tuple[str, str]:
    system_prompt, user_prompt, final_grade_message = get_suff_prompts(code, doc_a, doc_b)

    client = OpenAI()
    response0 = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=150,
        temperature=0,
    )
    new_prompt = user_prompt + ' ' + response0.choices[0].message.content + final_grade_message
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": new_prompt},
        ],
        max_tokens=5,
        temperature=0,
    )
    assessment = response.choices[0].message.content
    if len(assessment) != 1:
        print('GPT assessment has an unexpected form:', assessment)
        assessment = 'N'
    return assessment, response0.choices[0].message.content


def get_suff_responses(codes: List[str], docs_a: List[str], docs_b: List[str]) -> Tuple[float, List[str]]:
    assert len(docs_a) == len(docs_b) == len(codes)
    assessments = []
    for i in range(len(docs_a)):
        assessment, rationale = compare_pair_sufficiency_gpt(codes[i], docs_a[i], docs_b[i])
        assessments += [assessment]
    if sum([a in ['A', 'B'] for a in assessments]) == 0:
        score = 1/2
    else:
        score = sum([a == 'A' for a in assessments]) / sum([a in ['A', 'B'] for a in assessments])
    return score, assessments


def compute_pairwise_sufficiency_by_gpt(anchor_points: Dict[str, Any], anchor_exp_name: str,
                                        base_points: Dict[str, Any], base_exp_name, config_dict: DictConfig) -> Dict:
    assert len(anchor_points) == len(base_points), 'Lengths of matched sets of docs do not coincide'
    assert len(base_points) < config_dict.evaluation.gpt_onetime_query_limit, f'''Your config doesn\'t let you spend
     that much money on API ({len(base_points)} points vs {config_dict.evaluation.gpt_onetime_query_limit})'''

    base = []
    for data_point in anchor_points.values():
        base += [p | {DSch.ID_KEY: k} for k, p in base_points.items() if p[DSch.NAME_KEY] == data_point[DSch.NAME_KEY]]
    codes = [a[DSch.CODE_KEY] for a in anchor_points.values()]
    docs_a = [point[config_dict.evaluation.doc_to_eval_key] for point in anchor_points.values()]
    docs_b = [point[config_dict.evaluation.doc_to_eval_key] for point in base]
    score, assessments = get_suff_responses(codes, docs_a, docs_b)
    result_dict = {
        'score': ((1 - score) - 0.5) * 2,  # from -1 to 1; the higher, the better
        'anchor_name': anchor_exp_name,
        'base_file_name': base_exp_name,
        'pair_assessments': [(assessments[i] + f' ({base[i][DSch.NAME_KEY]}, {base[i][DSch.ID_KEY]})')
                             for i in range(len(base))],
        'specification': 'A for anchor being more sufficient, B for base (the current model) being more sufficient, C if comparable. The score spans from -1 to 1; larger score for more sufficient base.'
    }
    return result_dict


def compare_sufficiency_gpt(config_dict: DictConfig) -> None:
    """ Compares sufficiency pairwise to an anchor specified in config.
    Dumps a special statistics file at the end. """
    anchor_path = config_dict.evaluation.sufficiency_anchor_docs
    base_file_path = config_dict.evaluation.sufficiency_base_docs
    with open(anchor_path, 'r') as file:
        anchor = json.load(file)
        if isinstance(anchor, dict):
            anchor = anchor['test']
    with open(base_file_path, 'r') as file:
        base_ = json.load(file)
        if isinstance(base_, dict):
            base_ = base_['test']
    result_dict = compute_pairwise_sufficiency_by_gpt(anchor,
                                                      anchor_path.split('/')[-1].split('.')[0],
                                                      base_,
                                                      base_file_path.split('/')[-1].split('.')[0], config_dict)
    save_dir = config_dict.evaluation.sufficiency_stats_dir
    with open(save_dir+f'/Sufficiency_{result_dict["anchor_name"]}_VS_{result_dict["base_file_name"]}.json',
              'w') as file:
        json.dump(result_dict, file, indent=2)
