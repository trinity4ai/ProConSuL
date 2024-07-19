from typing import Any, Dict, Tuple

from omegaconf import DictConfig
from openai import OpenAI
from openai.types.chat import ChatCompletion

from proconsul.datasets_extraction.dataset_scheme import DatasetScheme


def get_fact_halluc_prompt(code: str, doc: str, cxt_used: str = '') -> Tuple[str, str]:
    system_prompt = f'You are a knowledgeable C/C++ code expert. \
You are here to help your colleagues with abstractive code summarization task. \
Your answers should be concise and substantial. Follow your instructions strictly. \
Try to give your answers in the form of a short list. \
Your colleagues would appreciate it if you give a short and accurate answer.'
    user_prompt = f'''Below we have a C/C++ code of a function and a docstring for that function \
(delimited with XML tags). \
We need to decide whether this function docstring gives a factual high-level summary of the code. \
Patiently go over each statement from this function docstring. \
Then give a list of details this docstring gets wrong - \
if it makes a mistake and says something that is not true - tell us; \
start by providing a short quotation. \
Also, mention if the docstring contains hallucinations - statements that can not be extracted from \
the given code or general context; give an explanation. \
Recall that the purpose of this docstring is a high-level summarization, \
so don't expect a comprehensive code summary. \
If the docstring omits details, it is fine, it is not a mistake or disadvantage from our perspective, \
do not mention it in your review.
Answer template example:
Wrong details:
    - ...
Statements from the docstring that can not be extracted from the given code or general context:
    - ...

<code>
{code.strip()} </code>

<docstring>
{doc.strip()} </docstring>'''
    return system_prompt, user_prompt


def ask_fact_halluc_gpt(code: str, doc: str, max_tokens: int = 500) -> Tuple[ChatCompletion, str]:
    system_prompt, user_prompt = get_fact_halluc_prompt(code, doc)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response, user_prompt


def get_gpt_fact_halluc_responses(
        data: Dict[str, Dict[str, Any]],
        eval_config: DictConfig
) -> Dict[str, Dict[str, str]]:
    doc_to_eval_key = eval_config.doc_to_eval_key
    response_key = eval_config.llm_messages['fact_halluc_response']
    prompt_key = eval_config.llm_messages['fact_halluc_user_prompt']
    messages = dict()
    for key, point in data.items():
        summ = point[doc_to_eval_key]
        code = point[DatasetScheme.CODE_KEY]
        response, user_prompt = ask_fact_halluc_gpt(code=code, doc=summ)
        messages[key] = {
            response_key: response.choices[0].message.content,
            prompt_key: user_prompt,
        }
    return messages
