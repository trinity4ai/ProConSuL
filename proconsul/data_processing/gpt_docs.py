from typing import Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion


def get_gpt_prompt(code: str, cxt: str = '') -> Tuple[str, str]:
    system_prompt = f'''You are a knowledgeable C/C++ code expert. \
Your task is to assist me with an abstractive code summarization task. \
I need you to provide an example of a well-thought-out, comprehensive, yet concise function summary. \
Write 1 to 4 short sentences that summarize the function below. \
If applicable, describe the purpose and effects of the code, but omit unnecessary details. \
Focus on abstraction and highlighting key points: be as terse as possible, like a Terminator.'''
    user_prompt = f'''Below is some information gathered from our code repository. \
To help you create a comprehensive summary, we provide additional context: a list of callees with their docstrings. \
This might help you understand the broader context of our project. \
The structure is as follows: a list of callees with docstrings, followed by the function code, and then a blank line for your response. \
The function and the list of callees are delimited with XML tags for clarity (<code> and <callees list> respectively).

<callees list>
{cxt.strip()} </callees list>

<code>
{code.strip()} </code>

Write a concise function summary below (only 1-4 sentences, as if you are a Terminator):'''
    return system_prompt, user_prompt


def get_gpt_doc(code: str,
                cxt: str,
                max_tokens: int = 140) -> Tuple[ChatCompletion, str]:
    system_prompt, user_prompt = get_gpt_prompt(code, cxt)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
        stop='\n',
    )
    return response, user_prompt
