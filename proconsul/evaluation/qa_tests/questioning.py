import os
import re
from typing import List, Dict

from openai import OpenAI

yes_or_no_pattern = re.compile(r'(Yes|No)')
yes_or_no_with_dot_pattern = re.compile(r'(Yes|No)\.* *')


def extract_answers(questions: List[str], response: str) -> Dict[str, Dict[str, str]]:
    print(f"\n[DEBUG] {response}")
    answers = re.findall(yes_or_no_pattern, response)
    comments_str = re.sub(yes_or_no_with_dot_pattern, "", response)
    comments = re.split(r"\n", comments_str)
    if len(answers) != len(questions):
        raise ValueError("Number of answers {} does not match number of questions {}\n\n"
                         .format(len(answers), len(questions)))
    output = {}
    for i, q in enumerate(questions):
        output[q] = {"answer": answers[i], "comment": comments[i]}
    return output


def ask_gpt_4_turbo_and_get_response(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content


def ask_questions_on_docstring(docstring: str, claims: List[str], question_start: str) -> Dict[str, Dict[str, str]]:
    system_prompt = """
    You are an expert in Programming. 
    You are here to help your colleagues with abstractive code summarization task.
    For your help to be effective you need to follow given instructions strictly. 
    Your task is to answer Yes or No to every question using only the information in the provided docstring.
    You should use the provided docstring as the only source of truth.
    Give a separate answer to every question in order.
    One answer per question on separate lines. Answer only Yes or No.
    If you are unsure about the answer to a question, add a comment to your answer on the same line.
    """
    docstring_str = "Docstring:\n\"\"\"{}\"\"\"\n\n".format(docstring)
    questions_str = "Questions:\n" + "\n".join([f"{question_start} {q}?" for q in claims])
    response = ask_gpt_4_turbo_and_get_response(system_prompt, docstring_str + questions_str)
    answers = re.split(r"\n", response)
    output = {}
    for i, q in enumerate(claims):
        output[f"{question_start} {q}?"] = answers[i]
    return output


def ask_sufficient_docstring_on_code_gpt(code: str) -> str:
    system_prompt = f'You are a knowledgeable C/C++ code expert. \
    You are here to help your colleagues with abstractive code summarization task. \
    Your answers should be concise and substantial. Follow your instructions strictly. \
    Try to give your answers in the form of a short list. \
    Your colleagues would appreciate it if you give a short and accurate answer.'
    user_prompt = f'''Below we have a C/C++ code of a function. \
    We need to get the high-level summary of the code. \
    There should be enough information to correctly understand what the function does. \
    Recall that the purpose of this docstring is a high-level summarization, \
    so a comprehensive code summary is not expected. \
    If the docstring omits details, it is fine, it is not a mistake or disadvantage.\
    The facts should be concise and substantial. \
    Answer template example:
    Docstring:
        - ...
        - ...

    <code>
    {code.strip()} </code>'''
    return ask_gpt_4_turbo_and_get_response(system_prompt, user_prompt)


def extract_facts_from_docstring(code: str, docstring: str) -> str:
    system_prompt = f'You are a knowledgeable C/C++ code expert. \
    You are here to help your colleagues to extract independent facts from function description. \
    Try to give your answers in the form of a short list. \
    Your colleagues would appreciate it if you give a short and accurate answer.'
    user_prompt = f"""You will be given a function code and a sentence describing it. \
    Please breakdown the sentence into independent claims.\
Example:
<code>
void sort(unsigned int numbers[], int size)
{{
    int i;
    int j;
    int min_index;
    unsigned int temp;
    
    for (i = 0; i < size-1; i++) {{
        min_index = i;
        for (j = i+1; j < size; j++) {{
            if (numbers[j] < numbers[min_index]) {{
                min_index = j;
            }}
        }}
        temp = numbers[i];
        numbers[i] = numbers[min_index];
        numbers[min_index] = temp;
    }}
}}
</code>
<sentence>
This function implements the Selection Sort algorithm to sort a given array of type unsigned integer in ascending order.
</sentence>
<claims>
- This function implements the Selection Sort algorithm.
- This function sorts an array.
- Given array is of type unsigned integer.
- Sort is made in ascending order.
</claims>

<code>
{code.strip()}</code>
<sentence>
{docstring.strip()}</sentence>
<claims>
</claims>"""
    return ask_gpt_4_turbo_and_get_response(system_prompt, user_prompt)
