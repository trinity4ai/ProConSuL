# ProConSuL: Project Context for Code Summarization with LLMs
We propose Project Context for Code Summarization with LLMs (ProConSuL), a new framework to provide a large language model (LLM) with precise information about the code structure from program analysis methods such as a compiler or IDE language services and use task decomposition derived from the code structure. ProConSuL builds a call graph to provide the context from callees and uses a two-phase training method (SFT+ preference alignment) to train the model to use the project context. We also provide a new evaluation benchmark for C/C++ functions and a set of proxy metrics. Experimental results demonstrate that ProConSuL allows to significantly improve code summaries and reduce the number of hallucinations compared to the base model (CodeLlama-7B-instruct). We make our code and dataset available here.

[Paper](https://github.com/trinity4ai/ProConSuL) | [Installation](#installation) | [Training](#training) | [Prediction](#prediction) | [Evaluation](#evaluation)

## Installation

```
git clone https://github.com/trinity4ai/ProConSuL.git && cd ProConSuL && pip install .
```

## Training
#### SFT phase:
Open `./proconsul/configs/config.json` and set `path_after_processing: ./datasets/SFT_obfuscated_dataset.json`. Also open `proconsul/configs/train/default_config.yaml` and set `model_checkpoint: /path_to_CodeLlama-7b-inst`. Then run training: 
```
deepspeed --no_local_rank --no_python proconsul run_train
```

#### Alignment phase:
Open `./proconsul/configs/config.json` and set `path_after_processing: ./datasets/ORPO_dataset.json`. Open `proconsul/configs/train/default_config.yaml` set `training_args: ORPO` and `model_checkpoint: /path_to_SFT_checkpoint`. Run alignment:
```
deepspeed --no_local_rank --no_python proconsul run_alignment
```

## Prediction

```
proconsul run_recursive_predict
```

## Evaluation

#### Triviality and Verbosity
```
proconsul evaluate_auto_metrics
```
#### Factuality and Hallucinations
Firstly, generate critiques by GPT:
```
proconsul add_gpt_fact_halluc_responses --api-key <OPENAI_TOKEN>
```
Then open jupyter notebook(`proconsul/evaluation/labeling_studio.ipynb`) for labeling according the instruction (see `labeling_inst/labeling_instruction.md`).

#### Pairwise sufficiency
Open `./proconsul/configs/evaluation/default_config.yaml` config and specify anchor prediction `sufficiency_anchor_docs: "/path_to_anchor/summaries.json"`. 

Run evaluation:
```
proconsul compare_pairwise_sufficiency_by_gpt --api-key <OPENAI_TOKEN>
```
