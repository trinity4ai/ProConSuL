# Sufficiency QA-based metric

---

### Prerequisites

Install python 3.6+, pytest 7.3+

##### How to run

1. Install proconsul library
2. Set environmental variable: `export OPENAI_API_KEY=<you_openai_api_key>`
3. Run tests, passing prediction results in JSON-files as --predict-results parameters.
   Tests will be run on test_dataset.json files located in [test_datasets](./../test_datasets) directory.
   You can change `--output-directory` that will be created in the current by default. For example:

`python -m proconsul.evaluation.qa_tests.labelled_dots_tests  --predict-results=/path_to_predicts.json --output-directory=/path_to_output_dir`

4. After the run in the output directory there will be created `qa_tests_results.json`,
   containing all functions from test_datasets, all corresponding information from prediction results,
   all scores and answers from model. Also, there will be `scores.csv` separately.
