import os

import click

from proconsul.cli_utils import config_dir_option, config_name_option, create_hydra_config


@click.group()
def cli():
    pass


@cli.command(name='merge_json', short_help='merge all datasets from given directory into one file')
@click.option('-i', '--input', 'input_dir', type=click.Path(exists=True), help='Path to directory with JSON files.')
@click.option('-o', '--output', 'output_file', type=click.Path(), help='Path to store final JSON.')
@click.option('-c', '--csv', 'csv_file', type=click.Path(exists=True),
              help='File with information about status of repository.')
def cli_merge_json(input_dir, output_file, csv_file):
    from proconsul.datasets_extraction.merge import merge_json

    merge_json(input_dir, output_file, csv_file)


@cli.command(name='find_duplicates', short_help='')
@click.option('-i', '--input', type=click.Path(exists=True), help='Path to JSON with input dataset.')
@click.option('-o', '--output', type=click.Path(), help='Path to store results.')
@click.option('-t', '--threshold', type=float, default=0.9, help='Jaccard similarity threshold - '
                                                                 'if similarity is higher - add to similar items.')
@click.option('-d', '--debug', is_flag=True, default=False, help='Debug launch.')
@click.option('-b', '--batch', type=int, default=None, help='batch_size to use during querying the index.')
@click.option('-u', '--usevendor', is_flag=True, default=False, help='If true - vendor functions will be used.')
def cli_find_duplicates(input, output, threshold, debug, batch, usevendor):
    from proconsul.datasets_extraction.nearest_neighbours import find_duplicates

    find_duplicates(input, output, threshold, debug, batch, usevendor)


@cli.command(name='convert_data_to_array', short_help='convert dict[id, value] to {"split_key": list_of_values}')
@click.option('-i', '--input', required=True, type=click.Path(exists=True), help='Path to input dataset.')
@click.option('-o', '--output', required=True, type=click.Path(exists=False), help='Path to output dataset.')
@click.option('-k', '--key', default='test', type=str, help='Split key')
def cli_convert_data_to_array(input: str, output: str, key: str) -> None:
    from proconsul.data_processing.utils_data_processing import convert_data_to_array

    convert_data_to_array(input, output, key)


@cli.command(name='get_subset', short_help='get subtree from chosen nodes for recursive generation. Input dataset format is Dict[str, value].')
@config_dir_option
@config_name_option
def cli_get_subset(config_dir: str, config_name: str):
    from proconsul.data_processing.rec_sum_utils.get_subset import run_get_subset

    run_get_subset(create_hydra_config(config_dir, config_name))


@cli.command(name='merge_and_dedup', short_help='merge repos, split into train and test, delete test dups from train')
@config_dir_option
@config_name_option
def cli_merge_and_dedup(config_dir: str, config_name: str):
    from proconsul.datasets_extraction.merge_and_dedup import run_merge_and_dedup

    run_merge_and_dedup(create_hydra_config(config_dir, config_name))


@cli.command(name='get_context', short_help='generate context and filter data')
@config_dir_option
@config_name_option
def cli_get_context(config_dir: str, config_name: str):
    from proconsul.data_processing.get_data_cxt import get_data_cxt

    get_data_cxt(create_hydra_config(config_dir, config_name))


@cli.command(name='obfuscate', short_help='obfuscates code and context in train dataset')
@config_dir_option
@config_name_option
def cli_obfuscate(config_dir: str, config_name: str):
    from proconsul.data_processing.obfuscate import obfuscate

    obfuscate(create_hydra_config(config_dir, config_name))


@cli.command(name='dedup_train', short_help='remove duplicates from train dataset')
@config_dir_option
@config_name_option
def cli_dedup_train(config_dir: str, config_name: str):
    from proconsul.datasets_extraction.nearest_neighbours import run_deduplicate_train

    run_deduplicate_train(create_hydra_config(config_dir, config_name))


@cli.command(name='run_train', short_help='run model training')
@config_dir_option
@config_name_option
def cli_run_train(config_dir: str, config_name: str):
    from proconsul.train.train import run_train

    run_train(create_hydra_config(config_dir, config_name))


@cli.command(name='run_predict', short_help='run model inference on val and test')
@config_dir_option
@config_name_option
def cli_run_predict(config_dir: str, config_name: str):
    from proconsul.train.predict import run_predict

    run_predict(create_hydra_config(config_dir, config_name))


@cli.command(name='run_recursive_predict', short_help='run model inference on val and test with recursive generation')
@config_dir_option
@config_name_option
def cli_run_recursive_predict(config_dir: str, config_name: str):
    from proconsul.train.predict import run_recursive_predict

    run_recursive_predict(create_hydra_config(config_dir, config_name))


@cli.command(name='run_synthesize', short_help='clean and filter dataset, run doc synthesis on train split')
@config_dir_option
@config_name_option
def cli_run_synthesize(config_dir: str, config_name: str):
    from proconsul.synth.synthesize import run_synthesize

    run_synthesize(create_hydra_config(config_dir, config_name))


@cli.command(
    name="run_finalize_synth",
    short_help="clean and process synthesized datapoints, add context, extract target datapoints and run train dedup",
)
@config_dir_option
@config_name_option
def cli_run_finalize_synth(config_dir: str, config_name: str):
    from proconsul.synth.synthesize import run_finalize_synth

    run_finalize_synth(create_hydra_config(config_dir, config_name))


@cli.command(name='evaluate_auto_metrics',
             short_help='evaluate automatic metrics and add to json with predictions and save it to downstream dir')
@config_dir_option
@config_name_option
def cli_evaluate_auto_metrics(config_dir: str, config_name: str):
    from proconsul.evaluation.eval_entrypoints import evaluate_auto_metrics

    evaluate_auto_metrics(create_hydra_config(config_dir, config_name))


@cli.command(name='add_gpt_fact_halluc_responses',
             short_help='add GPT-4 responses on factuality and hallucinations in docstings to evaluate')
@click.option('-k', '--api-key', required=True, type=str, help='OpenAI API key')
@config_dir_option
@config_name_option
def cli_add_gpt_fact_halluc_responses(api_key: str, config_dir: str, config_name: str):
    from proconsul.evaluation.eval_entrypoints import add_gpt_fact_halluc_responses

    os.environ['OPENAI_API_KEY'] = api_key
    add_gpt_fact_halluc_responses(create_hydra_config(config_dir, config_name))


@cli.command(name='compare_pairwise_sufficiency_by_gpt',
             short_help='match 2 sets of docstrings on sufficiency pairwise (based on gpt4)')
@click.option('-k', '--api-key', required=True, type=str, help='OpenAI API key')
@config_dir_option
@config_name_option
def cli_compare_pairwise_sufficiency_by_gpt(api_key: str, config_dir: str, config_name: str):
    from proconsul.evaluation.pairwise_sufficiency_gpt import compare_sufficiency_gpt

    os.environ['OPENAI_API_KEY'] = api_key
    compare_sufficiency_gpt(create_hydra_config(config_dir, config_name))


@cli.command(name='merge_all_metrics',
             short_help='merge auto and semi-auto metrics. Files must consist of the same datapoints in order.')
@config_dir_option
@config_name_option
def cli_merge_all_metrics(config_dir: str, config_name: str):
    from proconsul.evaluation.eval_entrypoints import merge_all_metrics

    merge_all_metrics(create_hydra_config(config_dir, config_name))


if __name__ == '__main__':
    cli()
