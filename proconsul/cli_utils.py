import os.path

import click
from hydra import initialize_config_dir, compose


config_dir_option = click.option(
    '-d',
    '--config-dir',
    default='./proconsul/configs/',
    type=click.Path(exists=True),
    help='Path to config directory.',
)

config_name_option = click.option(
    '-n',
    '--config-name',
    default='config',
    type=click.STRING,
    help='Main config name.'
)


def create_hydra_config(config_dir: str, config_name: str):
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), job_name='proconsul'):
        config_dict = compose(config_name=config_name)
    return config_dict
