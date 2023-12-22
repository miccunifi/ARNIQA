import argparse
import yaml
from dotmap import DotMap
from functools import reduce
from operator import getitem
from distutils.util import strtobool
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

def parse_config(config_file_path: str) -> DotMap:
    """Parse the YAML configuration file"""
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config, _dynamic=False)


def parse_command_line_args(config: DotMap) -> DotMap:
    """Parse the command-line arguments"""
    parser = argparse.ArgumentParser()

    # Automatically add command-line arguments based on the config structure
    def add_arguments(section, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                add_arguments(value, prefix=full_key)
            else:
                # Check if the value is a list
                if isinstance(value, list):
                    # Convert list to comma-separated string
                    parser.add_argument(f'--{full_key}', default=value, type=type(value[0]), nargs='+', help=f'Value for {full_key}')
                else:
                    if type(value) == bool:
                        parser.add_argument(f'--{full_key}', default=value, type=strtobool, help=f'Value for {full_key}')
                    else:
                        parser.add_argument(f'--{full_key}', default=value, type=type(value), help=f'Value for {full_key}')

    add_arguments(config)

    args, _ = parser.parse_known_args()
    args = DotMap(vars(args), _dynamic=False)
    return args


def merge_configs(config: DotMap, args: DotMap) -> DotMap:
    """Merge the command-line arguments into the config. The command-line arguments take precedence over the config file
    :rtype: object
    """
    keys_to_modify = []

    def update_config(config, key, value):
        *keys, last_key = key.split('.')
        reduce(getitem, keys, config)[last_key] = value

    # Recursively merge command-line parameters into the config
    def get_updates(section, args, prefix=''):
        for key, value in section.items():
            full_key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                get_updates(value, args, prefix=full_key)
            elif getattr(args, full_key, None) or getattr(args, full_key, None) != getattr(section, key, None):
                keys_to_modify.append((full_key, getattr(args, full_key)))

    get_updates(config, args)

    for key, value in keys_to_modify:
        update_config(config, key, value)

    return config

