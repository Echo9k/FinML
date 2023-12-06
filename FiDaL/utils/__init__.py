import os
import json
import pandas as pd


def load_config(config_file_path):
    """Loads a JSON configuration file."""
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def check_local_data(data_directory, ticker):
    """Checks if the data for a given ticker is available locally."""
    file_path = os.path.join(data_directory, f'{ticker}.parquet')
    if os.path.isfile(file_path):
        return pd.read_parquet(file_path)
    else:
        return None