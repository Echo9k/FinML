import os
import json
import pandas as pd
from statsmodels.tsa.stattools import adfuller


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


def ensure_data_directory(data_directory):
    """Ensures the existence of a data directory.

    Args:
        data_directory (str): The path to the data directory.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created and does not exist.

    Examples:
        >>> ensure_data_directory('data')
    """
    os.makedirs(data_directory, exist_ok=True)


def test_stationarity(timeseries, verbose=False):
    """Perform Dickey-Fuller test and return the results in a dictionary.
    Optionally print detailed results if verbose is True.

    Args:
    timeseries (pd.Series): The time series data.
    verbose (bool, optional): If True, prints detailed results. Default is False.
    """
    # Perform Dickey-Fuller test
    dftest = adfuller(timeseries, autolag='AIC')

    # Create a dictionary for the test results
    dfoutput = {
        'Test Statistic': dftest[0],
        'p-value': dftest[1],
        '#Lags Used': dftest[2],
        'Number of Observations Used': dftest[3],
        'Critical Values': dftest[4]
    }

    # Add interpretation
    interpretation = 'likely non-stationary'
    if dfoutput['p-value'] < 0.05 and dfoutput['Test Statistic'] < min(dfoutput['Critical Values'].values()):
        interpretation = 'likely stationary'
    dfoutput['Interpretation'] = interpretation

    # Print detailed results if verbose
    if verbose:
        print(f'Results of Dickey-Fuller Test:\n'
              f'  Test Statistic: {dfoutput["Test Statistic"]:.4f}\n'
              f'  p-value: {dfoutput["p-value"]:.4f}\n'
              f'  Interpretation: {interpretation}.')

    return dfoutput