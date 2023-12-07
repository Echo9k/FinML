import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from FiDaL.plot import plot_autocorrelation

# Create a fixture for the data that will be used in the tests
@pytest.fixture
def mock_data():
    dates = pd.date_range('20210101', periods=60)
    data = pd.DataFrame(np.random.randn(60, 4), index=dates, columns=list('ABCD'))
    data = pd.concat([data], keys=['Adj Close'], axis=1)
    return data

# Happy path tests with various realistic test values
@pytest.mark.parametrize("ticker, lags, test_id", [
    ('A', 20, 'happy_path_default_lags'),
    ('B', 10, 'happy_path_lower_lags'),
    ('C', 60, 'happy_path_higher_lags'),
])
def test_plot_autocorrelation_happy_path(mock_data, ticker, lags, test_id):
    # Arrange
    data = mock_data

    # Act
    with patch('matplotlib.pyplot.show'):
        plot_autocorrelation(data, ticker, lags=lags)

    # Assert
    # Since the function is for plotting, we assert that no exception is raised and the function completes

# Edge cases
@pytest.mark.parametrize("ticker, lags, test_id", [
    ('D', 0, 'edge_case_zero_lags'),
    ('A', -1, 'edge_case_negative_lags'),
    ('B', 1.5, 'edge_case_non_integer_lags'),
])
def test_plot_autocorrelation_edge_cases(mock_data, ticker, lags, test_id):
    # Arrange
    data = mock_data

    # Act & Assert
    with pytest.raises(ValueError), patch('matplotlib.pyplot.show'):
        plot_autocorrelation(data, ticker, lags=lags)

# Error cases
@pytest.mark.parametrize("ticker, lags, test_id", [
    (123, 40, 'error_case_non_string_ticker'),
    ('E', 40, 'error_case_nonexistent_ticker'),
    ('A', 'forty', 'error_case_string_lags'),
])
def test_plot_autocorrelation_error_cases(mock_data, ticker, lags, test_id):
    # Arrange
    data = mock_data

    # Act & Assert
    with pytest.raises((KeyError, TypeError)), patch('matplotlib.pyplot.show'):
        plot_autocorrelation(data, ticker, lags=lags)
