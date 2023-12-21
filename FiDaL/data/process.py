import os
import pandas as pd
import numpy as np


def __clean_column_names(columns):
    """Cleans column names by removing any numerical prefix.

    Args:
        columns (list of str): A list of column names from Alpha Vantage.

    Returns:
        list of str: A list of cleaned column names with numerical prefixes removed.

    Examples:
        >>> __clean_column_names(['1. open', '2. high'])
        ['open', 'high']
    """
    return [col.split('. ')[1] if '. ' in col else col for col in columns]


def normalize_col(data):
    """Normalizes a pandas series by scaling the data between 0 and 1.

    Args:
        data (pd.Series): The input pandas series to normalize.

    Returns:
        pd.Series: Normalized pandas series.

    Examples:
        >>> normalize_col(pd.Series([1, 2, 3, 4, 5]))
    """
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


def normalize_dataframe(dataframe, columns=None):
    """Normalizes specified columns of a DataFrame.

    Args:
        dataframe (pd.DataFrame): The dataframe to normalize.
        columns (list of str, optional): List of column names to normalize. Defaults to None, in which case all numeric columns are normalized.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> normalize_dataframe(df, ['A'])
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = columns
    dataframe[numeric_cols] = dataframe[numeric_cols].apply(normalize_col)
    return dataframe


class FeatureGenerator:
    """Class for generating features from stock price data."""

    @staticmethod
    def calculate_returns(data, column='Adj Close'):
        """
        Calculate simple and logarithmic returns for all tickers in the DataFrame.

        Parameters:
        data (pandas.DataFrame): DataFrame with 'Adj Close' as one of the columns
                                 containing adjusted close prices for each ticker.

        Returns:
        pandas.DataFrame: Updated DataFrame including 'return' and 'log_return'
                          for each ticker.
        """
        if FinancialDataProcessor.is_multiindex(data):
            # Assuming 'data' is a DataFrame where each column represents a ticker's 'colums' (price | Adj Close) data
            returns = {}
            for ticker in data.columns.get_level_values(0).unique(): 
                ticker_data = data.xs(ticker, level=0, axis=1)  # Adjust 'level' as needed
                returns[ticker] = FinancialDataProcessor._compute_returns(ticker_data, column=column,
                                                                          log=log,
                                                                          apply_smoothing=apply_smoothing,
                                                                          smoothing_factor=smoothing_factor)

            # Concatenate all returns DataFrames along the columns
            return pd.concat(returns, axis=1)
        else:
            return FinancialDataProcessor._compute_returns(data, column=column,
                                                           log=log,
                                                           apply_smoothing=apply_smoothing,
                                                           smoothing_factor=smoothing_factor)

    @staticmethod
    def _is_multiindex(data):
        """Check if the DataFrame has MultiIndex columns."""
        return isinstance(data.columns, pd.MultiIndex)

    @staticmethod
    def _check_column(data, column, is_multiindex):
        """Ensure the specified column is present in the DataFrame."""
        if is_multiindex and column not in data.columns.get_level_values(0):
            raise ValueError(f"'{column}' column not found in the DataFrame")
        if not is_multiindex and column not in data.columns:
            raise ValueError(f"'{column}' column not found in the DataFrame")

    @staticmethod
    def _compute_returns(data, column, is_multiindex):
        """Compute and insert simple and logarithmic returns."""
        if is_multiindex:
            adj_close = data[column]
            simple_returns = adj_close / adj_close.shift(1) - 1
            log_returns = np.log(adj_close / adj_close.shift(1))

            for ticker in adj_close.columns:
                data[('returns', ticker)] = simple_returns[ticker]
                data[('log_returns', ticker)] = log_returns[ticker]
        else:
            adj_close = data[column]
            data['returns'] = adj_close / adj_close.shift(1) - 1
            data['log_returns'] = np.log(adj_close / adj_close.shift(1))

        return data.sort_index(axis=1) if is_multiindex else data


class PriceAdjuster:
    def adjust_price_for_splits(df, column, split_ratios):
        """Adjust stock prices in the DataFrame for stock splits.

        Args:
            df (pd.DataFrame): DataFrame containing stock prices.
            column (str): Name of the column with stock prices to adjust.
            split_ratios (dict): Dictionary where keys are split dates (YYYY-MM-DD) and values are split ratios.

        Returns:
            pd.DataFrame: DataFrame with the adjusted stock prices.

        Examples:
            >>> df = pd.DataFrame({'price': [120, 60]}, index=['2020-01-01', '2020-01-02'])
            >>> adjust_price_for_splits(df, 'price', {'2020-01-02': 2})
        """
        adjusted_df = df.copy()
        adjusted_df.sort_index(inplace=True)
        for split_date, ratio in sorted(split_ratios.items(), reverse=True):
            split_date = pd.Timestamp(split_date)
            if split_date in adjusted_df.index:
                adjusted_df.loc[:split_date, column] /= ratio
        return adjusted_df

    def adjust_price_for_dividends(df, dividends):
        """Adjusts the close prices in the DataFrame for dividends.

        Args:
            df (pd.DataFrame): DataFrame containing stock prices.
            dividends (dict): Dictionary with ex-dividend dates as keys and the dividend amounts as values.

        Returns:
            pd.DataFrame: DataFrame with a new 'adjusted_close' column containing the adjusted close prices.

        Examples:
            >>> df = pd.DataFrame({'close': [100, 102, 101]}, index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
            >>> adjust_price_for_dividends(df, {'2020-01-02': 1})
        """
        adjusted_df = df.copy()
        adjusted_df.sort_index(inplace=True)
        adjusted_df['adjusted_close'] = adjusted_df['close'].copy()
        for ex_date, dividend in dividends.items():
            ex_date = pd.Timestamp(ex_date)
            if ex_date in adjusted_df.index:
                adjusted_df.loc[:ex_date, 'adjusted_close'] -= dividend
        return adjusted_df
