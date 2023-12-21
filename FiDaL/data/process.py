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


class DataNormalization:
    @staticmethod
    def normalize_col(data):
        """
        Normalizes a pandas series by scaling the data between 0 and 1.

        Args:
            data (pd.Series): The input pandas series to normalize.

        Returns:
            pd.Series: Normalized pandas series.
        """
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val)

    @staticmethod
    def normalize_dataframe(dataframe, columns=None):
        """
        Normalizes specified columns of a DataFrame.

        Args:
            dataframe (pd.DataFrame): The dataframe to normalize.
            columns (list of str, optional): List of column names to normalize.
                                             Defaults to None, normalizing all numeric columns.

        Returns:
            pd.DataFrame: DataFrame with normalized columns.
        """
        if columns is None:
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        dataframe[numeric_cols] = dataframe[numeric_cols].apply(DataNormalization.normalize_col)
        return dataframe


class FinancialDataProcessor:
    @staticmethod
    def _laplace_smoothing(data, smoothing_factor=1):
        return (data + smoothing_factor) / (data.sum() + smoothing_factor * len(data.unique()))

    @staticmethod
    def _compute_returns(data, column, log, apply_smoothing, smoothing_factor):
        try:
            # Ensure 'data' is a separate DataFrame to avoid SettingWithCopyWarning
            data = data.copy()  

            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

            adj_close = data[column]
            # Use .loc for setting new columns
            data.loc[:, 'returns'] = adj_close / adj_close.shift(1) - 1
            if log:
                data.loc[:, 'log_returns'] = np.log(adj_close / adj_close.shift(1))

            if apply_smoothing:
                data.loc[:, 'returns'] = FinancialDataProcessor._laplace_smoothing(data['returns'], smoothing_factor=smoothing_factor)
                if log:
                    data.loc[:, 'log_returns'] = FinancialDataProcessor._laplace_smoothing(data['log_returns'], smoothing_factor=smoothing_factor)
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
        

    @staticmethod
    def compute_returns(data, column='Adj Close', log=True, apply_smoothing=True, smoothing_factor=1):
        """
        Compute and insert simple and logarithmic returns with optional Laplace smoothing/correction.

        Args:
            data (pandas.DataFrame): DataFrame with stock price data.
            column (str): Name of the column to compute returns on. Defaults to 'Adj Close'.
            log (bool): If True, computes logarithmic returns. Defaults to True.
            apply_smoothing (bool): If True, applies Laplace smoothing. Defaults to False.
            smoothing_factor (int): Smoothing factor for Laplace smoothing. Defaults to 1.

        Returns:
            pandas.DataFrame: DataFrame with computed returns.
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
    def is_multiindex(data):
        """
        Check if the DataFrame has MultiIndex columns.
        """
        return isinstance(data.columns, pd.MultiIndex)

    @staticmethod
    def check_column(data, column):
        """
        Ensure the specified column is present in the DataFrame.
        """
        if FinancialDataProcessor.is_multiindex(data) and column not in data.columns.get_level_values(0):
            raise ValueError(f"'{column}' column not found in the DataFrame")
        if not FinancialDataProcessor.is_multiindex(data) and column not in data.columns:
            raise ValueError(f"'{column}' column not found in the DataFrame")


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
