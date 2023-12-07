from io import StringIO
import os
import pandas as pd
import requests
import logging

from FiDaL.utils import load_config
from FiDaL.data.process import ensure_data_directory, clean_column_names

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloaderBase:
    """Base class for downloading data from different financial data APIs."""

    def __init__(self, config: [str, dict], *, credentials: dict = None, data_directory: str = './data/'):
        """Initializes DataDownloaderBase with config and credentials."""
        self.config = load_config(config) if isinstance(config, str) else config
        self.credentials = credentials or {}
        self.data_directory = self.config.get('path.data', data_directory)
        ensure_data_directory(self.data_directory)


    def check_local_data(self, ticker: str) -> pd.DataFrame:
        """Checks for the presence of local data for a given ticker."""
        filename = os.path.join(self.data_directory, f"{ticker}.parquet")
        return pd.read_parquet(filename) if os.path.exists(filename) else None


    def save_data(self, data: pd.DataFrame, ticker: str) -> None:
        """Saves downloaded data to a local file in Parquet format."""
        filename = os.path.join(self.data_directory, f"{ticker}.parquet")
        data.to_parquet(filename, compression='brotli')


    def get_tickers_data(self, tickers: list[str]) -> dict:
        """Downloads or retrieves data for a list of tickers."""
        stocks_df = {}
        for ticker in tickers:
            logger.info(f"Processing data for {ticker}...")
            data = self.check_local_data(ticker)
            if data is None:
                data = self.get_data(ticker)
            if data is not None:
                stocks_df[ticker] = data.sort_index()
        return stocks_df


    def get_data(self, tickers: str) -> pd.DataFrame:
        """Method to be implemented in subclass for downloading data for a ticker."""
        raise NotImplementedError("This method should be implemented in a subclass")


class VantageDataDownloader(DataDownloaderBase):
    """Downloader for Alpha Vantage data."""

    def __init__(self, config_file_path: str, *, credentials: dict = None, data_directory: str = './data/'):
        """Initializes VantageDataDownloader with config and credentials."""
        super().__init__(config_file_path, credentials=credentials, data_directory=data_directory)
        self.api_key = self.credentials.get('API', {}).get('key')
        print(self.api_key)
        if not self.api_key:
            raise ValueError("API key is missing in the credentials.")
        self.ts = self.setup_alpha_vantage(self.api_key)

    @staticmethod
    def setup_alpha_vantage(api_key: str):
        """Sets up the Alpha Vantage TimeSeries object."""
        from alpha_vantage.timeseries import TimeSeries
        return TimeSeries(key=api_key, output_format='pandas')

    def get_data(self, tickers: str, data_type: str = 'daily_adjusted', save_data:bool=False) -> pd.DataFrame:
            """
            Downloads historical financial data for a given ticker symbol.

            Parameters:
                ticker (str): The ticker symbol of the financial instrument.
                data_type (str, optional): The type of data to download. Can be 'daily' or 'daily_adjusted'. 
                                          Defaults to 'daily_adjusted'.

            Returns:
                pd.DataFrame: A pandas DataFrame containing the downloaded data.

            Raises:
                ValueError: If an invalid data type is provided.

            Examples:
                >>> data = download_data('AAPL', 'daily_adjusted')
                >>> print(data.head())
                       Date        Open        High         Low       Close   Adj Close    Volume  Dividend  Split
                0  2021-01-04  133.520004  133.610001  126.760002  129.410004  128.997803  143301900         0      1
                1  2021-01-05  128.889999  131.740005  128.429993  131.009995  130.592697   97664900         0      1
                2  2021-01-06  127.720001  131.050003  126.379997  126.599998  126.196747  155088000         0      1
                3  2021-01-07  128.360001  131.630005  127.860001  130.919998  130.502991  109578200         0      1
                4  2021-01-08  132.429993  132.630005  130.229996  132.050003  131.628159  105158200         0      1
            """
            if data_type == 'daily_adjusted':
                data = self._get_daily_adjusted_data(tickers)
            elif data_type == 'daily':
                data = self._get_daily_data(tickers)
            else:   
                raise ValueError(f"Invalid data type: {data_type}")
            if save_data and data is not None:
                self.save_data(data, tickers)
            return data


    def _get_daily_data(self, ticker: str) -> pd.DataFrame:
        """Downloads standard daily data."""
        try:
            data, _ = self.ts.get_daily(symbol=ticker, outputsize='full')
            data.columns = clean_column_names(data.columns)
            return data
        except Exception as e:
            logger.error(f"Error downloading daily data for {ticker}: {e}")

    def _get_daily_adjusted_data(self, ticker: str) -> pd.DataFrame:
        """Downloads daily adjusted data."""
        try:
            parameters = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',
                'datatype': 'csv',
                'apikey': self.api_key
            }
            response = requests.get('https://www.alphavantage.co/query', params=parameters)
            response.raise_for_status()
            
            csv_text = StringIO(response.text).read()
            
            if "You may subscribe to any of the premium plans" in csv_text:
                logger.error(f"API response issue for {ticker}: Premium plan required.")
                return None
            return pd.read_csv(csv_text)
        except Exception as e:
            logger.error(f"Error downloading daily adjusted data for {ticker}: {e}")


class YFDataDownloader(DataDownloaderBase):
    """Downloader for Yahoo Finance data.

    Inherits from DataDownloaderBase and implements downloading functionality for the Yahoo Finance API.
    """

    def get_data(self, tickers, start_date=None, end_date=None, interval='1d', save_data:bool=False):
        """Downloads data for a given ticker using the Yahoo Finance API.

        Args:
            ticker (str): Ticker symbol of the security.
            start_date (str, optional): Start date for the data retrieval in the format YYYY-MM-DD. Defaults to None.
            end_date (str, optional): End date for the data retrieval in the format YYYY-MM-DD. Defaults to None.
            interval (str): Data interval. Valid intervals: '1d', '1wk', '1mo', etc. Defaults to '1d'.

        Returns:
            pd.DataFrame: DataFrame with the downloaded data.
        """
        import yfinance as yf
        try:
            data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
            if save_data and data is not None:
                self.save_data(data, tickers)
            return data
        except Exception as e:
            print(f"An error occurred while downloading data for {tickers}: {e}")