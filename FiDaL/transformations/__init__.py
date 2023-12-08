from statsmodels.tsa.stattools import adfuller
import pandas as pd


def test_stationarity(timeseries):
    """Perform Dickey-Fuller test and print the results."""
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(
        dftest[:4],
        index=[
            'Test Statistic',
            'p-value',
            '#Lags Used',
            'Number of Observations Used'
            ]
        )


    print(dfoutput)
    print('Critical Values:')
    for key, value in dftest[4].items():
        print(f'\t{key}: {value}')