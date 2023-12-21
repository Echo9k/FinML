import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Function to plot time series data
def time_series(df, column, title='Stock Prices Over Time', ylabel='Price'):
    """Plots time series data for a single column."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Function to plot histogram of returns
def histogram(df, column, title='Distribution of Returns', bins=50):
    """Plots a histogram of the returns."""
    plt.figure(figsize=(12, 6))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()


# Function to plot correlation heatmap
def correlation_heatmap(df, title='Correlation Heatmap'):
    """Plots a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.show()



# Function to plot a scatter diagram
def scatter(df, column1, column2, title='Scatter Diagram'):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[column1], df[column2], alpha=0.5)
    plt.title(title)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()


def stats(product, ax):
    # Calculate mean and standard deviation
    mean_val = product.mean()
    std_dev = product.std()

    # Plot mean and 2 standard deviations
    ax.axhline(y=mean_val, color='tab:cyan', linestyle='--', label='Mean', alpha=0.7, linewidth=.95)
    ax.axhline(y=product.median(), color='tab:green', linestyle='--', label='Median', alpha=0.7, linewidth=.95)
    ax.axhline(y=mean_val + 1.5 * std_dev, color='tab:purple', linestyle=':', label='Mean + 1.5 SD', alpha=0.7)
    ax.axhline(y=mean_val - 1.5 * std_dev, color='tab:pink', linestyle=':', label='Mean - 1.5 SD', alpha=0.7)


def moving_average(adj_close, volume, window_sizes, include_stats=False, log_scale=False):
    """
    Plots the product of adjusted close price and volume with moving averages and optional statistical lines.

    Parameters:
    adj_close (pd.Series): Adjusted close prices.
    volume (pd.Series): Trading volume.
    window_sizes (list): List of integers for moving average window sizes.
    include_stats (bool): If True, include mean and 2 standard deviations in the plot.
    log_scale (bool): If True, use logarithmic scale for the y-axis.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_title('Adj Close * Volume with Moving Averages')
    ax.set_ylabel('Price $')

    # Calculate the product
    product = adj_close * volume

    # Plot the adjusted close price times the volume
    ax.plot(product, label='Adj Close * Volume', alpha=0.8, linewidth=.2, linestyle='-', color='tab:gray')

    # Plot moving averages
    for window in window_sizes:
        moving_average = product.rolling(window=window, min_periods=1).mean()
        ax.plot(moving_average, label=f'Moving Average ({window} days)', linewidth=1.2)

    if include_stats:
        stats(product, ax)
    
    # Set y-axis scale
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Date')
    plt.legend()
    plt.show()


def adj_close_volume(data, columns:list, ticker=None, y_log=False, ax=None):
    """Plot the adjusted close price and volume of a stock.

    Args:
        data (pd.DataFrame): The data to plot.
        ticker (str): The ticker of the stock.
        columns (list): The columns to plot.
        y_log (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
    """
    plt.rcParams["figure.figsize"] = (5, 2)
    
    if ticker and ticker in data.columns:
        data = data[ticker]

    # Plot the adjusted close price and volume
    if ax is None:
        fig, ax = plt.subplots(len(columns), 1, figsize=(15, 5 * len(columns)))

    for i, column in enumerate(columns):
        if ticker:
            ax[i].set_title(f'{ticker} {column}')
        else:
            ax[i].set_title(column)
        ax[i].set_ylabel('Price $' if column.lower() in {'adj close', 'price'} else 'units')
        if y_log:
            ax[i].set_yscale('log')
        ax[i].plot(data[column], color='tab:blue' if column.lower() in {'adj close', 'price'} else 'tab:orange', label=column, linewidth=1.5 if column.lower() in {'adj close', 'price'} else .9, linestyle='-', alpha=0.8)
        ax[i].grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.xlabel('Date')
    plt.show()

def autocorrelation(series, title, lags=40, figsize=(15, 5)):
    """
    Plots autocorrelation for a time series.

    Args:
    series (pd.Series): The time series data.
    title (str): The title for the plot.
    lags (int, optional): The number of lags to include. Default is 40.
    figsize (tuple, optional): Figure size as a tuple (width, height). Default is (15, 5).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_acf(series, lags=lags, ax=ax)
    ax.set_title(f'Autocorrelation for {title}')
    plt.tight_layout()
    plt.show()



def partial_autocorrelation(series, title, lags=40):
    """
    Plots partial autocorrelation for a time series.

    Args:
    series (pd.Series): The time series data.
    title (str): The title for the plot.
    lags (int, optional): The number of lags to include. Default is 40.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    plot_pacf(series, lags=lags, ax=ax)
    ax.set_title(f'Partial Autocorrelation for {title}')
    plt.tight_layout()
    plt.show()


def efficient_frontier(random_portfolios, optimal_volatility, optimal_return):
    plt.figure(figsize=(10, 6))
    plt.scatter(random_portfolios[0,:], random_portfolios[1,:], c=random_portfolios[2,:], cmap='YlGnBu', marker='o')
    plt.title('Efficient Frontier with Selected Assets')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Returns')
    plt.colorbar(label='Sharpe Ratio')

    # Plotting the optimized portfolio
    plt.scatter(optimal_volatility, optimal_return, marker='*', color='r', s=100, label='Optimized Portfolio')
    plt.legend()

    plt.show()
    

def decomposed_time_series(decomposed, ticker=None, figsize=(5, 7)):
    """
    Plots the observed, trend, seasonal, and residual components of a decomposed time series.

    Args:
    decomposed (DecomposeResult): The decomposed time series object.
    figsize (tuple): The size of the figure (width, height in inches). Defaults to (10, 6).

    Returns:
    matplotlib.figure.Figure: The matplotlib figure object.
    """

    # Create a figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Plot each component of the decomposition
    decomposed.observed.plot(ax=axes[0], title='Observed')
    decomposed.trend.plot(ax=axes[1], title='Trend')
    decomposed.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposed.resid.plot(ax=axes[3], title='Residual')

    if ticker:
        fig.suptitle(f'{ticker} | Component Decomposition')
    else:
        fig.suptitle('Component Decomposition')

    # Adjust layout
    plt.tight_layout()

    # Return the figure object for further manipulation if necessary
    return fig

def analyze_moving_averages(timeseries, short_term_lag=10, long_term_lag=20, figsize=(12, 6)):
    """
    Apply SMA and EMA models to time series data, visualize and evaluate the results.

    Args:
    timeseries (pd.Series): The time series data.
    short_term_lag (int): The number of lags for short-term moving average. Default is 10.
    long_term_lag (int): The number of lags for long-term moving average. Default is 20.
    figsize (tuple): Figure size for the plot. Default is (12, 6).

    Returns:
    None
    """
    # Apply Moving Average Models
    sma_short = timeseries.rolling(window=short_term_lag).mean()
    sma_long = timeseries.rolling(window=long_term_lag).mean()
    ema_short = timeseries.ewm(span=short_term_lag, adjust=False).mean()
    ema_long = timeseries.ewm(span=long_term_lag, adjust=False).mean()

    # Trim the original series to align with the moving average series
    start_point_short = short_term_lag - 1
    start_point_long = long_term_lag - 1
    trimmed_timeseries_short = timeseries[start_point_short:]
    trimmed_timeseries_long = timeseries[start_point_long:]

    # Visualization
    plt.figure(figsize=figsize)
    plt.plot(timeseries, label='Original', alpha=0.7, color='gray', linestyle='-', linewidth=0.2)
    plt.plot(sma_short, label=f'SMA {short_term_lag}-lags', linewidth=0.8)
    plt.plot(sma_long, label=f'SMA {long_term_lag}-lags', linewidth=0.8)
    plt.plot(ema_short, label=f'EMA {short_term_lag}-lags', linewidth=0.8)
    plt.plot(ema_long, label=f'EMA {long_term_lag}-lags', linewidth=0.8)
    plt.legend()
    plt.title('Moving Average Models')
    plt.show()

    # Evaluation using RMSE
    rmse_sma_short = mean_squared_error(trimmed_timeseries_short, sma_short[start_point_short:], squared=False)
    rmse_sma_long = mean_squared_error(trimmed_timeseries_long, sma_long[start_point_long:], squared=False)
    rmse_ema_short = mean_squared_error(trimmed_timeseries_short, ema_short[start_point_short:], squared=False)
    rmse_ema_long = mean_squared_error(trimmed_timeseries_long, ema_long[start_point_long:], squared=False)

    print(f'RMSE - SMA Short: {rmse_sma_short}')
    print(f'RMSE - SMA Long: {rmse_sma_long}')
    print(f'RMSE - EMA Short: {rmse_ema_short}')
    print(f'RMSE - EMA Long: {rmse_ema_long}')