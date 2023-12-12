import seaborn as sns
import matplotlib.pyplot as plt
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


def adj_close_volume(data, ticker, columns:list, y_log=False, ax=None):
    """Plot the adjusted close price and volume of a stock.

    Args:
        data (pd.DataFrame): The data to plot.
        ticker (str): The ticker of the stock.
        columns (list): The columns to plot.
        y_log (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
    """
    plt.rcParams["figure.figsize"] = (5, 2)
    data = data[ticker]

    # Plot the adjusted close price and volume
    if ax is None:
        fig, ax = plt.subplots(len(columns), 1, figsize=(15, 5 * len(columns)))

    for i, column in enumerate(columns):
        ax[i].set_title(f'{ticker} {column}')
        ax[i].set_ylabel('Price $' if column.lower() in {'adj close', 'price'} else 'units')
        if y_log:
            ax[i].set_yscale('log')
        ax[i].plot(data[column], color='tab:blue' if column.lower() in {'adj close', 'price'} else 'tab:orange', label=column, linewidth=1.5 if column.lower() in {'adj close', 'price'} else .9, linestyle='-', alpha=0.8)
        ax[i].grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.xlabel('Date')
    plt.show()


def autocorrelation(series, title, lags=40):
    """
    Plots autocorrelation for a time series.

    Args:
    series (pd.Series): The time series data.
    title (str): The title for the plot.
    lags (int, optional): The number of lags to include. Default is 40.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
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