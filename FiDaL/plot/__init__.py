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
        moving_average = product.rolling(window=window).mean()
        ax.plot(moving_average, label=f'Moving Average ({window} days)', linewidth=1.2)

    if include_stats:
        stats(product, ax)
    
    # Set y-axis scale
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Date')
    plt.legend()
    plt.show()


def adj_close_volume(data, ticker, y_log=False):
    # Get the 'Adj Close' and 'Volume' variables for the given ticker
    adj_close = data[('Adj Close', ticker)]
    volume = data[('Volume', ticker)]

    # Plot the adjusted close price and volume
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].set_title(f'{ticker} Adj Close')
    ax[0].set_ylabel('Price $')
    if y_log:
        ax[0].set_yscale('log')
    ax[0].plot(adj_close, color='tab:blue', label='Adj Close', linewidth=1.5, linestyle='-', alpha=0.8)
    ax[0].grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=0.5)

    ax[1].set_title(f'{ticker} Volume')
    ax[1].set_ylabel('units')
    if y_log:
        ax[1].set_yscale('log')
    ax[1].plot(volume, color='tab:orange', label='Volume', linewidth=.9, linestyle='-', alpha=0.8)
    ax[1].grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=0.5)

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
