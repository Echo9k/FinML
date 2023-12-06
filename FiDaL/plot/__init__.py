import matplotlib.pyplot as plt
import seaborn as sns


# Function to plot time series data
def plot_time_series(df, column, title='Stock Prices Over Time', ylabel='Price'):
    """Plots time series data for a single column."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Function to plot histogram of returns
def plot_histogram(df, column, title='Distribution of Returns', bins=50):
    """Plots a histogram of the returns."""
    plt.figure(figsize=(12, 6))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()


# Function to plot correlation heatmap
def plot_correlation_heatmap(df, title='Correlation Heatmap'):
    """Plots a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.show()


# Function to plot a scatter diagram
def plot_scatter(df, column1, column2, title='Scatter Diagram'):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[column1], df[column2], alpha=0.5)
    plt.title(title)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()
