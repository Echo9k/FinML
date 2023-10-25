import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(message)s'
)


# Create a function to log and print the results of an experiment
def log_experiment_results(experiment):
    """
    Logs the results of an experiment.
    :param experiment:
    :return:
    """
    if experiment.__name__:
        logging.info(f"Experiment:{experiment.__name__}")
    logging.info(f"Experiment Results:\n"
                 f"\t• Number of Trials = {experiment.num_trials}\n"
                 # f"\t• Raw Data - {experiment.raw_data}\n"
                 f"\t• Statistics : {experiment.statistics}")


class Experiment:
    """
    A class representing an experiment.

    Args:
        num_trials (int): The number of trials to run in the experiment.

    Raises:
        ValueError: If num_trials is not a positive integer.

    Attributes:
        num_trials (int): The number of trials in the experiment.
        raw_data (dict): A dictionary to store the raw data of the experiment.
        statistics (dict): A dictionary to store the statistics of the experiment.

    Methods:
        simulate(): Placeholder method to simulate the experiment.
        analyze(): Placeholder method to analyze the results of the experiment.
        plot(title, x_label, y_label): Placeholder method to plot the results of the experiment.

    Note: This class serves as a base class and should be subclassed to define the specific behavior of an experiment.

    Examples:
        This class is typically subclassed to define the behavior of a specific experiment.
    """

    def __init__(self, num_trials: int):
        if num_trials <= 0:
            raise ValueError("num_trials must be a positive integer")
        self.__name__ = None
        self.num_trials = num_trials
        self.raw_data = {}
        self.statistics = {}

    def simulate(self):
        print('This is a placeholder method for simulation')  # Placeholder method

    def analyze(self):
        print('This is a placeholder method for analyzing the results.')

    def plot(self, title: str, x_label: str, y_label: str):
        print('This is a placeholder method for plotting')


class Analysis:
    """
    Calculates statistics for a given list of data.

    Args:
        data (list): A list of numerical data.

    Returns:
        dict: A dictionary containing the mean and standard deviation of the data.

    Examples:
        >>> Analysis([1, 2, 3, 4, 5])
        {'mean': 3.0, 'std_dev': 1.4142135623730951}
        >>> Analysis([10, 20, 30, 40, 50])
        {'mean': 30.0, 'std_dev': 15.811388300841896}
    """

    @staticmethod
    def calculate_statistics(data: list) -> dict:
        return {
            'mean': np.mean(data),
            'std_dev': np.std(data)
        }


class DiceExperiment(Experiment):
    """
    A class representing a dice experiment that consists of rolling two dice and recording the sum of the rolls.

    Args:
        num_trials (int): The number of trials to run in the experiment.

    Methods:
        - simulate(): Simulates the dice experiment and stores the results.
        - analyze(): Calculates the statistics of the dice experiment and logs the results.
        - plot(title, x_label, y_label): Plots a histogram of the dice experiment.

    Note: This class inherits from the Experiment class.

    Examples:
        >>> experiment = DiceExperiment(100)
        >>> experiment.simulate()  # This function simulates the experiment.
        >>> experiment.analyze()  # This function calculates the statistics.
        >>> experiment.plot()  # This function plots the histogram.
    """

    def __init__(self, num_trials: int):
        super().__init__(num_trials)
        self.expectation = None
        self.__name__ = "Dice Experiment"

    def simulate(self):
        """
        Analyzes the results of a dice experiment.

        Args:
            self: The instance of the class.

        Returns:
            None

        Note: This method combines the successful and unsuccessful rolls, calculates the statistics using the Analysis class, and logs the mean and standard deviation of the dice experiment.

        Examples:
            This method is typically called within the context of an instance of the class.
        """
        # Vectorized simulation
        first_die_rolls = np.random.randint(1, 7, self.num_trials)
        second_die_rolls = np.random.randint(1, 7, self.num_trials)
        sum_of_rolls = first_die_rolls + second_die_rolls
        successful_rolls = sum_of_rolls[(sum_of_rolls < 3) | (sum_of_rolls > 10)]
        unsuccessful_rolls = sum_of_rolls[(sum_of_rolls >= 3) & (sum_of_rolls <= 10)]
        self.raw_data = {
            'successful_rolls': successful_rolls.tolist(),
            'unsuccessful_rolls': unsuccessful_rolls.tolist()
        }

    def analyze(self):
        """
        Calculates the statistics of the dice experiment and logs the results.

        Args:
            None

        Returns:
            None
        """
        # Concatenate the lists of successful and unsuccessful rolls
        combined_data = self.raw_data['successful_rolls'] + self.raw_data['unsuccessful_rolls']
        self.statistics = Analysis.calculate_statistics(combined_data)
        # Use the log_experiment_results function to log the results
        log_experiment_results(self)

    def plot(self, title="Dice Roll Distribution", x_label="Sum of Rolls", y_label="Frequency"):
        """
        Plots a histogram of the dice experiment. The histogram contains two bars:
            one for successful rolls and one for unsuccessful rolls.

        Args:
            title (str): The title of the plot.
            x_label (str): The label of the x-axis.
            y_label (str): The label of the y-axis.

        Returns:
            None

        Examples:
            >>> experiment = DiceExperiment(100)
            >>> experiment.simulate()  # This function simulates the experiment.
            >>> experiment.analyze()  # This function calculates the statistics.
            >>> experiment.plot()  # This function plot the histogram.
        """
        successful_rolls = self.raw_data['successful_rolls']
        unsuccessful_rolls = self.raw_data['unsuccessful_rolls']
        plt.hist([successful_rolls, unsuccessful_rolls],
                 bins=range(1, 14), edgecolor='black',
                 label=['Successful', 'Unsuccessful'], color=['blue', 'red'])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()


class HeightExperiment(Experiment):
    """
    A class representing a height experiment that consists of sampling heights from a normal distribution.

    Args:
        num_customers (int): The number of customers to sample heights from.
        average_height (float): The average height in centimeters (default: 160).
        height_std_dev (float): The standard deviation of heights in centimeters (default: 20).
        **kwargs: Additional keyword arguments.

    Methods:
        - simulate(): Simulates the height experiment and calculates the expectation and dispersion.
        - analyze_and_plot(): Analyzes and plots the results of the height experiment.

    Note: This class inherits from the Experiment class.

    Examples:
        >>> experiment = HeightExperiment(100)
        >>> experiment.simulate()  # This function simulates the experiment.
        >>> experiment.analyze()  # This function calculates the statistics.
        >>> experiment.plot()  # This function plot the histogram.
    """

    def __init__(self, num_customers, average_height=160, height_std_dev=20, **kwargs):
        """
        Initializes the height experiment.

        Args:
            num_customers (int): The number of customers to sample heights from.
            average_height (float): The average height in centimeters (default: 160).
            height_std_dev (float): The standard deviation of heights in centimeters (default: 20).
        """
        super().__init__(num_customers)
        self.__name__ = "Height Experiment"
        self.dispersion = None
        self.average_height = average_height
        self.height_std_dev = height_std_dev

    def simulate(self):
        """
        Simulates a coin experiment.

        Args:
            self: The instance of the class.

        Returns:
            None

        Note: This method is part of a class and updates the raw data and expectation attributes of the instance.

        Examples:
            >>> experiment = HeightExperiment(100)
            >>> experiment.simulate()  # This function simulates the experiment.
        """

        sampled_heights = np.random.normal(self.average_height, self.height_std_dev, self.num_trials)
        self.raw_data = sampled_heights
        self.expectation = np.mean(sampled_heights)
        self.dispersion = np.std(sampled_heights)

    def analyze(self):
        """
        Analyzes the results of the experiment and prints the results.

        Args:
            None

        Returns:
            None

        Examples:
            >>> experiment = HeightExperiment(100)
            >>> experiment.simulate()  # This function simulates the experiment.
            >>> experiment.analyze()  # This function calculates the statistics.
        """
        self.statistics = Analysis.calculate_statistics(self.raw_data)

        log_experiment_results(self)  # use the log_experiment_results function to log the results

    def plot(self, title="Height Distribution", x_label="Height (cm)", y_label="Frequency"):
        """
        Plots the height distribution based on the given result dictionary.

        Args:
            title (str): The title of the plot.
            x_label (str): The label of the x-axis.
            y_label (str): The label of the y-axis.

        Returns:
            None

        Examples:
            >>> experiment = HeightExperiment(100)
            >>> experiment.simulate()  # This function simulates the experiment.
            >>> experiment.analyze()  # This function calculates the statistics.
            >>> experiment.plot()  # This function plot the histogram.
            (displays a histogram plot)
        """
        plt.hist(self.raw_data, bins=50, edgecolor='black')
        plt.axvline(x=self.expectation, color='red', linestyle='--', label=f'Mean: {self.expectation:.2f}')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()  # Show legend to display the mean value
        plt.show()


class CoinExperiment(Experiment):
    """
    A class representing a coin experiment that consists of flipping a coin multiple times.
    """

    def __init__(self, num_trials=1_000, num_simulations=1, head_bias=0.5, **kwargs):
        super().__init__(num_trials)
        self.__name__ = "Coin Experiment"
        self.num_trials = num_trials
        self.num_simulations = num_simulations
        self.head_bias = head_bias
        self.raw_data = None
        self.statistics = None

    def simulate(self):
        """
        Simulate multiple simulations in multiple trials in a vectorized manner.
        The results are stored in the raw_data attribute.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # Simulate multiple simulations in multiple trials in a vectorized manner
        self.raw_data = np.random.choice(
            [0, 1],
            (self.num_simulations, self.num_trials),
            p=[1 - self.head_bias, self.head_bias]
        )

    def analyze(self):
        """
        Analyzes the results of the experiment and prints the results.
        It calculates the statistics of the experiment and logs the results such as:
            - The mean proportion of heads across simulations
            - The standard deviation of the proportion of heads across simulations
            - The theoretical expectation of the experiment
            - The empirical expectation of the experiment
            - The difference between the theoretical and empirical expectations

        Note: This method assumes that the simulate method has been run.

        Args:
            None

        Returns:
            None
        """
        if self.raw_data is None:
            print("No data to analyze. Please run the simulation method first.")
            return

        # Aggregate statistics across simulations
        heads_proportions = np.mean(self.raw_data, axis=1)
        theoretical_expectation = self.head_bias
        empirical_expectation = np.mean(heads_proportions)
        diff_expectation = np.abs(theoretical_expectation - empirical_expectation)

        # Use Analysis module to calculate mean and standard deviation
        analysis_results = Analysis.calculate_statistics(heads_proportions)
        mean_proportion = analysis_results['mean']
        std_dev_proportion = analysis_results['std_dev']

        self.statistics = {
            'mean_proportion': mean_proportion,
            'std_dev_proportion': std_dev_proportion,
            'theoretical_expectation': theoretical_expectation,
            'empirical_expectation': empirical_expectation,
            'diff_expectation': diff_expectation
        }

        log_experiment_results(self)  # Assuming log_experiment_results is a pre-defined function

    def plot(self, title="Coin Flip Distribution", x_label="Outcome", y_label="Probability"):
        """
        Plots the coin flip distribution based on the given result dictionary.

        If the number of trials is 1, then the plot contains two bars: one for heads and one for tails.
        Otherwise, the plot contains a histogram of the distribution of heads across simulations.

        Note: This method assumes that the simulate method has been run.

        Args:
            title (str): The title of the plot.
            x_label (str): The label of the x-axis.
            y_label (str): The label of the y-axis.

        Returns:
            None
        """

        if self.raw_data is None:
            print("No data to plot. Please run the simulate method first.")
            return

        # For the case of 1 trial 1 simulation, plot a bar plot
        if self.num_trials == 1:
            plt.bar(
                ['Heads', 'Tails'],
                [np.mean(self.raw_data == 1), np.mean(self.raw_data == 0)]
            )
            plt.title(f"Single experiment - {title}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            self._hist_maker(title, x_label, y_label)
        plt.show()

    def _hist_maker(self, title, x_label, y_label):
        """Helper function to plot the histogram of the distribution of heads across simulations."""
        # Otherwise, plot the distribution
        plt.hist(
            np.mean(self.raw_data, axis=1),
            bins='auto',
            edgecolor='black'
        )
        aggregated_mean = np.mean(self.raw_data)
        plt.axvline(
            x=aggregated_mean,
            color='red',
            linestyle='--',
            label=f'Aggregated Mean: {aggregated_mean:.2%}'
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()  # Show legend to display the aggregated mean value
