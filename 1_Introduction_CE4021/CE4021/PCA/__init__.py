import numpy as np
import random as rand
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple, Optional


def generate_data(num_points: int = 20, a_x: float = 0.05, a_y: float = 10) -> np.ndarray:
    """
    Generate synthetic 2D data.

    Args:
        num_points (int): Number of data points.
        a_x (float): Factor for x-axis noise.
        a_y (float): Factor for y-axis noise.

    Returns:
        np.ndarray: Generated data.
    """
    return np.array([[n*(1+a_x*(rand.random()-0.5)), 4*n+a_y*(rand.random()-0.5)] for n in range(num_points)])


def perform_pca(data: np.ndarray, num_components: int = 1) -> Tuple[Optional[np.ndarray], Optional[PCA]]:
    """
    Perform PCA and return the transformed data and PCA model.

    Args:
        data (np.ndarray): Original data.
        num_components (int): Number of principal components.

    Returns:
        tuple: Transformed data and PCA model.
    """
    try:
        pca = PCA(n_components=num_components)
        transformed_data = pca.fit_transform(data)
        return transformed_data, pca
    except Exception as e:
        print(f"An error occurred during PCA: {e}")
        return None, None


def plot_data(x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str, legend_label: Optional[str] = None):
    """
    Plot data with matplotlib.

    Args:
        x (np.ndarray): X-axis data.
        y (np.ndarray): Y-axis data.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        legend_label (str, optional): Legend label.
    """
    plt.scatter(x, y, label=legend_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend_label:
        plt.legend()
    plt.show()


def reverse_transform(transformed_data: np.ndarray, pca_model: PCA) -> Optional[np.ndarray]:
    """
    Perform reverse transformation using only NumPy.

    Args:
        transformed_data (np.ndarray): Transformed data.
        pca_model (PCA): Scikit-learn PCA model.

    Returns:
        np.ndarray: Reconstructed data.
    """
    try:
        return np.dot(transformed_data, pca_model.components_) + pca_model.mean_
    except Exception as e:
        print(f"An error occurred during reverse transformation: {e}")
        return None


def plot_pair_one(original: np.ndarray, transformed: np.ndarray):
    """
    Create subplots for Original and PCA Transformed data.

    Args:
        original (np.ndarray): Original data.
        transformed (np.ndarray): Transformed (PCA) data.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Original Data
    axs[0].scatter(original[:, 0], original[:, 1])
    axs[0].set_title('Original Data')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')

    # Transformed Data
    axs[1].scatter(transformed, [0]*len(transformed), c='black', marker='1')
    axs[1].set_title('PCA Transformed Data')
    axs[1].set_xlabel('Principal Component 1')

    plt.tight_layout()
    plt.show()

def plot_pair_two(reconstructed: np.ndarray, original: np.ndarray):
    """
    Create subplots for Reconstructed data and Original vs. Reconstructed data.

    Args:
        reconstructed (np.ndarray): Reconstructed data after inverse PCA.
        original (np.ndarray): Original data.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    _rename_axes(axs, 0, reconstructed, 'Reconstructed Data')
    # Original vs Reconstructed Data
    axs[1].scatter(original[:, 0], original[:, 1], label='Original Data')
    _rename_axes(
        axs, 1, reconstructed, 'Original vs Reconstructed Data'
    )
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def _rename_axes(axs, arg1, reconstructed, arg3):
    # Reconstructed Data
    axs[arg1].scatter(
        reconstructed[:, 0],
        reconstructed[:, 1],
        label='Reconstructed Data',
        marker='x',
        c='red',
    )
    axs[arg1].set_title(arg3)
    axs[arg1].set_xlabel('Feature 1')
    axs[arg1].set_ylabel('Feature 2')
