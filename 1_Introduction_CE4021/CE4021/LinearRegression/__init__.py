import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For matrix operations
from typing import List, Union, TypeAlias, Optional

# Define a custom type alias for a matrix
Matrix: TypeAlias = List[List[Union[int, float]]]
Vector: TypeAlias = List[Union[int, float]]


def linreg_weights(X, y):
    # Calculation of weights using pseudo-inverse. Note that X needs to contain the bias of 1
    return np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)


def linreg_predict(w, X):
    # Calculation of outputs given weights and data (X). Note that X needs to contain the bias of 1.
    out = [w.T.dot(x) for x in X]
    return np.array(out)
