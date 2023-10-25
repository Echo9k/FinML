import numpy as np
from fractions import Fraction
from typing import Iterable, Union, List

Number = Union[int, float, Fraction]


def __drop_leftmost_zeros__(lst: List[Number]) -> List[Number]:
    """
    Removes the leftmost zeros from a list.

    Args:
        lst (List[Number]): A list of numbers.

    Returns:
        List[Number]: A new list with the leftmost zeros removed.
    """
    return next((lst[i:] for i, num in enumerate(lst) if num != 0), [])


def derive_poly(poly: Iterable[Number], increasing: bool = False):
    """
    Symbolically calculates the derivative of a polynomial.

    Args:
        poly (list): A polynomial represented as a list of coefficients.
                  For example, the polynomial x^3 + 2x^2 - 5x + 1 would be represented as [1, 0, 2, -5, 1].
        increasing (bool, optional): If True, the polynomials is provided in increasing order of powers. i.e.  [0, 1, 2] for 2x^2 + 1x + 0.
                                     If False (default), polynomials are in decreasing order of powers.  i.e.  [2, 1, 0] for 2x^2 + 1x + 0.

    Returns:
        list: The derivative of the polynomial p, also represented as a list of coefficients.
    """
    if not increasing:
        poly = reversed(poly)
    result = [i * e for e, i in enumerate(poly)]
    if increasing:
        return __drop_leftmost_zeros__(result)
    result = reversed(result)
    return list(result)[:-1]


def evaluate_polynomial(x: int, *coefficients: Number) -> float:
    """
    Evaluate a polynomial for a given value of its variable.

    Args:
        x (int): The value at which to evaluate the polynomial.
        *coefficients (Number): Coefficients of the polynomial in descending order of power.

    Returns:
        float: The result of evaluating the polynomial at the given value x.
    """
    return sum(
        coefficient * (x ** (len(coefficients) - 1 - i))
        for i, coefficient in enumerate(coefficients)
    )


def integral(poly: dict, a: Number, b: Number):
    """
    Calculate the integral of a polynomial from point A to B.

    Parameters:
    - Poly (dict): A dictionary representing the polynomial. The keys are the powers, and the values are the coefficients.
    - a (float): The lower bound of the integral.
    - b (float): The upper bound of the integral.

    Returns:
    - float: The result of the integral from a to b.
    """
    return sum(
        coefficient * ((b ** (power + 1) - a ** (power + 1)) / (power + 1))
        for power, coefficient in poly.items()
    )


def evaluate_poly(poly, x):
    """
    Evaluate a polynomial at a specific point.

    Args:
        poly (list of tuple): List of tuples representing polynomial coefficients and their powers.
        x (float): The point at which to evaluate the polynomial.

    Returns:
        float: The result of evaluating the polynomial at point x.
    """
    return sum(coef * x ** power for coef, power in poly)


def riemann_integral_approximation(poly, step, start, stop):
    """
    Numerically integrate a polynomial using a basic rectangular method.

    Args:
        poly (list of tuple): List of tuples representing polynomial coefficients and their powers.
        step (float): Step size for numerical integration.
        start (float): Starting point of integration.
        stop (float): Ending point of integration.

    Returns:
        float: The result of the numerical integration.
    """
    return sum(evaluate_poly(poly, x) * step for x in np.arange(start, stop, step))
