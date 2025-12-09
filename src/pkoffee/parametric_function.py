"""Parametric functions.

This module provides functions with signature f(x, *args, **kwargs), where `x` is the function's input and the other
arguments are the function parameters.
"""

from collections.abc import Callable
from typing import Concatenate

import numpy as np

from pkoffee.data import data_dtype

# Define the function type in this module: users of parametric function should use this type to annotate the function
# to make sure they have the correct signature.
ParamFunctionType = Callable[Concatenate[np.ndarray, ...], np.ndarray]


def quadratic(x: np.ndarray, a0: data_dtype, a1: data_dtype, a2: data_dtype) -> np.ndarray:
    """Quadratic (polynomial) model: f(x) = a₀ + a₁x + a₂x².

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a0 : pkoffe_data_dtype
        Constant term.
    a1 : pkoffe_data_dtype
        Linear coefficient.
    a2 : pkoffe_data_dtype
        Quadratic coefficient.

    Returns
    -------
    np.ndarray
        Model predictions at each point in `x`.

    References
    ----------
    1. Wikipedia contributors. (2025, September 16). Quadratic function. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:28, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Quadratic_function&oldid=1311755644
    """
    x = np.asarray(x)
    return a0 + a1 * x + a2 * x**2


def michaelis_menten_saturation(x: np.ndarray, v_max: data_dtype, k: data_dtype, y0: data_dtype) -> np.ndarray:
    """Michaelis-Menten (saturating) model: f(x) = y₀ + Vₘₐₓ·x/(K + x).

    This model describes saturation behavior common in enzyme kinetics
    and can represent diminishing returns.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    v_max : pkoffe_data_dtype
        Maximum rate/value.
    k : pkoffe_data_dtype
        Half-saturation constant (Michaelis constant).
    y0 : pkoffe_data_dtype
        Baseline offset.

    Returns
    -------
    np.ndarray
        Model predictions.

    References
    ----------
    1. Wikipedia contributors. (2025, December 1). Michaelis-Menten kinetics. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:32, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Michaelis%E2%80%93Menten_kinetics&oldid=1325118298
    """
    x = np.asarray(x)
    return y0 + v_max * (x / np.maximum(k + x, 1e-9))


def logistic(x: np.ndarray, L: data_dtype, k: data_dtype, x0: data_dtype, y0: data_dtype) -> np.ndarray:  # noqa: N803 L is fine as argument as it follows model definition
    """Logistic (sigmoid) model: f(x) = y₀ + L/(1 + e^(-k(x - x₀))).

    Models S-shaped growth with lower and upper asymptotes.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    L : pkoffe_data_dtype
        Maximum value of the curve (carrying capacity).
    k : pkoffe_data_dtype
        Steepness of the curve.
    x0 : pkoffe_data_dtype
        Midpoint (inflection point) of the sigmoid.
    y0 : pkoffe_data_dtype
        Minimum value (lower asymptote).

    Returns
    -------
    np.ndarray
        Model predictions.

    References
    ----------
    1. Wikipedia contributors. (2025, November 29). Logistic regression. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:34, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Logistic_regression&oldid=1324697470
    """
    x = np.asarray(x)
    return y0 + L / (1.0 + np.exp(-k * (x - x0)))


def peak_model(x: np.ndarray, a: data_dtype, b: data_dtype) -> np.ndarray:
    """Peak model (gamma-like): f(x) = a·x·e^(-x/b).

    Models a single peak with exponential decay, useful for
    representing optimal consumption with negative effects beyond peak.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : pkoffe_data_dtype
        Amplitude scaling factor.
    b : pkoffe_data_dtype
        Decay rate parameter.

    Returns
    -------
    np.ndarray
        Model predictions.

    References
    ----------
    1. Wikipedia contributors. (2025, November 4). Gamma distribution. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:38, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1320436343
    """
    x = np.asarray(x)
    return a * x * np.exp(-x / np.maximum(b, 1e-9))


def peak2_model(x: np.ndarray, a: data_dtype, b: data_dtype) -> np.ndarray:
    """Quadratic peak model: f(x) = a·x²·e^(-x/b).

    Similar to peak_model but with quadratic growth before decay.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : pkoffe_data_dtype
        Amplitude scaling factor.
    b : pkoffe_data_dtype
        Decay rate parameter.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return a * (x**2) * np.exp(-x / np.maximum(b, 1e-9))
