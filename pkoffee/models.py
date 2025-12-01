"""
Mathematical models for coffee productivity relationships.

This module provides various parametric models that can be fitted to
coffee consumption vs productivity data.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pkoffee.metrics import compute_r2


@dataclass
class ModelResult:
    """
    Container for fitted model results.

    Attributes
    ----------
    name : str
        Human-readable name of the model.
    function : Callable
        The fitted model function.
    parameters : np.ndarray
        Optimal parameter values from fitting.
    r_squared : float
        R² score indicating goodness of fit.
    predictions : np.ndarray
        Model predictions on the training data.
    """

    name: str
    function: Callable[[np.ndarray, ...], np.ndarray]
    parameters: np.ndarray
    r_squared: float
    predictions: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new input values.

        Parameters
        ----------
        x : np.ndarray
            Input values (e.g., number of cups).

        Returns
        -------
        np.ndarray
            Predicted productivity values.
        """
        return self.function(x, *self.parameters)

    def __repr__(self) -> str:
        """Return a formatted string representation of the model result."""
        return f"ModelResult(name='{self.name}', R²={self.r_squared:.4f})"


def quadratic_model(x: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    """
    Quadratic (polynomial) model: f(x) = a₀ + a₁x + a₂x².

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a0 : float
        Constant term.
    a1 : float
        Linear coefficient.
    a2 : float
        Quadratic coefficient.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return a0 + a1 * x + a2 * x**2


def michaelis_menten_model(
    x: np.ndarray, v_max: float, k: float, y0: float
) -> np.ndarray:
    """
    Michaelis-Menten (saturating) model: f(x) = y₀ + Vₘₐₓ·x/(K + x).

    This model describes saturation behavior common in enzyme kinetics
    and can represent diminishing returns.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    v_max : float
        Maximum rate/value.
    k : float
        Half-saturation constant (Michaelis constant).
    y0 : float
        Baseline offset.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return y0 + v_max * (x / np.maximum(k + x, 1e-9))


def logistic_model(
    x: np.ndarray, L: float, k: float, x0: float, y0: float
) -> np.ndarray:
    """
    Logistic (sigmoid) model: f(x) = y₀ + L/(1 + e^(-k(x - x₀))).

    Models S-shaped growth with lower and upper asymptotes.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    L : float
        Maximum value of the curve (carrying capacity).
    k : float
        Steepness of the curve.
    x0 : float
        Midpoint (inflection point) of the sigmoid.
    y0 : float
        Minimum value (lower asymptote).

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return y0 + L / (1.0 + np.exp(-k * (x - x0)))


def peak_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Peak model (gamma-like): f(x) = a·x·e^(-x/b).

    Models a single peak with exponential decay, useful for
    representing optimal consumption with negative effects beyond peak.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : float
        Amplitude scaling factor.
    b : float
        Decay rate parameter.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return a * x * np.exp(-x / np.maximum(b, 1e-9))


def peak2_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Quadratic peak model: f(x) = a·x²·e^(-x/b).

    Similar to peak_model but with quadratic growth before decay.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    a : float
        Amplitude scaling factor.
    b : float
        Decay rate parameter.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    x = np.asarray(x)
    return a * (x**2) * np.exp(-x / np.maximum(b, 1e-9))


@dataclass
class ModelConfig:
    """Configuration for a model including initial parameters and bounds."""

    name: str
    function: Callable
    initial_params: list[float]
    bounds: tuple


def _get_model_configs(
    x: np.ndarray, y: np.ndarray
) -> list[ModelConfig]:
    """
    Generate model configurations with smart initial parameter guesses.

    Parameters
    ----------
    x : np.ndarray
        Input data (cups).
    y : np.ndarray
        Output data (productivity).

    Returns
    -------
    list[ModelConfig]
        List of model configurations ready for fitting.
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    dy = max(1e-8, y_max - y_min)
    x_mid = 0.5 * (x_min + x_max)

    return [
        ModelConfig(
            name="Quadratic",
            function=quadratic_model,
            initial_params=[y_min, 0.0, 0.01],
            bounds=(-np.inf, np.inf),
        ),
        ModelConfig(
            name="Michaelis-Menten",
            function=michaelis_menten_model,
            initial_params=[dy, max(1.0, 0.2 * (x_min + x_max)), y_min],
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        ),
        ModelConfig(
            name="Logistic",
            function=logistic_model,
            initial_params=[dy, 0.5, x_mid, y_min],
            bounds=(
                [-np.inf, 0.0, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf],
            ),
        ),
        ModelConfig(
            name="Peak",
            function=peak_model,
            initial_params=[max(y_min, y_max), max(1.0, x_mid)],
            bounds=([-np.inf, 0.0], [np.inf, np.inf]),
        ),
        ModelConfig(
            name="Peak²",
            function=peak2_model,
            initial_params=[max(1e-6, y_max / max(1.0, x_max**2)), max(1.0, x_mid)],
            bounds=([-np.inf, 0.0], [np.inf, np.inf]),
        ),
    ]


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    config: ModelConfig,
    max_iterations: int = 20000,
) -> Optional[ModelResult]:
    """
    Fit a single model to the data.

    Parameters
    ----------
    x : np.ndarray
        Input data (independent variable).
    y : np.ndarray
        Output data (dependent variable).
    config : ModelConfig
        Model configuration including function and parameters.
    max_iterations : int, optional
        Maximum number of optimization iterations, by default 20000.

    Returns
    -------
    Optional[ModelResult]
        Fitted model result, or None if fitting failed.
    """
    try:
        optimal_params, _ = curve_fit(
            config.function,
            x,
            y,
            p0=config.initial_params,
            bounds=config.bounds,
            maxfev=max_iterations,
        )
        
        predictions = config.function(x, *optimal_params)
        r_squared = compute_r2(y, predictions)
        
        return ModelResult(
            name=config.name,
            function=config.function,
            parameters=optimal_params,
            r_squared=r_squared,
            predictions=predictions,
        )
    except Exception as e:
        print(f"Warning: Failed to fit {config.name} model: {e}")
        return None


def fit_all_models(
    data: "pd.DataFrame",
    max_iterations: int = 20000,
) -> list[ModelResult]:
    """
    Fit all available models to the data and rank by R².

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'cups' and 'productivity' columns.
    max_iterations : int, optional
        Maximum iterations for optimization, by default 20000.

    Returns
    -------
    list[ModelResult]
        List of successfully fitted models, sorted by R² (descending).

    Examples
    --------
    >>> from pkoffee.data import load_data
    >>> data = load_data("coffee_productivity.csv")
    >>> models = fit_all_models(data)
    >>> for model in models:
    ...     print(f"{model.name}: R² = {model.r_squared:.4f}")
    Logistic: R² = 0.9523
    Peak: R² = 0.9401
    """
    from pkoffee.data import extract_arrays
    
    x, y = extract_arrays(data)
    configs = _get_model_configs(x, y)
    
    results = []
    for config in configs:
        result = fit_model(x, y, config, max_iterations)
        if result is not None:
            results.append(result)
    
    # Sort by R² descending
    results.sort(
        key=lambda r: r.r_squared if np.isfinite(r.r_squared) else -np.inf,
        reverse=True,
    )
    
    return results
