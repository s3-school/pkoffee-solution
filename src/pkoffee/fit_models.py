"""Mathematical models for coffee productivity relationships.

This module provides various parametric models that can be fitted to
coffee consumption vs productivity data.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.optimize import curve_fit

from pkoffee.data import data_dtype
from pkoffee.metrics import compute_r2
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    ParametersBounds,
    ParametricFunction,
    Peak2Model,
    PeakModel,
    Quadratic,
)


@dataclass
class Model:
    """Model defined by a prediction function, parameters and parameter's bounds.

    Attributes
    ----------
    name : str
        Name of the model
    function : ParametricFunction
        The model prediction function
    params : tuple[data_types]
        Model parameters passed to the predict function
    bounds : ParametersBounds
        Boundary values for the model parameters.
    """

    name: str
    function: ParametricFunction
    params: dict[str, data_dtype]
    bounds: ParametersBounds
    r_squared: data_dtype = -data_dtype(np.inf)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the model on input `x`.

        Parameters
        ----------
        x : np.ndarray
            The model input as 1D array.

        Returns
        -------
        np.ndarray
            Prediction of the model, same shape as `x`.
        """
        return self.function(x, **self.params)

    def __repr__(self) -> str:
        """Return a formatted string representation of the model result."""
        return f"ModelFit(name='{self.name}', R²={self.r_squared:.3f})"

    @classmethod
    def sort(cls, models: list[Self]) -> None:
        """Sort `models` by R² (descending), in-place.

        Parameters
        ----------
        models : list[Self]
            List of models to sort by R²
        """
        models.sort(
            key=lambda r: r.r_squared if np.isfinite(r.r_squared) else -np.inf,
            reverse=True,
        )


def default_models(x: np.ndarray, y: np.ndarray) -> list[Model]:
    """Generate model configurations with suited initial parameter guesses.

    Parameters
    ----------
    x : np.ndarray
        Input data (cups).
    y : np.ndarray
        Output data (productivity).

    Returns
    -------
    list[Model]
        List of model configurations ready for fitting.
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    return [
        Model(
            name="Quadratic",
            function=Quadratic(),
            params=Quadratic.param_guess(y_min=y_min),
            bounds=Quadratic.param_bounds(),
        ),
        Model(
            name="Michaelis-Menten",
            function=MichaelisMentenSaturation(),
            params=MichaelisMentenSaturation.param_guess(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
            bounds=MichaelisMentenSaturation.param_bounds(),
        ),
        Model(
            name="Logistic",
            function=Logistic(),
            params=Logistic.param_guess(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
            bounds=Logistic.param_bounds(),
        ),
        Model(
            name="Peak",
            function=PeakModel(),
            params=PeakModel.param_guess(x_min=x_min, x_max=x_max, y_max=y_max),
            bounds=PeakModel.param_bounds(),
        ),
        Model(
            name="Peak²",
            function=Peak2Model(),
            params=Peak2Model.param_guess(x_min=x_min, x_max=x_max, y_max=y_max),
            bounds=Peak2Model.param_bounds(),
        ),
    ]


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    max_iterations: int = 20000,
) -> tuple[Model, np.ndarray]:
    """Fit a single model to the data.

    Parameters
    ----------
    x : np.ndarray
        Input data (independent variable).
    y : np.ndarray
        Output data (dependent variable).
    model : Model
        Model including function and parameters.
    max_iterations : int, optional
        Maximum number of optimization iterations, by default 20000.

    Returns
    -------
    tuple[FittedModel, np.ndarray] | None
        tuple with Fitted model and predictions on training data, or None if fitting failed.

    Raises
    ------
    ValueError
        If either x or y contain NaNs.
    RuntimeError
        If the least-squares minimization fails.
    """
    params = model.params.keys()
    optimal_params, _ = curve_fit(
        model.function,
        x,
        y,
        p0=list(model.params.values()),  # same order as params
        bounds=(
            tuple(model.bounds.min[p] for p in params),
            tuple(model.bounds.max[p] for p in params),
        ),  # converting to tuples while respecting params order
        maxfev=max_iterations,
    )
    predictions = model.function(x, *optimal_params)
    r_squared = compute_r2(y, predictions)
    return (
        Model(
            model.name,
            function=model.function,
            params=dict(
                zip(model.params.keys(), optimal_params, strict=True)
            ),  # build back dictionary of parameters from values, the order is the same as the params of model
            bounds=model.bounds,
            r_squared=r_squared,
        ),
        predictions,
    )
