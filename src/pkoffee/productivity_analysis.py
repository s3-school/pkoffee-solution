"""Coffee Productivity analysis module."""

import argparse
import logging

import numpy as np
import pandas as pd

from pkoffee.data import RequiredColumn, data_dtype, extract_arrays, load_csv
from pkoffee.fit_model import Model, fit_model
from pkoffee.parametric_function import Logistic, MichaelisMentenSaturation, Peak2Model, PeakModel, Quadratic
from pkoffee.visualization import FigureParameters, Show, create_comparison_plot, format_model_rankings, plot_models


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


def fit_all_models(
    data: pd.DataFrame,
    max_iterations: int = 20000,
) -> list[Model]:
    r"""Fit all available models to the data and rank by R².

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'cups' and 'productivity' columns.
    max_iterations : int, optional
        Maximum iterations for optimization, by default 20000.

    Returns
    -------
    list[ModelResult]
        List of fitted models, sorted by R² (descending). Models for which fitting failed
        are still in the list with default values (R²=-inf).

    Examples
    --------
    >>> data = load_csv(Path(tmpfile.name))  # doctest: +SKIP
    >>> models = fit_all_models(data)  # doctest: +SKIP
    >>> for model in models:  # doctest: +SKIP
    >>>     print(f"{model.name}: R² = {model.r_squared:.4f}")# doctest: +SKIP
    Quadratic: R² = 0.9978
    Peak²: R² = 0.9115
    Logistic: R² = 0.7525
    Peak: R² = 0.6699
    Michaelis-Menten: R² = 0.2347
    """
    logger = logging.getLogger(__name__)
    x, y = extract_arrays(data)
    models = default_models(x, y)

    fitted_models = []
    for mdl in models:
        logger.info("Fitting model %s", mdl.name)
        try:
            fitted_model, _ = fit_model(x, y, mdl, max_iterations)
            fitted_models.append(fitted_model)
        except (ValueError, RuntimeError):
            logger.warning("Warning: failed to fit %s model.", mdl.name, exc_info=True)
            fitted_models.append(mdl)
        logger.info("Successfully fitted model %s", mdl.name)

    Model.sort(fitted_models)

    return fitted_models


def analyze(args: argparse.Namespace) -> None:
    """Execute the analyze command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Load data
    logger = logging.getLogger(__name__)
    logger.info("Loading data from: %s", args.data_file)
    data = load_csv(args.data_file)
    logger.info("Loaded %s data points", len(data))

    # Fit models
    logger.info("Fitting models...")
    models = fit_all_models(data)
    logger.info("Successfully fitted %s models", len([m for m in models if m.r_squared > -data_dtype(np.inf)]))

    # Print rankings if requested
    if args.show_rankings:
        print(format_model_rankings(models))  # noqa: T201 Print is explicitly requested by user

    # Determine y-axis limits
    y_min = args.y_min if args.y_min is not None else data[RequiredColumn.PRODUCTIVITY].min()
    y_max = args.y_max if args.y_max is not None else data[RequiredColumn.PRODUCTIVITY].max()
    y_limits = (y_min, y_max)

    # Create main analysis plot
    if not args.no_plot:
        logger.info("Creating analysis plot: %s", args.output)
        plot_models(
            data,
            models,
            output_path=args.output,
            fig_params=FigureParameters(y_limits=y_limits, dpi=args.dpi),
            show=Show.YES if args.no_show else Show.NO,
        )

    # Create comparison plot if requested
    if args.comparison is not None:
        logger.info("Creating comparison plot: %s", args.comparison)
        create_comparison_plot(
            data,
            models,
            output_path=args.comparison,
            fig_params=FigureParameters(dpi=args.dpi),
            show=Show.YES if args.no_show else Show.NO,
        )

    logger.info("✓ Analysis complete!")
