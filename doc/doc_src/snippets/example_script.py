"""Script demonstrating how to use the PKoffee package to analyze coffee productivity data."""  # noqa: INP001 example script

import logging
from pathlib import Path

from pkoffee.data import load_csv
from pkoffee.fit_models import Model
from pkoffee.productivity_analysis import fit_all_models
from pkoffee.visualization import FigureParameters, Show, format_model_rankings, plot_models


def main() -> None:
    """Run example analysis."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    # Path to your data file
    data_file = Path("../example_data/coffee_productivity.csv")

    # Load data
    logger.info("Loading data...")
    data = load_csv(data_file)
    logger.info("Loaded %s data points", len(data))
    logger.info("Cups range: %s to %s", data["cups"].min(), data["cups"].max())
    logger.info(
        "Productivity range: %s to %s", f"{data['productivity'].min():.2f}", f"{data['productivity'].max():.2f}"
    )

    # Fit models
    logger.info("Fitting models...")
    models = fit_all_models(data)
    logger.info("Successfully fitted %s models", len([m for m in models if m.r_squared > 0]))

    # sort the models per best fit
    Model.sort(models)

    # logger.info rankings
    logger.info(format_model_rankings(models))

    # Create visualization
    logger.info("Creating visualization...")
    output_path = Path("analysis_results.png")
    plot_models(
        data,
        models,
        output_path=output_path,
        fig_params=FigureParameters(y_limits=(-0.2, 8.0)),
        show=Show.NO,  # deactivate showing the figure to run in CI
    )

    logger.info("✓ Analysis complete! Results saved to: %s", output_path)

    # Access best model
    best_model = models[0]
    logger.info("Best model: %s", best_model.name)
    logger.info("R² score: %s", f"{best_model.r_squared:.4f}")
    logger.info("Parameters: %s", best_model.params)


if __name__ == "__main__":
    main()
