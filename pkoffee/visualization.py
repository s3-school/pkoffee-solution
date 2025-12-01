"""
Visualization utilities for coffee productivity analysis.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pkoffee.models import ModelResult


def create_analysis_plot(
    data: pd.DataFrame,
    fitted_models: list[ModelResult],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple[float, float] = (12, 7),
    dpi: int = 150,
    y_limits: Optional[tuple[float, float]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive analysis plot with data distribution and model fits.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.
    fitted_models : list[ModelResult]
        List of fitted models to overlay on the plot.
    output_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple[float, float], optional
        Figure size in inches (width, height), by default (12, 7).
    dpi : int, optional
        Resolution for saved figure, by default 150.
    y_limits : tuple[float, float], optional
        Y-axis limits (min, max). If None, uses automatic scaling.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.

    Examples
    --------
    >>> from pkoffee.data import load_data
    >>> from pkoffee.models import fit_all_models
    >>> data = load_data("coffee_productivity.csv")
    >>> models = fit_all_models(data)
    >>> fig = create_analysis_plot(data, models, "analysis.png")
    """
    # Set seaborn style for beautiful plots
    sns.set_theme(style="whitegrid", palette="husl")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot showing distribution by cups
    sns.violinplot(
        data=data,
        x="cups",
        y="productivity",
        hue="cups",
        ax=ax,
        inner="quartile",
        cut=0,
        density_norm="width",
        palette="Greens",
        linewidth=0.8,
        legend=False,
        alpha=0.7,
    )
    
    # Prepare smooth x values for plotting fitted curves
    x_min = data["cups"].min()
    x_max = data["cups"].max()
    x_smooth = np.linspace(x_min, x_max, 300)
    
    # Plot each fitted model
    colors = plt.cm.tab10(np.linspace(0, 1, len(fitted_models)))
    
    for idx, (model, color) in enumerate(zip(fitted_models, colors)):
        y_smooth = model.predict(x_smooth)
        label = f"{model.name} (R² = {model.r_squared:.3f})"
        ax.plot(
            x_smooth,
            y_smooth,
            lw=2.5,
            label=label,
            color=color,
            alpha=0.9,
            zorder=10 + idx,
        )
    
    # Styling
    ax.set_xlabel("Coffee Cups Consumed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Productivity", fontsize=12, fontweight="bold")
    ax.set_title(
        "Coffee Consumption vs Productivity: Distribution and Model Fits",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    
    if y_limits is not None:
        ax.set_ylim(y_limits)
    
    # Add legend with model rankings
    if fitted_models:
        ax.legend(
            loc="upper left",
            frameon=True,
            shadow=True,
            fancybox=True,
            fontsize=10,
            title="Model Rankings",
            title_fontsize=11,
        )
    else:
        ax.text(
            0.5,
            0.95,
            "No models successfully fitted",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="crimson",
            fontsize=12,
            fontweight="bold",
        )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
    
    # Display figure if requested
    if show:
        plt.show()
    
    return fig


def create_comparison_plot(
    data: pd.DataFrame,
    fitted_models: list[ModelResult],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a multi-panel comparison plot showing each model separately.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.
    fitted_models : list[ModelResult]
        List of fitted models to display.
    output_path : str or Path, optional
        Path to save the figure.
    figsize : tuple[float, float], optional
        Figure size in inches, by default (14, 10).
    dpi : int, optional
        Resolution for saved figure, by default 150.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.
    """
    n_models = len(fitted_models)
    if n_models == 0:
        raise ValueError("No models provided for comparison plot")
    
    # Calculate grid dimensions
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    x = data["cups"].values
    y = data["productivity"].values
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    for idx, (model, ax) in enumerate(zip(fitted_models, axes)):
        # Scatter plot of data
        ax.scatter(x, y, alpha=0.3, s=10, color="gray", label="Data")
        
        # Model fit
        y_smooth = model.predict(x_smooth)
        ax.plot(x_smooth, y_smooth, lw=2, color="blue", label="Model Fit")
        
        # Styling
        ax.set_xlabel("Cups", fontsize=10)
        ax.set_ylabel("Productivity", fontsize=10)
        ax.set_title(f"{model.name}\nR² = {model.r_squared:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        "Model Comparison: Individual Fits",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Comparison plot saved to: {output_path}")
    
    return fig


def print_model_rankings(fitted_models: list[ModelResult]) -> None:
    """
    Print a formatted table of model rankings.

    Parameters
    ----------
    fitted_models : list[ModelResult]
        List of fitted models, should be sorted by R².

    Examples
    --------
    >>> from pkoffee.data import load_data
    >>> from pkoffee.models import fit_all_models
    >>> data = load_data("coffee_productivity.csv")
    >>> models = fit_all_models(data)
    >>> print_model_rankings(models)
    
    Model Rankings:
    ═══════════════════════════════════════
    Rank  Model              R² Score
    ───────────────────────────────────────
       1  Logistic           0.9523
       2  Peak               0.9401
       3  Michaelis-Menten   0.9234
    """
    if not fitted_models:
        print("No models to display.")
        return
    
    print("\nModel Rankings:")
    print("═" * 50)
    print(f"{'Rank':<6} {'Model':<20} {'R² Score':<10}")
    print("─" * 50)
    
    for rank, model in enumerate(fitted_models, start=1):
        r2_str = f"{model.r_squared:.4f}" if np.isfinite(model.r_squared) else "N/A"
        print(f"{rank:<6} {model.name:<20} {r2_str:<10}")
    
    print("═" * 50)
