"""
PKoffee - Coffee Productivity Analysis Package.

A comprehensive toolkit for analyzing the relationship between coffee consumption
and productivity through statistical modeling and visualization.
"""

from pkoffee.data import load_data
from pkoffee.models import fit_all_models, ModelResult
from pkoffee.metrics import compute_r2
from pkoffee.visualization import create_analysis_plot

__version__ = "0.1.0"
__all__ = [
    "load_data",
    "fit_all_models",
    "ModelResult",
    "compute_r2",
    "create_analysis_plot",
]
