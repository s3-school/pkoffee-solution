# PKoffee - Coffee Productivity Analysis

A Python package for analyzing the relationship between coffee consumption and productivity through statistical modeling and visualization.

Project inspired by _Le Café - Oldelaf_ for the S3 School 2026.

[![Le Café - Oldelaf - on YouTube](http://img.youtube.com/vi/UGtKGX8B9hU/0.jpg)](http://www.youtube.com/watch?v=UGtKGX8B9hU "Le Café - Oldelaf")


## Features

- **Data Analysis**: Load and analyze coffee consumption vs productivity datasets
- **Multiple Models**: Fit various mathematical models (quadratic, Michaelis-Menten, logistic, peak functions)
- **Visualization**: Beautiful violin plots with model overlays using Seaborn
- **Model Comparison**: Automatic R² scoring and ranking of models
- **Type-Safe**: Full type hints for better IDE support and code quality

## Installation



```bash
pip install -e .
```

## Quick Start

```python
from pkoffee.data import load_data
from pkoffee.models import fit_all_models
from pkoffee.visualization import create_analysis_plot

# Load your data
data = load_data("coffee_productivity.csv")

# Fit models
fitted_models = fit_all_models(data)

# Create visualization
create_analysis_plot(data, fitted_models, output_path="results.png")
```

## Command Line Interface

```bash
# Analyze
cd analysis
pkoffee-sol analyze coffee_productivity.csv --output analysis.png
```

# View model rankings

```
cd analysis
pkoffee-sol analyze coffee_productivity.csv --show-rankings
```

## Project Structure

```
pkoffee-sol/
├── pkoffee/
│   ├── __init__.py          # Package initialization
│   ├── data.py              # Data loading utilities
│   ├── models.py            # Mathematical models and fitting
│   ├── metrics.py           # Model evaluation metrics
│   ├── visualization.py     # Plotting functions
│   └── cli.py               # Command-line interface
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
├── README.md
├── pyproject.toml
└── setup.py
```

## Models Implemented

1. **Quadratic**: `f(x) = a₀ + a₁x + a₂x²`
2. **Michaelis-Menten (Saturating)**: `f(x) = y₀ + Vₘₐₓ·x/(K + x)`
3. **Logistic**: `f(x) = y₀ + L/(1 + e^(-k(x - x₀)))`
4. **Peak**: `f(x) = a·x·e^(-x/b)`
5. **Peak2**: `f(x) = a·x²·e^(-x/b)`

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy pkoffee/

# Linting
ruff check pkoffee/
```

## License

MIT License
