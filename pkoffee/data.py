"""
Data loading and preprocessing utilities for coffee productivity analysis.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load coffee productivity data from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing the data.
        Expected columns: 'cups' (int) and 'productivity' (float).

    Returns
    -------
    pd.DataFrame
        DataFrame with validated columns 'cups' and 'productivity'.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing or contain invalid data.

    Examples
    --------
    >>> data = load_data("coffee_productivity.csv")
    >>> print(data.head())
       cups  productivity
    0     3      15.25
    1     5      22.50
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e
    
    # Validate required columns
    required_columns = {"cups", "productivity"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data types and values
    if not pd.api.types.is_numeric_dtype(data["cups"]):
        raise ValueError("Column 'cups' must contain numeric values")
    
    if not pd.api.types.is_numeric_dtype(data["productivity"]):
        raise ValueError("Column 'productivity' must contain numeric values")
    
    # Remove any rows with NaN values
    initial_size = len(data)
    data = data.dropna(subset=["cups", "productivity"])
    if len(data) < initial_size:
        dropped = initial_size - len(data)
        print(f"Warning: Dropped {dropped} rows with missing values")
    
    return data


def extract_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract cups and productivity as numpy arrays from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (cups, productivity) as float arrays.

    Examples
    --------
    >>> data = load_data("coffee_productivity.csv")
    >>> cups, productivity = extract_arrays(data)
    >>> print(cups.shape, productivity.shape)
    (1000,) (1000,)
    """
    cups = data["cups"].values.astype(float)
    productivity = data["productivity"].values.astype(float)
    return cups, productivity


def get_data_bounds(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Calculate min and max values for each column in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.

    Returns
    -------
    dict[str, tuple[float, float]]
        Dictionary mapping column names to (min, max) tuples.

    Examples
    --------
    >>> data = load_data("coffee_productivity.csv")
    >>> bounds = get_data_bounds(data)
    >>> print(bounds)
    {'cups': (0.0, 12.0), 'productivity': (0.5, 25.3)}
    """
    return {
        col: (float(data[col].min()), float(data[col].max()))
        for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
    }
