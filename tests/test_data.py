"""
Unit tests for data loading and preprocessing.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkoffee.data import extract_arrays, get_data_bounds, load_data


def test_load_data_valid_file() -> None:
    """Test loading valid CSV data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("cups,productivity\n")
        f.write("1,10.5\n")
        f.write("2,15.3\n")
        f.write("3,18.2\n")
        temp_path = f.name
    
    try:
        data = load_data(temp_path)
        assert len(data) == 3
        assert "cups" in data.columns
        assert "productivity" in data.columns
        assert data["cups"].iloc[0] == 1
        assert data["productivity"].iloc[0] == 10.5
    finally:
        Path(temp_path).unlink()


def test_load_data_missing_file() -> None:
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_load_data_missing_columns() -> None:
    """Test that ValueError is raised for missing required columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("cups,wrong_column\n")
        f.write("1,10.5\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_data_with_nan_values() -> None:
    """Test that rows with NaN values are dropped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("cups,productivity\n")
        f.write("1,10.5\n")
        f.write("2,\n")  # Missing productivity
        f.write("3,18.2\n")
        temp_path = f.name
    
    try:
        data = load_data(temp_path)
        assert len(data) == 2  # One row dropped
        assert not data.isnull().any().any()
    finally:
        Path(temp_path).unlink()


def test_extract_arrays() -> None:
    """Test extracting numpy arrays from DataFrame."""
    data = pd.DataFrame({
        "cups": [1, 2, 3],
        "productivity": [10.5, 15.3, 18.2]
    })
    
    cups, productivity = extract_arrays(data)
    
    assert isinstance(cups, np.ndarray)
    assert isinstance(productivity, np.ndarray)
    assert cups.dtype == np.float64
    assert productivity.dtype == np.float64
    assert len(cups) == 3
    assert len(productivity) == 3
    np.testing.assert_array_equal(cups, [1.0, 2.0, 3.0])


def test_get_data_bounds() -> None:
    """Test calculating data bounds."""
    data = pd.DataFrame({
        "cups": [1, 2, 3, 4, 5],
        "productivity": [10.5, 15.3, 18.2, 22.1, 25.0]
    })
    
    bounds = get_data_bounds(data)
    
    assert "cups" in bounds
    assert "productivity" in bounds
    assert bounds["cups"] == (1.0, 5.0)
    assert bounds["productivity"] == (10.5, 25.0)
