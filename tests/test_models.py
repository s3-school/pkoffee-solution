"""
Unit tests for model functions and fitting.
"""

import numpy as np
import pytest

from pkoffee.models import (
    ModelConfig,
    ModelResult,
    fit_model,
    logistic_model,
    michaelis_menten_model,
    peak2_model,
    peak_model,
    quadratic_model,
)


def test_quadratic_model() -> None:
    """Test quadratic model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    result = quadratic_model(x, a0=1.0, a1=2.0, a2=0.5)
    
    expected = np.array([1.0, 3.5, 7.0, 11.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_michaelis_menten_model() -> None:
    """Test Michaelis-Menten model evaluation."""
    x = np.array([0.0, 1.0, 5.0, 10.0])
    result = michaelis_menten_model(x, v_max=10.0, k=2.0, y0=1.0)
    
    # At x=0: 1.0 + 10*0/(2+0) = 1.0
    # At x=2: 1.0 + 10*2/(2+2) = 6.0
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0 + 10.0 * 1.0 / 3.0)


def test_logistic_model() -> None:
    """Test logistic model evaluation."""
    x = np.array([0.0, 5.0, 10.0])
    result = logistic_model(x, L=10.0, k=1.0, x0=5.0, y0=0.0)
    
    # At x0 (midpoint), should be y0 + L/2
    assert result[1] == pytest.approx(5.0, abs=0.01)
    # Function should be monotonically increasing
    assert result[0] < result[1] < result[2]


def test_peak_model() -> None:
    """Test peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    result = peak_model(x, a=5.0, b=2.0)
    
    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should increase then decrease (peak behavior)
    assert result[1] < result[2]
    assert result[2] > result[3]


def test_peak2_model() -> None:
    """Test quadratic peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    result = peak2_model(x, a=1.0, b=3.0)
    
    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should have peak behavior
    assert result[1] < result[2]


def test_model_result_predict() -> None:
    """Test ModelResult prediction method."""
    def simple_model(x: np.ndarray, m: float, b: float) -> np.ndarray:
        return m * x + b
    
    model_result = ModelResult(
        name="Linear",
        function=simple_model,
        parameters=np.array([2.0, 1.0]),
        r_squared=0.95,
        predictions=np.array([3.0, 5.0, 7.0])
    )
    
    x_new = np.array([0.0, 1.0, 2.0])
    predictions = model_result.predict(x_new)
    
    expected = np.array([1.0, 3.0, 5.0])
    np.testing.assert_array_almost_equal(predictions, expected)


def test_fit_model_success() -> None:
    """Test successful model fitting."""
    # Create simple linear data
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 50)
    
    config = ModelConfig(
        name="Test Linear",
        function=lambda xx, a, b: a * xx + b,
        initial_params=[1.0, 0.0],
        bounds=(-np.inf, np.inf)
    )
    
    result = fit_model(x, y, config)
    
    assert result is not None
    assert result.name == "Test Linear"
    assert result.r_squared > 0.9  # Should fit well
    assert len(result.parameters) == 2
    assert result.parameters[0] == pytest.approx(2.0, abs=0.2)


def test_fit_model_failure() -> None:
    """Test that fit_model returns None on failure."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    
    # Create a config that will fail
    config = ModelConfig(
        name="Bad Model",
        function=lambda xx, a: np.sqrt(a) * xx,
        initial_params=[-10.0], 
        bounds=([-100.0], [-1.0])
    )
    
    result = fit_model(x, y, config)
    
    # Should return None on failure (RuntimeWarning from sqrt of negative)
    assert result is None 


def test_model_result_repr() -> None:
    """Test ModelResult string representation."""
    model_result = ModelResult(
        name="TestModel",
        function=lambda x: x,
        parameters=np.array([1.0, 2.0]),
        r_squared=0.8765,
        predictions=np.array([1.0, 2.0])
    )
    
    repr_str = repr(model_result)
    
    assert "TestModel" in repr_str
    assert "0.8765" in repr_str
