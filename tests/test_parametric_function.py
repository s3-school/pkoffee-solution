"""Unit test for parametric functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pkoffee.data import data_dtype as dt
from pkoffee.parametric_function import (
    logistic,
    michaelis_menten_saturation,
    peak2_model,
    peak_model,
    quadratic,
)


def test_quadratic_model() -> None:
    """Test quadratic model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    result = quadratic(x, a0=dt(1.0), a1=dt(2.0), a2=dt(0.5))

    expected = np.array([1.0, 3.5, 7.0, 11.5])
    assert_allclose(result, expected)


def test_michaelis_menten_model() -> None:
    """Test Michaelis-Menten model evaluation."""
    x = np.array([0.0, 1.0, 5.0, 10.0])
    result = michaelis_menten_saturation(x, v_max=dt(10.0), k=dt(2.0), y0=dt(1.0))

    # At x=0: 1.0 + 10*0/(2+0) = 1.0
    # At x=2: 1.0 + 10*2/(2+2) = 6.0
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0 + 10.0 * 1.0 / 3.0)


def test_logistic_model() -> None:
    """Test logistic model evaluation."""
    x = np.array([0.0, 5.0, 10.0])
    result = logistic(x, L=dt(10.0), k=dt(1.0), x0=dt(5.0), y0=dt(0.0))

    # At x0 (midpoint), should be y0 + L/2
    assert result[1] == pytest.approx(5.0, abs=0.01)
    # Function should be monotonically increasing
    assert result[0] < result[1] < result[2]


def test_peak_model() -> None:
    """Test peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    result = peak_model(x, a=dt(5.0), b=dt(2.0))

    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should increase then decrease (peak behavior)
    assert result[1] < result[2]
    assert result[2] > result[3]


def test_peak2_model() -> None:
    """Test quadratic peak model evaluation."""
    x = np.array([0.0, 1.0, 2.0, 10.0])
    result = peak2_model(x, a=dt(1.0), b=dt(3.0))

    # At x=0, result should be 0
    assert result[0] == pytest.approx(0.0)
    # Should have peak behavior
    assert result[1] < result[2]
