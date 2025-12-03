"""Unit tests for models and fitting."""

import numpy as np
import pytest

from pkoffee.data import data_dtype as dt
from pkoffee.fit_models import (
    Model,
    ParametersBounds,
    fit_model,
)


def test_model_result_predict() -> None:
    """Test ModelResult prediction method."""

    def simple_model(x: np.ndarray, m: float, b: float) -> np.ndarray:
        return m * x + b

    model_result = Model(
        name="Linear",
        function=simple_model,
        params={"m": dt(2.0), "b": dt(1.0)},
        bounds=ParametersBounds(min=(dt(-10), dt(10)), max=(dt(-100), dt(100))),
        r_squared=dt(0.95),
    )

    x_new = np.array([0.0, 1.0, 2.0])
    predictions = model_result.predict(x_new)

    expected = np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(predictions, expected)


def test_fit_model_success() -> None:
    """Test successful model fitting."""
    # Create simple linear data
    x = np.linspace(0, 10, 50)
    rng = np.random.default_rng(1337)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 50)

    lin_model = Model(
        name="Test Linear",
        function=lambda xx, a, b: a * xx + b,
        params={"a": dt(1.0), "b": dt(0.0)},
        bounds=ParametersBounds(min=(dt(-np.inf),), max=(dt(np.inf),)),
    )

    result, _ = fit_model(x, y, lin_model)

    assert result is not None
    assert result.name == "Test Linear"
    assert result.r_squared > 0.9  # Should fit well  # noqa: PLR2004
    assert len(result.params) == 2  # noqa: PLR2004
    assert result.params["a"] == pytest.approx(2.0, abs=0.2)


def test_fit_model_failure() -> None:
    """Test that fit_model returns None on failure."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    # Create a config that will fail
    config = Model(
        name="Bad Model",
        function=lambda xx, a: np.sqrt(a) * xx,
        params={"a": dt(-10.0)},
        bounds=ParametersBounds(min=(dt(-100.0),), max=(dt(-1.0),)),
    )

    with pytest.raises(ValueError):  # noqa: PT011 error raised by scipy
        _ = fit_model(x, y, config)


def test_model_result_repr() -> None:
    """Test ModelResult string representation."""
    model = Model(
        name="TestModel",
        function=lambda x: x,
        params={},
        r_squared=dt(0.8765),
        bounds=ParametersBounds((dt(0),), (dt(1),)),
    )

    repr_str = repr(model)

    assert repr_str == f"ModelFit(name='{model.name}', R²={model.r_squared:.3f})"


def test_model_sort() -> None:
    """Test Model.sort."""
    r_squared_min = dt(0.1)
    r_squared_max = dt(0.3)
    model_list = [
        Model(
            name="a",
            function=lambda x: x,
            params={},
            r_squared=r_squared_max,
            bounds=ParametersBounds(min=(dt(0),), max=(dt(10.0),)),
        ),
        Model(
            name="b",
            function=lambda x: x,
            params={},
            r_squared=r_squared_min,
            bounds=ParametersBounds(min=(dt(0),), max=(dt(10.0),)),
        ),
    ]
    Model.sort(model_list)
    assert model_list[0].r_squared == r_squared_max
    assert model_list[1].r_squared == r_squared_min
