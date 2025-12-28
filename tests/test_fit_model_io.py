"""Test suite of `pkoffee.fit_model_io`."""

import json
from dataclasses import astuple
from pathlib import Path

import tomlkit

from pkoffee.data import data_dtype as dt
from pkoffee.fit_model import Model
from pkoffee.fit_model_io import (
    ModelFileFormat,
    pkoffee_function_id_mapping,
    save_models,
    save_models_json,
    save_models_toml,
)
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    ParametricFunction,
    Peak2Model,
    PeakModel,
    Quadratic,
)


def test_pkoffee_function_id_mapping() -> None:
    """Test the function to string identifier bijection mapping."""
    mapping = pkoffee_function_id_mapping()
    assert mapping["Quadratic"] == Quadratic
    assert mapping.inv[Quadratic] == "Quadratic"
    assert mapping["Logistic"] == Logistic
    assert mapping.inv[Logistic] == "Logistic"
    assert mapping["MichaelisMentenSaturation"] == MichaelisMentenSaturation
    assert mapping.inv[MichaelisMentenSaturation] == "MichaelisMentenSaturation"
    assert mapping["PeakModel"] == PeakModel
    assert mapping.inv[PeakModel] == "PeakModel"
    assert mapping["Peak2Model"] == Peak2Model
    assert mapping.inv[Peak2Model] == "Peak2Model"


def assert_model_equal(lhs: Model, rhs: Model) -> None:
    """Assert 2 models are equal."""
    # Assert both are models
    assert isinstance(lhs, Model)
    assert type(lhs) is type(rhs)
    # Assert models functions are the same
    assert type(lhs.function) is type(rhs.function)
    # assert equality of all other fields
    for lhs_field, rhs_field in zip(astuple(lhs), astuple(rhs), strict=True):
        if isinstance(lhs_field, ParametricFunction):
            continue
        assert lhs_field == rhs_field


def test_save_models_json(tmp_path: Path) -> None:
    """Test model saving to json."""
    ref_model = Model(
        name="TestModel",
        function=Quadratic(),
        params=Quadratic.param_guess(dt(0.0)),
        bounds=Quadratic.param_bounds(),
        r_squared=dt(0.8765),
    )
    model_json_file = tmp_path / "test_save_models_json.json"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models_json([ref_model.to_dict(fct_id_to_fct.inv)], model_json_file)
    with model_json_file.open("r") as mdlf:
        read_model = Model.from_dict(json.loads(mdlf.read())["Models"][0], fct_id_to_fct)

    assert_model_equal(ref_model, read_model)


def test_save_models_toml(tmp_path: Path) -> None:
    """Test model saving to toml."""
    ref_model = Model(
        name="TestModel",
        function=Quadratic(),
        params=Quadratic.param_guess(dt(0.0)),
        bounds=Quadratic.param_bounds(),
        r_squared=dt(0.8765),
    )
    model_toml_file = tmp_path / "test_save_models_toml.toml"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models_toml([ref_model.to_dict(fct_id_to_fct.inv)], model_toml_file)
    with model_toml_file.open("r") as mdlf:
        read_model = Model.from_dict(tomlkit.parse(mdlf.read())["Models"][0], fct_id_to_fct)  # pyright: ignore[reportArgumentType, reportIndexIssue]

    assert_model_equal(ref_model, read_model)


def test_save_models(tmp_path: Path) -> None:
    """Test saving several models."""
    models = [
        Model(
            name="QuadraticBest",
            function=Quadratic(),
            params=Quadratic.param_guess(dt(0.0)),
            bounds=Quadratic.param_bounds(),
            r_squared=dt(0.8765),
        ),
        Model(
            name="QuadraticWorse",
            function=Quadratic(),
            params=Quadratic.param_guess(dt(0.0)),
            bounds=Quadratic.param_bounds(),
            r_squared=dt(0.2244),
        ),
        Model(
            name="Logistic",
            function=Logistic(),
            params=Logistic.param_guess(dt(0), dt(5), dt(0), dt(10)),
            bounds=Logistic.param_bounds(),
            r_squared=dt(0.7542),
        ),
    ]
    model_toml_file = tmp_path / "test_save_models_toml.toml"
    fct_id_to_fct = pkoffee_function_id_mapping()
    save_models(models, fct_id_to_fct.inv, model_toml_file, ModelFileFormat.TOML)
    with model_toml_file.open("r") as mdlf:
        model_dicts = tomlkit.parse(mdlf.read())["Models"]
        for ref_md, read_md in zip(models, model_dicts, strict=True):  # pyright: ignore[reportArgumentType] model_dicts is iterable alright
            assert_model_equal(ref_md, Model.from_dict(read_md, fct_id_to_fct))
