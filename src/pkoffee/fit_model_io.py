"""Input/Output for models.

A model's `ParametricFunction` is not directly saved, only an identifier string is written to file. In order to
reconstruct the model from the file, the same mapping from function to identifier needs to be available. This module
implements a function returning a bidirectional mapping for the `ParametricFunction` implemented in the `pkoffeee`
package. This mapping can be extended with additional functions to save other models.
"""

import json
import typing
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path

from bidict import bidict
from tomlkit import aot, document, item

if typing.TYPE_CHECKING:
    from pkoffee.fit_model import Model  # only use Model for type hints of save models functions
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    ParametricFunction,
    Peak2Model,
    PeakModel,
    Quadratic,
)


class FunctionNotFoundInMappingError(KeyError):
    """Exception when a function is not found in the function to str mapping."""

    def __init__(self, function: type[ParametricFunction], mapping: Mapping) -> None:
        super().__init__(f"Function {function} not found in function to str mapping {mapping}")


class FunctionIdNotFoundInMappingError(KeyError):
    """Exception when a function Identifier is not found in the function Id to function mapping."""

    def __init__(self, function_id: str, mapping: Mapping) -> None:
        super().__init__(f"Function Identifier {function_id} not found in mapping to function {mapping}")


class ModelParsingError(ValueError):
    """Exception when a model dictionary representation can not be parsed into a model."""

    def __init__(self, model_dict: Mapping) -> None:
        super().__init__(f"Could not parse model dictionary {model_dict}, missing fields or bad types?")


def pkoffee_function_id_mapping() -> bidict:
    """Bidirectional mapping from string Identifiers to the ParametricFunctions implemented in `pkoffee`."""
    return bidict(
        {
            "Quadratic": Quadratic,
            "MichaelisMentenSaturation": MichaelisMentenSaturation,
            "Logistic": Logistic,
            "PeakModel": PeakModel,
            "Peak2Model": Peak2Model,
        }
    )


class UnsupportedModelFormatError(NotImplementedError):
    """Exception for non-implemented model file format."""

    def __init__(self, file_format: str) -> None:
        super().__init__(f"Model format {file_format} not supported. See ModelFileFormat.")


class ModelFileFormat(Enum):
    """Available format for saving models to file."""

    TOML = 1
    JSON = 2


def save_models_json(model_dicts: Iterable[dict], output_path: Path) -> None:
    """Save the model dictionary representation to a json file.

    Parameters
    ----------
    models : Iterable[dict]
        Models dictionary representation
    output_path : Path
        Path to save the models
    """
    with output_path.open("w") as of:
        of.write(json.dumps({"Models": model_dicts}))


def save_models_toml(model_dicts: Iterable[dict], output_path: Path) -> None:
    """Save the model dictionaries representation to a toml file.

    Parameters
    ----------
    model_dicts : Iterable[dict]
        Models dictionary representation
    output_path : Path
        Path to save the models
    """
    toml_doc = document()
    models_array = aot()
    for md in model_dicts:
        models_array.append(item(md))
    toml_doc.append("Models", models_array)
    with output_path.open("w") as of:
        of.write(toml_doc.as_string())


def save_models(
    models: "Iterable[Model]", function_to_str: Mapping, output_path: Path, file_format: ModelFileFormat
) -> None:
    """Save the models to disk.

    Parameters
    ----------
    models : Iterable[Model]
        Collection of models to save
    function_to_str : Mapping
        Mapping of function to string identifier used as function representation in the model's file.
    output_path : Path
        Path to the model's file.
    file_format : ModelFileFormat
        The format of the model's file
    """
    model_dicts = [m.to_dict(function_to_str) for m in models]
    match file_format:
        case ModelFileFormat.JSON:
            save_models_json(model_dicts, output_path)
        case ModelFileFormat.TOML:
            save_models_toml(model_dicts, output_path)
        case _:
            raise ModuleNotFoundError(str(file_format))
