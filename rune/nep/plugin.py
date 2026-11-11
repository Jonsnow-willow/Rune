from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ase.data import atomic_numbers

from .model import NEP


PLUGIN_NAME = "nep4"


_DESCRIPTOR_DEFAULTS: dict[str, Any] = {
    "rcut_radial": 8.0,
    "rcut_angular": 4.0,
    "n_max_radial": 4,
    "n_max_angular": 4,
    "basis_size_radial": 8,
    "basis_size_angular": 8,
    "l_max": 4,
    "l_max_4body": 2,
    "l_max_5body": 0,
    "has_q_112": 0,
    "has_q_123": 0,
    "has_q_233": 0,
    "has_q_134": 0,
    "max_neighbors_radial": 100,
    "max_neighbors_angular": 30,
    "zbl": None,
}


def resolve_params(values: dict) -> dict:
    resolved = _resolve_descriptor_config(values, extra_defaults={"hidden_dims": [30]})
    hidden_dims = resolved["hidden_dims"]
    if (
        not isinstance(hidden_dims, list)
        or len(hidden_dims) != 1
        or not isinstance(hidden_dims[0], int)
        or isinstance(hidden_dims[0], bool)
        or not 1 <= hidden_dims[0] <= 200
    ):
        raise ValueError(
            "NEP hidden_dims must contain exactly one positive integer in [1, 200]. "
            "RUNE currently supports single-hidden-layer NEP models for GPUMD export."
        )
    resolved["hidden_dims"] = list(hidden_dims)
    return resolved


def build_model(params: dict):
    return NEP(params)


def get_ddp_options(params: dict) -> dict:
    # A batch generally activates only a subset of element-specific MLPs.
    return {"find_unused_parameters": True}


def load_model(path, device=None):
    from ..runtime.checkpoint import read_checkpoint

    path = Path(path)
    if path.suffix.lower() == ".txt":
        return NEP.from_nep_txt(path, device=device)
    checkpoint = read_checkpoint(path, map_location=device)
    identity = checkpoint.get("plugin")
    if not isinstance(identity, dict) or identity.get("name") != PLUGIN_NAME:
        raise ValueError(f"Checkpoint is not a {PLUGIN_NAME} model: {path}")
    params = checkpoint.get("model_params")
    if not isinstance(params, dict):
        raise ValueError(f"Checkpoint does not contain resolved model_params: {path}")
    model = NEP(params)
    parameter_name, _ = next(model.named_parameters())
    model.to(dtype=checkpoint["model_state_dict"][parameter_name].dtype)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if device is not None:
        model.to(device)
    return model


def export_model(model, path):
    path = Path(path)
    if path.name == "model":
        path = path.with_name("nep.txt")
    elif path.name.startswith("model_epoch_"):
        path = path.with_name(path.name.replace("model_", "nep_") + ".txt")
    model.save_nep_txt(path)
    return path


def build_calculator(artifact, **kwargs):
    from .interface.calculator import Calculator

    return Calculator(artifact, **kwargs)


def _resolve_descriptor_config(
    params: Mapping[str, Any],
    *,
    extra_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve shared NEP-descriptor fields for descriptor-based plugins."""

    if not isinstance(params, Mapping):
        raise TypeError("Model params must be a TOML table.")
    defaults = dict(_DESCRIPTOR_DEFAULTS)
    defaults.update(dict(extra_defaults or {}))
    unknown = sorted(set(params) - set(defaults) - {"elements"})
    if unknown:
        raise ValueError(f"Unknown model parameter(s): {', '.join(unknown)}")
    if "elements" not in params:
        raise ValueError('Model params must define elements = ["..."].')

    values = {**defaults, **dict(params)}
    elements = values["elements"]
    if (
        not isinstance(elements, list)
        or not elements
        or not all(isinstance(element, str) for element in elements)
    ):
        raise ValueError("elements must be a non-empty list of chemical symbols.")
    if len(set(elements)) != len(elements):
        raise ValueError("elements must not contain duplicates.")
    if len(elements) > 94:
        raise ValueError("GPUMD supports at most 94 elements (H through Pu).")
    invalid = [element for element in elements if element not in atomic_numbers]
    if invalid:
        raise ValueError(f"Unknown chemical element symbols: {invalid}.")
    unsupported = [
        element for element in elements if not 1 <= atomic_numbers[element] <= 94
    ]
    if unsupported:
        raise ValueError(
            f"GPUMD supports elements H through Pu; unsupported: {unsupported}."
        )

    for name in ("rcut_radial", "rcut_angular"):
        if (
            not isinstance(values[name], (int, float))
            or isinstance(values[name], bool)
            or not isfinite(values[name])
        ):
            raise ValueError(f"{name} must be a finite number.")
    if not 3.0 <= values["rcut_angular"] <= values["rcut_radial"] <= 100.0:
        raise ValueError(
            "NEP cutoffs must satisfy 3 <= rcut_angular <= rcut_radial <= 100 Å."
        )

    # GPUMD a8416f3: src/main_nep/parameters.cu input and descriptor limits.
    integer_ranges = {
        "n_max_radial": (0, 12),
        "n_max_angular": (0, 8),
        "basis_size_radial": (0, 16),
        "basis_size_angular": (0, 12),
        "l_max": (2, 8),
    }
    for name, (lower, upper) in integer_ranges.items():
        if (
            not isinstance(values[name], int)
            or isinstance(values[name], bool)
            or not lower <= values[name] <= upper
        ):
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")

    higher_order_fields = (
        "l_max_4body",
        "l_max_5body",
        "has_q_112",
        "has_q_123",
        "has_q_233",
        "has_q_134",
    )
    for name in higher_order_fields:
        allowed = (0, 2) if name == "l_max_4body" else (0, 1)
        if (
            not isinstance(values[name], int)
            or isinstance(values[name], bool)
            or values[name] not in allowed
        ):
            raise ValueError(f"{name} must be one of {allowed}.")
    if (values["has_q_123"] or values["has_q_233"]) and values["l_max"] < 3:
        raise ValueError("q_123 and q_233 require l_max >= 3.")
    if values["has_q_134"] and values["l_max"] < 4:
        raise ValueError("q_134 requires l_max >= 4.")
    num_angular_channels = values["l_max"] + sum(
        values[name] > 0 for name in higher_order_fields
    )
    if (values["n_max_angular"] + 1) * num_angular_channels > 90:
        raise ValueError("The number of angular descriptors must not exceed 90.")

    for name in ("max_neighbors_radial", "max_neighbors_angular"):
        if (
            not isinstance(values[name], int)
            or isinstance(values[name], bool)
            or values[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    return {
        "elements": list(elements),
        "rcut_radial": float(values["rcut_radial"]),
        "rcut_angular": float(values["rcut_angular"]),
        "n_desc_radial": int(values["n_max_radial"]) + 1,
        "n_desc_angular": int(values["n_max_angular"]) + 1,
        "k_max_radial": int(values["basis_size_radial"]) + 1,
        "k_max_angular": int(values["basis_size_angular"]) + 1,
        "l_max": int(values["l_max"]),
        "l_max_4body": int(values["l_max_4body"]),
        "l_max_5body": int(values["l_max_5body"]),
        "has_q_222": int(int(values["l_max_4body"]) > 0),
        "has_q_1111": int(int(values["l_max_5body"]) > 0),
        "has_q_112": int(int(values["has_q_112"]) > 0),
        "has_q_123": int(int(values["has_q_123"]) > 0),
        "has_q_233": int(int(values["has_q_233"]) > 0),
        "has_q_134": int(int(values["has_q_134"]) > 0),
        "max_neighbors_radial": int(values["max_neighbors_radial"]),
        "max_neighbors_angular": int(values["max_neighbors_angular"]),
        "n_types": len(elements),
        **{name: _copy_config_value(values[name]) for name in (extra_defaults or {})},
        **(
            {"zbl": _copy_config_value(values["zbl"])}
            if values["zbl"] is not None
            else {}
        ),
    }


def _copy_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_config_value(item) for item in value]
    return value
