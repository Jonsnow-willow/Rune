from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ase.data import atomic_numbers

from .model import Mobius


PLUGIN_NAME = "mobius-force-field"


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


_DEFAULTS = {
    "hidden_dim": 96,
    "blocks": 3,
    "ffn_dim": 192,
    "activation": "silu",
    "message_layers": 2,
    "message_dim": 64,
    "pair_dim": 16,
    "message_groups": 8,
    "vector_channels": 8,
    "message_basis_size": None,
    "message_cutoff": None,
    "message_mode": "scalar",
    "equivariant_channels": 8,
    "equivariant_l_max": None,
    "norm_eps": 1.0e-8,
}


def resolve_params(values: dict) -> dict:
    resolved = _resolve_descriptor_config(values, extra_defaults=_DEFAULTS)
    for name in ("hidden_dim", "ffn_dim", "message_dim", "pair_dim", "message_groups"):
        if (
            not isinstance(resolved[name], int)
            or isinstance(resolved[name], bool)
            or resolved[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("blocks", "message_layers", "vector_channels"):
        if (
            not isinstance(resolved[name], int)
            or isinstance(resolved[name], bool)
            or resolved[name] < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer.")
    if resolved["message_mode"] not in ("scalar", "equivariant"):
        raise ValueError("message_mode must be 'scalar' or 'equivariant'.")
    if resolved["message_mode"] == "equivariant":
        channels = resolved["equivariant_channels"]
        order = resolved["equivariant_l_max"]
        if order is None:
            order = resolved["l_max"]
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("equivariant_channels must be a positive integer.")
        if (
            not isinstance(order, int)
            or isinstance(order, bool)
            or not 1 <= order <= min(resolved["l_max"], 8)
        ):
            raise ValueError("equivariant_l_max must be in 1..min(l_max, 8).")
        if resolved["message_layers"] < 1:
            raise ValueError("equivariant message_mode requires message_layers >= 1.")
        resolved["equivariant_l_max"] = order
    else:
        if "equivariant_channels" in values or "equivariant_l_max" in values:
            raise ValueError("Equivariant parameters require message_mode='equivariant'.")
        # Preserve the resolved identity of existing scalar/Local checkpoints.
        for name in ("message_mode", "equivariant_channels", "equivariant_l_max"):
            resolved.pop(name)
    if resolved["hidden_dim"] % resolved["message_groups"] != 0:
        raise ValueError("hidden_dim must be divisible by message_groups.")
    if resolved["message_basis_size"] is None:
        resolved["message_basis_size"] = resolved["k_max_radial"]
    if (
        not isinstance(resolved["message_basis_size"], int)
        or resolved["message_basis_size"] < 1
    ):
        raise ValueError("message_basis_size must be a positive integer.")
    if resolved["message_cutoff"] is None:
        resolved["message_cutoff"] = resolved["rcut_radial"]
    if not 0.0 < float(resolved["message_cutoff"]) <= resolved["rcut_radial"]:
        raise ValueError(
            "message_cutoff must be positive and cannot exceed rcut_radial."
        )
    resolved["message_cutoff"] = float(resolved["message_cutoff"])
    if not isinstance(resolved["activation"], str) or not resolved["activation"]:
        raise ValueError("activation must be a non-empty string.")
    if not isinstance(resolved["norm_eps"], (int, float)) or resolved["norm_eps"] <= 0:
        raise ValueError("norm_eps must be positive.")
    resolved["norm_eps"] = float(resolved["norm_eps"])
    return resolved


def build_model(params: dict):
    return Mobius(params)


def get_ddp_options(params: dict) -> dict:
    return {"find_unused_parameters": False}


def load_model(path, device=None):
    from ..runtime.checkpoint import read_checkpoint

    path = Path(path)
    checkpoint = read_checkpoint(path, map_location=device)
    identity = checkpoint.get("plugin")
    if not isinstance(identity, dict) or identity.get("name") != PLUGIN_NAME:
        raise ValueError(f"Checkpoint is not a {PLUGIN_NAME} model: {path}")
    params = checkpoint.get("model_params")
    if not isinstance(params, dict):
        raise ValueError(f"Checkpoint does not contain resolved model_params: {path}")
    model = Mobius(params)
    parameter_name, _ = next(model.named_parameters())
    model.to(dtype=checkpoint["model_state_dict"][parameter_name].dtype)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if device is not None:
        model.to(device)
    return model


def build_calculator(artifact, **kwargs):
    from .interface.calculator import Calculator

    return Calculator(artifact, **kwargs)


def export_model(model, path):
    from .export import export_model as write_model

    return write_model(model, path)


def _resolve_descriptor_config(
    params: Mapping[str, Any],
    *,
    extra_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve Möbius-owned invariant-descriptor fields."""

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
    invalid = [element for element in elements if element not in atomic_numbers]
    if invalid:
        raise ValueError(f"Unknown chemical element symbols: {invalid}.")

    for name in ("rcut_radial", "rcut_angular"):
        if (
            not isinstance(values[name], (int, float))
            or isinstance(values[name], bool)
            or values[name] <= 0
        ):
            raise ValueError(f"{name} must be positive.")
    for name in (
        "n_max_radial",
        "n_max_angular",
        "basis_size_radial",
        "basis_size_angular",
        "l_max",
        "max_neighbors_radial",
        "max_neighbors_angular",
    ):
        if (
            not isinstance(values[name], int)
            or isinstance(values[name], bool)
            or values[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if values["l_max"] > 8:
        raise ValueError("l_max must be in 1..8.")

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
