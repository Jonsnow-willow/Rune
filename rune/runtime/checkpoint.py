from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .distributed import unwrap_model


class Checkpoint:
    def __init__(
        self,
        path: str | Path,
        config: dict,
        plugin_identity: dict,
        model_params: dict,
        create_dirs: bool = True,
    ):
        self.directory = Path(path)
        self.config = config
        self.plugin_identity = plugin_identity
        self.model_params = model_params
        if create_dirs:
            create_output_dirs(path)

    def path(self, name: str) -> Path:
        return self.directory / "checkpoints" / name

    def save(
        self,
        name: str,
        model,
        optimizer,
        epoch: int,
        best_metric: float,
        metrics: dict[str, Any] | None = None,
        scheduler=None,
        *,
        overwrite: bool = True,
    ) -> Path:
        path = self.path(name)
        if not overwrite and os.path.lexists(path):
            raise FileExistsError(
                f"Checkpoint already exists: {path}. "
                "Use a new run directory to retain a different training history."
            )
        base_model = unwrap_model(model)
        payload = {
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "metrics": metrics or {},
            "plugin": dict(self.plugin_identity),
            "model_params": dict(self.model_params),
            "run_config": self.config,
            "rng_state": _get_rng_state(),
        }
        stream = tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        )
        temporary_path = Path(stream.name)
        try:
            with stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary_path, path)
            else:
                # Publish only a complete file, without replacing an existing name.
                os.link(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def load(
        self,
        path: str | Path,
        model,
        optimizer=None,
        scheduler=None,
        device=None,
        set_rng: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        if not path.is_absolute():
            path = self.directory / path
        checkpoint = read_checkpoint(path, map_location=device)
        base_model = unwrap_model(model)
        _check_plugin(
            checkpoint.get("plugin"),
            self.plugin_identity,
            checkpoint_label=str(path),
        )
        _check_params(
            checkpoint.get("model_params"),
            self.model_params,
            checkpoint_label=str(path),
        )
        if optimizer is not None:
            if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
                raise ValueError(
                    "Full-state resume requires optimizer_state_dict; "
                    "use weights-only fine-tuning for inference checkpoints."
                )
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler is not None and not isinstance(scheduler_state, dict):
                raise ValueError("Full-state resume requires scheduler_state_dict.")
            if scheduler is None and scheduler_state is not None:
                raise ValueError(
                    "Full-state resume cannot discard the checkpoint scheduler."
                )
            saved_scheduler = checkpoint.get("run_config", {}).get("scheduler", {})
            if saved_scheduler.get("name") != self.config["scheduler"]["name"]:
                raise ValueError(
                    "Full-state resume requires the same scheduler as the checkpoint."
                )
            rng_state = checkpoint.get("rng_state")
            if set_rng and (
                not isinstance(rng_state, dict)
                or any(rng_state.get(key) is None for key in ("python", "numpy", "torch"))
            ):
                raise ValueError("Full-state resume requires complete rng_state.")
        base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if set_rng:
            _set_rng_state(checkpoint.get("rng_state"))
        return checkpoint


def _check_plugin(
    checkpoint_identity: Any,
    current_identity: Any,
    *,
    checkpoint_label: str = "checkpoint",
) -> None:
    if not isinstance(checkpoint_identity, dict):
        raise ValueError(
            f"Cannot safely load {checkpoint_label}: checkpoint has no plugin identity."
        )
    if not isinstance(current_identity, dict):
        raise ValueError("Current run config has no plugin identity.")
    keys = ("name", "reference")
    differences = [
        f"{key}: checkpoint={checkpoint_identity.get(key)!r}, current={current_identity.get(key)!r}"
        for key in keys
        if checkpoint_identity.get(key) != current_identity.get(key)
    ]
    if differences:
        raise ValueError(
            f"Cannot safely load {checkpoint_label}: plugin identity differs "
            f"({'; '.join(differences)})."
        )


def _check_params(
    checkpoint_params: Any,
    model_params: Any,
    *,
    checkpoint_label: str = "checkpoint",
) -> None:
    """Reject changes to a plugin's fully resolved model parameters."""

    if not isinstance(checkpoint_params, dict):
        raise ValueError(
            f"Cannot safely load {checkpoint_label}: checkpoint has no resolved model_params."
        )
    left = _normalize(checkpoint_params)
    right = _normalize(model_params)
    if left != right:
        raise ValueError(
            f"Cannot safely load {checkpoint_label}: resolved model parameters differ."
        )


def _normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _normalize(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, dict):
        return tuple(
            (str(key), _normalize(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_checkpoint(path: str | Path, map_location=None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def create_output_dirs(path: str | Path) -> None:
    path = Path(path)
    for name in ("checkpoints", "exports", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu()
        torch.set_rng_state(torch_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [
            cuda_state.detach().cpu()
            if isinstance(cuda_state, torch.Tensor)
            else cuda_state
            for cuda_state in state["cuda"]
        ]
        _set_cuda_rng_state(cuda_states)


def _set_cuda_rng_state(cuda_states: list[Any]) -> None:
    if not cuda_states:
        return
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return

    local_rank = _parse_int(os.environ.get("LOCAL_RANK"))
    if local_rank is not None and 0 <= local_rank < len(cuda_states):
        torch.cuda.set_rng_state(
            cuda_states[local_rank], device=torch.cuda.current_device()
        )
        return

    torch.cuda.set_rng_state_all(cuda_states[:device_count])


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
