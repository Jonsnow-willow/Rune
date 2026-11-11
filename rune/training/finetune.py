from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import ModuleType

from torch import nn

from .train import train


def finetune(
    path: str | Path,
    config: dict,
    model: nn.Module,
    *,
    plugin: ModuleType,
    model_params: dict,
):
    """Start a weights-only stage, or resume its completed-epoch training state.

    The shared trainer validates checkpoint identity and model parameters,
    retains the descriptor scaler, and owns optimization and checkpoint IO.
    """
    config = deepcopy(config)
    runtime = config["runtime"]
    if runtime["resume"]:
        if runtime["resume_model_only"]:
            raise ValueError(
                "Fine-tuning resume restores full training state. Omit "
                "runtime.resume_model_only; use finetune.checkpoint for a new stage."
            )
    else:
        checkpoint = config.get("finetune", {}).get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "Fine-tuning requires [finetune] checkpoint = 'path/to/model.pt', "
                "or --resume to continue an existing run."
            )
        source = Path(checkpoint).expanduser()
        if not source.is_absolute():
            source = Path(path) / source
        if not source.is_file():
            raise FileNotFoundError(source)
        if any((Path(path) / "checkpoints").glob("*.pt")):
            raise FileExistsError(
                "Starting fine-tuning requires a new run directory. "
                "Use --resume to continue its existing checkpoints."
            )
        runtime["resume"] = str(source.resolve())
        runtime["resume_model_only"] = True

    return train(path, config, model, plugin=plugin, model_params=model_params)
