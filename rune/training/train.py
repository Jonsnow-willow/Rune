from __future__ import annotations

from pathlib import Path
from types import ModuleType

from torch import nn

from .potential import GradientTrainer


def train(
    path: str | Path,
    config: dict,
    model: nn.Module,
    *,
    plugin: ModuleType,
    model_params: dict,
):
    export_model = getattr(plugin, "export_model", None)
    return GradientTrainer(
        path,
        config,
        model,
        model_params=model_params,
        plugin_identity={
            "name": plugin.PLUGIN_NAME,
            "reference": config["model"]["plugin"],
        },
        ddp_options=plugin.get_ddp_options(model_params),
        export_model=export_model,
    ).fit()
