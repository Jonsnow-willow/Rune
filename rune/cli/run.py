from __future__ import annotations

import argparse
import copy
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
from ..training.finetune import finetune
from ..training.train import train
from ..runtime.checkpoint import read_checkpoint
from ..runtime.random import seed_all


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

_SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "data": {
        "train_file": "train.xyz",
        "train_repeats": None,
        "test_file": None,
        "data_format": "auto",
        "batch_size": 5,
        "num_workers": 0,
        "pin_memory": "auto",
        "persistent_workers": "auto",
        "prefetch_factor": 2,
        "shuffle": True,
        "lazy_threshold_mb": 512.0,
        "max_train_frames": None,
        "max_test_frames": None,
        "frame_stride": 1,
        "index_dir": ".rune_index",
        "cache_index": True,
    },
    "optimizer": {
        "name": "adamw",
        "epochs": 500,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
    },
    "scheduler": {
        "name": "none",
        "min_learning_rate": 0.0,
        "step_size": 100,
        "gamma": 0.5,
        "patience": 20,
        "factor": 0.5,
    },
    "loss": {"energy": 1.0, "forces": 1.0, "virial": 0.1},
    "finetune": {"checkpoint": None},
    "runtime": {
        "device": "auto",
        "seed": 42,
        "save_every": 1,
        "keep_every": 0,
        "export_every": 0,
        "resume": None,
        "resume_model_only": False,
        "resume_reset_best": False,
        "progress_log_interval": 0,
        "compute_descriptor_scaler_once": True,
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": None,
        "descriptor_scaler_max_batches": None,
    },
}


def load_config(path: str | Path = ".") -> dict[str, Any]:
    """Read the rune.toml in path and return its complete run configuration."""

    config_path = Path(path).expanduser().resolve() / "rune.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    if type(raw) is not dict:
        raise TypeError(f"{config_path} must contain a TOML table.")

    allowed_sections = {"model", *_SECTION_DEFAULTS}
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown rune.toml section(s): {', '.join(unknown_sections)}")

    model_section = raw.get("model")
    if not isinstance(model_section, Mapping):
        raise ValueError("rune.toml must define a [model] table.")
    unknown_model_keys = sorted(set(model_section) - {"plugin", "params"})
    if unknown_model_keys:
        raise ValueError(f"Unknown [model] key(s): {', '.join(unknown_model_keys)}")
    reference = model_section.get("plugin")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('[model] must define plugin = "package.name".')
    params = model_section.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("[model.params] must be a TOML table.")

    config: dict[str, Any] = {
        "model": {
            "plugin": reference.strip(),
            "params": dict(params),
        }
    }
    for name, defaults in _SECTION_DEFAULTS.items():
        section = raw.get(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"[{name}] must be a TOML table.")
        unknown = sorted(set(section) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown [{name}] key(s): {', '.join(unknown)}")
        config[name] = {**copy.deepcopy(defaults), **dict(section)}

    _check_config(config)
    return config


def load_plugin(reference: str) -> ModuleType:
    """Import a model plugin and check the small training contract."""

    plugin = import_module(f"{reference}.plugin")
    plugin_name = getattr(plugin, "PLUGIN_NAME", None)
    if not isinstance(plugin_name, str) or not plugin_name.strip():
        raise TypeError(
            f"Model plugin {reference!r} must define a non-empty PLUGIN_NAME."
        )
    for action in ("resolve_params", "build_model", "get_ddp_options"):
        if not callable(getattr(plugin, action, None)):
            raise TypeError(
                f"Model plugin {reference!r} must define callable {action}()."
            )
    return plugin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rune",
        description="RUNE multi-model atomistic training tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command, help_text in (
        ("train", "Train a run directory containing rune.toml."),
        ("finetune", "Fine-tune from [finetune].checkpoint; --resume continues a run."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("path", nargs="?", default=Path("."), type=Path)
        command_parser.add_argument("--resume", type=str, default=None)
        command_parser.add_argument("--device", type=str, default=None)
        command_parser.add_argument("--epochs", type=int, default=None)

    report_parser = subparsers.add_parser("report", help="Report checkpoint metadata.")
    report_parser.add_argument("path", nargs="?", default=Path("."), type=Path)
    report_parser.add_argument("--checkpoint", default="checkpoints/best.pt")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["train"])

    path = args.path.expanduser().resolve()
    if args.command in {"train", "finetune"}:
        config = load_config(path)
        if args.device is not None:
            config["runtime"]["device"] = args.device
        if args.resume is not None:
            config["runtime"]["resume"] = args.resume
        if args.epochs is not None:
            config["optimizer"]["epochs"] = args.epochs
        _check_config(config)

        plugin = load_plugin(config["model"]["plugin"])
        params = plugin.resolve_params(config["model"]["params"])
        seed_all(config["runtime"]["seed"])
        model = plugin.build_model(params)
        action = finetune if args.command == "finetune" else train
        action(path, config, model, plugin=plugin, model_params=params)
        return
    if args.command == "report":
        report_checkpoint(path, args.checkpoint)
        return
    parser.error(f"Unknown command: {args.command}")


def report_checkpoint(path: str | Path, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(path) / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
    print(f"Checkpoint: {checkpoint_path}")
    plugin = checkpoint.get("plugin", {})
    print(f"  plugin: {plugin.get('name')} ({plugin.get('reference')})")
    print(f"  epoch: {checkpoint.get('epoch')}")
    print(f"  best_metric: {checkpoint.get('best_metric')}")
    print(f"  has_optimizer: {checkpoint.get('optimizer_state_dict') is not None}")
    print(f"  has_scheduler: {checkpoint.get('scheduler_state_dict') is not None}")
    print(f"  has_rng_state: {checkpoint.get('rng_state') is not None}")
    if checkpoint.get("metrics"):
        print(f"  metrics: {checkpoint['metrics']}")


def _check_config(config: Mapping[str, Any]) -> None:
    data = config["data"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    runtime = config["runtime"]

    train_files = data["train_file"]
    if isinstance(train_files, str):
        train_files = [train_files]
    if not isinstance(train_files, list) or not train_files or any(
        not isinstance(value, str) or not value.strip() for value in train_files
    ):
        raise ValueError("data.train_file must be a path or a nonempty list of paths.")
    repeats = data["train_repeats"]
    if repeats is not None and (
        not isinstance(repeats, list)
        or len(repeats) != len(train_files)
        or any(type(value) is not int or value < 1 for value in repeats)
    ):
        raise ValueError("data.train_repeats must contain one positive integer per train_file.")
    for label, value in (
        ("data.test_file", data["test_file"]),
        ("finetune.checkpoint", config["finetune"]["checkpoint"]),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{label} must be a path or omitted.")
    _check_positive_int(data, "batch_size")
    _check_nonnegative_int(data, "num_workers")
    _check_positive_int(data, "frame_stride")
    if data["data_format"] not in {"auto", "eager", "lazy_xyz"}:
        raise ValueError("data.data_format must be auto, eager, or lazy_xyz.")
    for name in ("pin_memory", "persistent_workers"):
        if data[name] not in {True, False, "auto"}:
            raise ValueError(f'data.{name} must be true, false, or "auto".')
    if data["prefetch_factor"] is not None and data["prefetch_factor"] < 1:
        raise ValueError("data.prefetch_factor must be >= 1 or omitted.")
    if data["lazy_threshold_mb"] < 0:
        raise ValueError("data.lazy_threshold_mb must be nonnegative.")

    optimizer["name"] = str(optimizer["name"]).lower()
    if optimizer["name"] != "adamw":
        raise ValueError("optimizer.name must currently be adamw.")
    _check_positive_int(optimizer, "epochs")
    for name in ("learning_rate", "eps"):
        if optimizer[name] <= 0:
            raise ValueError(f"optimizer.{name} must be positive.")
    if optimizer["weight_decay"] < 0:
        raise ValueError("optimizer.weight_decay must be nonnegative.")

    scheduler["name"] = str(scheduler["name"]).lower()
    if scheduler["name"] not in {"none", "cosine", "step", "plateau"}:
        raise ValueError("scheduler.name must be none, cosine, step, or plateau.")
    for name, value in loss.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"loss.{name} must be nonnegative.")
    _check_positive_int(runtime, "gradient_accumulation_steps")
    _check_nonnegative_int(runtime, "save_every")
    _check_nonnegative_int(runtime, "keep_every")
    _check_nonnegative_int(runtime, "export_every")
    _check_nonnegative_int(runtime, "progress_log_interval")
    if runtime["descriptor_scaler_max_batches"] is not None:
        _check_positive_int(runtime, "descriptor_scaler_max_batches")


def _check_positive_int(section: Mapping[str, Any], name: str) -> None:
    value = section[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _check_nonnegative_int(section: Mapping[str, Any], name: str) -> None:
    value = section[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


if __name__ == "__main__":
    main()
