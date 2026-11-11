from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from ase.io import read

from ..runtime.distributed import DistributedContext
from .dataset import StructureDataset, collate_fn
from .lazy_xyz import LazyDataset, get_index_path


def build_dataloaders(
    path: str | Path,
    config: dict,
    model_params: dict,
    context: DistributedContext | None = None,
):
    options = get_dataloader_options(config, context)

    train_dataset = build_dataset(
        path,
        config,
        model_params,
        split="train",
        context=context,
    )
    train_sampler = _build_sampler(
        train_dataset,
        context=context,
        split="train",
        shuffle=config["data"]["shuffle"],
        seed=config["runtime"]["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=config["data"]["shuffle"] if train_sampler is None else False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **options,
    )

    test_loader = None
    test_path = get_dataset_path(path, config, "test")

    if test_path is not None:
        if not test_path.exists():
            raise FileNotFoundError(
                f"Configured test dataset does not exist: {test_path}"
            )

        test_dataset = build_dataset(
            path,
            config,
            model_params,
            split="test",
            context=context,
        )
        test_sampler = _build_sampler(
            test_dataset,
            context=context,
            split="test",
            shuffle=False,
            seed=config["runtime"]["seed"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            sampler=test_sampler,
            collate_fn=collate_fn,
            **options,
        )

    return train_loader, test_loader


def build_dataset(
    path: str | Path,
    config: dict,
    model_params: dict,
    split: str,
    context: DistributedContext | None = None,
):
    data = config["data"]
    if data["frame_stride"] < 1:
        raise ValueError("frame_stride must be >= 1.")
    if split == "train":
        xyz_paths = get_dataset_path(path, config, "train")
        if not isinstance(xyz_paths, list):
            xyz_paths = [xyz_paths]
        repeats = data.get("train_repeats")
        if repeats is None:
            repeats = [1] * len(xyz_paths)
        if (
            not isinstance(repeats, list)
            or len(repeats) != len(xyz_paths)
            or any(type(count) is not int or count < 1 for count in repeats)
        ):
            raise ValueError(
                "train_repeats must contain one positive integer per train_file."
            )
        max_frames = data["max_train_frames"]
    elif split == "test":
        xyz_path = get_dataset_path(path, config, "test")
        if xyz_path is None:
            raise ValueError("No test file configured.")
        xyz_paths = [xyz_path]
        repeats = [1]
        max_frames = data["max_test_frames"]
    else:
        raise ValueError(f"Unknown split: {split}")

    datasets = []
    for xyz_path, count in zip(xyz_paths, repeats):
        dataset = _build_file_dataset(
            path, config, model_params, split, xyz_path, max_frames, context
        )
        # Repeat references so each selected frame appears exactly count times.
        datasets.extend([dataset] * count)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _build_file_dataset(
    path: str | Path,
    config: dict,
    model_params: dict,
    split: str,
    xyz_path: Path,
    max_frames: int | None,
    context: DistributedContext | None,
):
    data = config["data"]
    if _use_lazy(config, xyz_path):
        return _build_lazy_dataset(
            path,
            config,
            model_params,
            split=split,
            xyz_path=xyz_path,
            max_frames=max_frames,
            context=context,
        )

    frames = read(str(xyz_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if data["frame_stride"] > 1:
        frames = frames[:: data["frame_stride"]]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No frames found in {xyz_path}")
    return StructureDataset(
        frames=frames,
        model_params=model_params,
        require_forces=config["loss"]["forces"] != 0.0,
    )


def get_dataset_path(path: str | Path, config: dict, split: str):
    key = "train_file" if split == "train" else "test_file"
    value = config["data"].get(key)
    if value is None:
        return None
    if split == "train" and isinstance(value, list):
        return [
            Path(item) if Path(item).is_absolute() else Path(path) / item
            for item in value
        ]
    data_path = Path(value)
    return data_path if data_path.is_absolute() else Path(path) / data_path


def _use_lazy(config: dict, xyz_path) -> bool:
    data_format = config["data"]["data_format"]
    if data_format == "eager":
        return False
    if data_format == "lazy_xyz":
        return True
    size_mb = xyz_path.stat().st_size / (1024 * 1024)
    return size_mb >= config["data"]["lazy_threshold_mb"]


def _build_lazy_dataset(
    path: str | Path,
    config: dict,
    model_params: dict,
    split: str,
    xyz_path,
    max_frames: int | None,
    context: DistributedContext | None,
):
    index_path = get_index_path(
        Path(path),
        config["data"]["index_dir"],
        xyz_path,
        split=split,
    )
    kwargs = {
        "xyz_path": xyz_path,
        "model_params": model_params,
        "index_path": index_path,
        "cache_index": config["data"]["cache_index"],
        "frame_stride": config["data"]["frame_stride"],
        "max_frames": max_frames,
        "require_forces": config["loss"]["forces"] != 0.0,
    }
    if context is not None and context.is_distributed:
        dataset = None
        if context.is_main_process:
            dataset = LazyDataset(**kwargs)
        context.barrier()
        if dataset is None:
            dataset = LazyDataset(**kwargs)
        context.barrier()
        return dataset
    return LazyDataset(**kwargs)


def get_dataloader_options(
    config: dict, context: DistributedContext | None = None
) -> dict:
    data = config["data"]
    num_workers = data["num_workers"]
    pin_memory_default = context is not None and context.device.type == "cuda"
    pin_memory = _parse_bool(data["pin_memory"], default=pin_memory_default)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = _parse_bool(
            data["persistent_workers"], default=True
        )
        if data["prefetch_factor"] is not None:
            if data["prefetch_factor"] < 1:
                raise ValueError("prefetch_factor must be >= 1.")
            kwargs["prefetch_factor"] = data["prefetch_factor"]
    return kwargs


def _parse_bool(value: bool | str, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value == "auto":
        return default
    raise ValueError(f"Expected boolean or auto, got {value!r}.")


def _build_sampler(
    dataset,
    context: DistributedContext | None,
    split: str,
    shuffle: bool,
    seed: int,
):
    if context is None or not context.is_distributed:
        return None

    if split == "train":
        return DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    if split == "test":
        return DistributedEvalSampler(
            dataset,
            rank=context.rank,
            world_size=context.world_size,
        )

    raise ValueError(f"Unknown dataset split: {split}")


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, *, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid rank/world_size for DistributedEvalSampler.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return (
            0
            if remaining <= 0
            else (remaining + self.world_size - 1) // self.world_size
        )
