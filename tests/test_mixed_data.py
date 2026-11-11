from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from rune.datasets.data import build_dataloaders, build_dataset, get_dataset_path
from rune.datasets.dataset import StructureDataset
from rune.datasets.lazy_xyz import LazyDataset
from rune.runtime.distributed import DistributedContext


class MixedDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.first = self.write_xyz("first/train.xyz", range(6))
        self.second = self.write_xyz("second/train.xyz", range(100, 106))
        self.params = {
            "elements": ["H"],
            "rcut_radial": 3.0,
            "rcut_angular": 3.0,
            "max_neighbors_radial": 8,
            "max_neighbors_angular": 8,
        }

    def write_xyz(self, filename, energies):
        path = self.path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                "1\n"
                f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} '
                'Lattice="10 0 0 0 10 0 0 0 10" pbc="F F F"\nH 0 0 0 0 0 0\n'
                for energy in energies
            )
        )
        return path

    def config(self, *, data_format="lazy_xyz", **changes):
        data = {
            "train_file": ["first/train.xyz", str(self.second)],
            "test_file": None,
            "data_format": data_format,
            "frame_stride": 1,
            "max_train_frames": None,
            "max_test_frames": None,
            "index_dir": ".rune_index",
            "cache_index": True,
            "batch_size": 3,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
        }
        data.update(changes)
        return {"data": data, "loss": {"forces": 1.0}, "runtime": {"seed": 42}}

    @staticmethod
    def energies(dataset):
        return [float(dataset[index]["energy"]) for index in range(len(dataset))]

    def test_exact_repeats_apply_after_each_source_selection(self):
        for data_format in ("lazy_xyz", "eager"):
            with self.subTest(data_format=data_format):
                config = self.config(
                    data_format=data_format,
                    train_repeats=[1, 8],
                    frame_stride=2,
                    max_train_frames=2,
                )
                dataset = build_dataset(self.path, config, self.params, "train")
                self.assertIsInstance(dataset, ConcatDataset)
                self.assertEqual(self.energies(dataset), [0, 2] + [100, 102] * 8)
                self.assertEqual(len(dataset), 18)
                self.assertTrue(
                    all(part is dataset.datasets[1] for part in dataset.datasets[1:])
                )

    def test_lazy_sources_have_distinct_indices_with_same_basename(self):
        config = self.config()
        dataset = build_dataset(self.path, config, self.params, "train")
        first, second = dataset.datasets
        self.assertIsInstance(first, LazyDataset)
        self.assertIsInstance(second, LazyDataset)
        self.assertNotEqual(first.index_path, second.index_path)
        self.assertTrue(first.index_path.is_file())
        self.assertTrue(second.index_path.is_file())
        self.assertEqual(self.energies(dataset), list(range(6)) + list(range(100, 106)))
        reloaded = build_dataset(self.path, config, self.params, "train")
        self.assertEqual(self.energies(reloaded), self.energies(dataset))

    def test_single_file_and_single_item_list_preserve_dataset_type(self):
        for data_format, expected_type in (
            ("lazy_xyz", LazyDataset),
            ("eager", StructureDataset),
        ):
            for train_file in ("first/train.xyz", ["first/train.xyz"]):
                with self.subTest(data_format=data_format, train_file=train_file):
                    config = self.config(data_format=data_format, train_file=train_file)
                    dataset = build_dataset(self.path, config, self.params, "train")
                    self.assertIsInstance(dataset, expected_type)
                    self.assertEqual(self.energies(dataset), list(range(6)))

    def test_shuffle_visits_every_virtual_frame_and_test_is_not_repeated(self):
        config = self.config(
            train_repeats=[1, 8],
            frame_stride=2,
            max_train_frames=2,
            test_file="first/train.xyz",
            max_test_frames=1,
        )
        train_loader, test_loader = build_dataloaders(self.path, config, self.params)
        observed = Counter(
            float(energy) for batch in train_loader for energy in batch["energy"]
        )
        self.assertEqual(observed, Counter({0.0: 1, 2.0: 1, 100.0: 8, 102.0: 8}))
        self.assertEqual(len(test_loader.dataset), 1)
        self.assertIsInstance(test_loader.dataset, LazyDataset)

    def test_distributed_loader_keeps_existing_sampler_and_exact_coverage(self):
        config = self.config(train_repeats=[1, 8], max_train_frames=2)
        observed = Counter()
        for rank in range(2):
            context = DistributedContext(rank, rank, 2, "gloo", torch.device("cpu"))
            loader, _ = build_dataloaders(self.path, config, self.params, context)
            self.assertIsInstance(loader.sampler, DistributedSampler)
            loader.sampler.set_epoch(3)
            observed.update(
                float(loader.dataset[index]["energy"]) for index in loader.sampler
            )
        self.assertEqual(observed, Counter({0.0: 1, 1.0: 1, 100.0: 8, 101.0: 8}))

    def test_paths_resolve_individually(self):
        config = self.config()
        self.assertEqual(
            get_dataset_path(self.path, config, "train"), [self.first, self.second]
        )
        config["data"]["train_file"] = "first/train.xyz"
        self.assertEqual(get_dataset_path(self.path, config, "train"), self.first)
        self.assertIsNone(get_dataset_path(self.path, config, "test"))

    def test_invalid_repeat_counts_are_rejected(self):
        for repeats in ([1], [0, 1], [1, -1], [1, True], [1, 1.5], 2):
            with self.subTest(repeats=repeats):
                with self.assertRaisesRegex(ValueError, "train_repeats"):
                    build_dataset(
                        self.path,
                        self.config(train_repeats=repeats),
                        self.params,
                        "train",
                    )


if __name__ == "__main__":
    unittest.main()
