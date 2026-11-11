from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from rune.cli.run import load_config, main
from rune.runtime.checkpoint import Checkpoint, read_checkpoint


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_run(self, name, *, epochs=5, keep_every=None, save_every=1):
        path = self.root / name
        path.mkdir()
        config = f"""\
[model]
plugin = "rune.nep"
[model.params]
elements = ["H"]
hidden_dims = [4]
rcut_radial = 3.0
rcut_angular = 3.0
n_max_radial = 1
n_max_angular = 1
basis_size_radial = 1
basis_size_angular = 1
l_max = 2
l_max_4body = 0
[data]
batch_size = 1
[optimizer]
epochs = {epochs}
[scheduler]
name = "step"
step_size = 2
gamma = 0.5
[runtime]
device = "cpu"
save_every = {save_every}
"""
        if keep_every is not None:
            config += f"keep_every = {keep_every}\n"
        (path / "rune.toml").write_text(config)
        frames = []
        for distance in (0.7, 0.85, 1.05):
            atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]])
            atoms.calc = SinglePointCalculator(
                atoms, energy=distance**2, forces=np.zeros((2, 3))
            )
            frames.append(atoms)
        write(path / "train.xyz", frames, format="extxyz")
        return path

    def train(self, path, *args):
        torch.manual_seed(123)
        with redirect_stdout(io.StringIO()):
            main(["train", str(path), *args])

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_configuration_accepts_only_nonnegative_integers(self):
        default = self.make_run("default")
        self.assertEqual(load_config(default)["runtime"]["keep_every"], 0)
        for value in ("0", "3", "-1", "true", "false", "1.5", '"5"'):
            with self.subTest(value=value):
                path = self.make_run(f"config-{value}", keep_every=value)
                if value in ("0", "3"):
                    self.assertEqual(
                        load_config(path)["runtime"]["keep_every"], int(value)
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "keep_every"):
                        load_config(path)

    def test_intervals_final_epoch_and_default_compatibility(self):
        for epochs, keep_every, save_every, retained in (
            (3, None, 1, []),
            (5, 2, 0, [2, 4, 5]),
            (4, 2, 3, [2, 4]),
            (3, 10, 1, [3]),
        ):
            with self.subTest(epochs=epochs, keep_every=keep_every):
                path = self.make_run(
                    f"run-{epochs}-{keep_every}",
                    epochs=epochs,
                    keep_every=keep_every,
                    save_every=save_every,
                )
                self.train(path)
                expected = {"best.pt", "last.pt"} | {
                    f"epoch-{epoch:04d}.pt" for epoch in retained
                }
                self.assertEqual(
                    {file.name for file in (path / "checkpoints").iterdir()}, expected
                )
                for epoch in retained:
                    checkpoint = read_checkpoint(
                        path / "checkpoints" / f"epoch-{epoch:04d}.pt"
                    )
                    self.assertEqual(checkpoint["epoch"], epoch)
                    self.assertEqual(checkpoint["metrics"]["epoch"], epoch)
                    self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
                    self.assertIsNotNone(checkpoint["scheduler_state_dict"])
                    self.assertIn("torch", checkpoint["rng_state"])
                final = read_checkpoint(path / "checkpoints" / "last.pt")
                self.assertEqual(final["epoch"], epochs)
                if retained:
                    self.assert_state_equal(checkpoint, final)

    def test_resume_from_history_matches_uninterrupted_training(self):
        original = self.make_run("original", keep_every=2)
        self.train(original)
        historical_path = original / "checkpoints" / "epoch-0002.pt"
        historical_bytes = historical_path.read_bytes()
        branch = self.make_run("branch", keep_every=2)
        self.train(branch, "--resume", str(historical_path))
        self.assertEqual(
            sorted(file.name for file in (branch / "checkpoints").glob("epoch-*.pt")),
            ["epoch-0004.pt", "epoch-0005.pt"],
        )
        self.assertEqual(historical_path.read_bytes(), historical_bytes)
        uninterrupted = read_checkpoint(original / "checkpoints" / "last.pt")
        resumed = read_checkpoint(branch / "checkpoints" / "last.pt")
        for key in (
            "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
            "epoch", "best_metric", "plugin", "model_params", "rng_state",
        ):
            with self.subTest(key=key):
                self.assert_state_equal(uninterrupted[key], resumed[key])

    def test_rewinding_does_not_overwrite_history_or_rolling_checkpoints(self):
        path = self.make_run("rewind", epochs=3, keep_every=2)
        self.train(path)
        before = {
            file: file.read_bytes() for file in (path / "checkpoints").iterdir()
        }
        with self.assertRaisesRegex(FileExistsError, "new run directory"):
            self.train(path, "--resume", "checkpoints/epoch-0002.pt")
        for file, data in before.items():
            self.assertEqual(file.read_bytes(), data)

    def test_completed_resume_keeps_checkpoint_metrics_and_artifacts(self):
        path = self.make_run("completed", epochs=2)
        self.train(path)
        before = {
            file: file.read_bytes()
            for directory in ("checkpoints", "exports", "logs")
            for file in (path / directory).rglob("*")
            if file.is_file()
        }
        self.train(path, "--resume", "checkpoints/last.pt")
        for file, data in before.items():
            self.assertEqual(file.read_bytes(), data, str(file))

    def test_fresh_training_refuses_an_existing_run(self):
        path = self.make_run("existing", epochs=1)
        self.train(path)
        before = {file: file.read_bytes() for file in path.rglob("*") if file.is_file()}
        with self.assertRaisesRegex(FileExistsError, "--resume"):
            self.train(path)
        for file, data in before.items():
            self.assertEqual(file.read_bytes(), data, str(file))

    def test_full_resume_rejects_missing_training_state(self):
        source = self.make_run("source", epochs=1)
        self.train(source)
        payload = read_checkpoint(source / "checkpoints" / "last.pt")
        for key in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
            with self.subTest(key=key):
                incomplete = dict(payload)
                incomplete.pop(key)
                artifact = self.root / f"missing-{key}.pt"
                torch.save(incomplete, artifact)
                target = self.make_run(key, epochs=2)
                with self.assertRaisesRegex(ValueError, key):
                    self.train(target, "--resume", str(artifact))
                self.assertFalse((target / "checkpoints" / "last.pt").exists())

    def test_full_resume_rejects_changed_scheduler(self):
        source = self.make_run("scheduler-source", epochs=1)
        self.train(source)
        for name in ("none", "cosine"):
            with self.subTest(scheduler=name):
                target = self.make_run(f"scheduler-{name}", epochs=2)
                config = target / "rune.toml"
                config.write_text(config.read_text().replace('name = "step"', f'name = "{name}"'))
                with self.assertRaisesRegex(ValueError, "scheduler"):
                    self.train(target, "--resume", str(source / "checkpoints" / "last.pt"))


class AtomicCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoints = Checkpoint(
            temporary.name, {}, {"name": "test", "reference": "test"}, {}
        )
        self.model = torch.nn.Linear(1, 1)

    def save(self, name="last.pt", epoch=1, *, overwrite=True):
        return self.checkpoints.save(
            name, self.model, None, epoch, 0.0, overwrite=overwrite
        )

    def test_rolling_checkpoint_can_be_replaced(self):
        path = self.save()
        self.save(epoch=2)
        self.assertEqual(read_checkpoint(path)["epoch"], 2)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_partial_serialization_preserves_previous_checkpoint(self):
        path = self.save()
        before = path.read_bytes()

        def fail_after_writing(payload, stream):
            stream.write(b"partial checkpoint")
            raise OSError("simulated full disk")

        for name, overwrite in (("last.pt", True), ("epoch-0002.pt", False)):
            with self.subTest(name=name):
                with patch("rune.runtime.checkpoint.torch.save", fail_after_writing):
                    with self.assertRaisesRegex(OSError, "full disk"):
                        self.save(name, epoch=2, overwrite=overwrite)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_publication_preserves_previous_checkpoint(self):
        path = self.save()
        before = path.read_bytes()
        with patch("rune.runtime.checkpoint.os.replace", side_effect=OSError("publish")):
            with self.assertRaisesRegex(OSError, "publish"):
                self.save(epoch=2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_history_collision_during_serialization_is_protected(self):
        path = self.checkpoints.path("epoch-0001.pt")
        original_save = torch.save

        def concurrent_save(payload, stream):
            original_save(payload, stream)
            path.write_bytes(b"checkpoint created by another writer")

        with patch("rune.runtime.checkpoint.torch.save", concurrent_save):
            with self.assertRaises(FileExistsError):
                self.save(path.name, overwrite=False)
        self.assertEqual(path.read_bytes(), b"checkpoint created by another writer")
        self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
