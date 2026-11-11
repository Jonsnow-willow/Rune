"""Configured seeds cover model initialization as well as training randomness."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from rune.cli.run import main
from rune.runtime.checkpoint import read_checkpoint
from rune.training.potential import GradientTrainer


class InitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_run(self, name, *, variant="local", seed=73):
        path = self.root / name
        path.mkdir()
        message = "message_layers = 0"
        if variant == "scalar":
            message = "message_layers = 2"
        elif variant == "equivariant":
            message = """message_layers = 2
message_mode = "equivariant"
equivariant_channels = 2
equivariant_l_max = 2"""
        config = f"""[model]
plugin = "rune.mobius"
[model.params]
elements = ["H"]
hidden_dim = 8
blocks = 1
ffn_dim = 16
message_dim = 4
pair_dim = 4
message_groups = 2
vector_channels = 2
{message}
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
epochs = 1
[runtime]
device = "cpu"
seed = {seed}
"""
        (path / "rune.toml").write_text(config)
        frames = []
        for distance in (0.7, 0.85, 1.05):
            atoms = Atoms(
                "H2", positions=[[0, 0, 0], [distance, 0.1, 0.0]],
                cell=[10, 10, 10],
            )
            atoms.calc = SinglePointCalculator(
                atoms, energy=distance**2, forces=np.zeros((2, 3)),
                stress=np.zeros(6),
            )
            frames.append(atoms)
        write(path / "train.xyz", frames, format="extxyz")
        return path

    def run_cli(self, path, *, ambient_seed):
        random.seed(ambient_seed)
        np.random.seed(ambient_seed)
        torch.manual_seed(ambient_seed)
        with redirect_stdout(io.StringIO()):
            main(["train", str(path)])

    def initial_state(self, path, *, ambient_seed):
        captured = {}

        def capture(trainer):
            captured.update(
                (name, tensor.detach().clone())
                for name, tensor in trainer.model.state_dict().items()
            )
            return trainer.model

        # Configuration, plugin construction, dataloaders and trainer are real.
        # Stop before scaler initialization or optimizer steps alter the weights.
        with patch.object(GradientTrainer, "fit", autospec=True, side_effect=capture):
            self.run_cli(path, ambient_seed=ambient_seed)
        self.assertTrue(captured)
        return captured

    def assert_same_weights(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)

    def test_runtime_seed_controls_initial_weights_for_every_variant(self):
        for variant in ("local", "scalar", "equivariant"):
            with self.subTest(variant=variant):
                path = self.make_run(variant, variant=variant)
                first = self.initial_state(path, ambient_seed=11)
                repeated = self.initial_state(path, ambient_seed=999)
                self.assert_same_weights(first, repeated)
                changed_path = self.make_run(f"{variant}-different", variant=variant, seed=74)
                changed = self.initial_state(changed_path, ambient_seed=11)
                self.assertFalse(torch.equal(
                    first["descriptor.radial.c_table"],
                    changed["descriptor.radial.c_table"],
                ))

    def test_variants_share_all_common_initial_parameters(self):
        states = [
            self.initial_state(
                self.make_run(variant, variant=variant), ambient_seed=100 + index
            )
            for index, variant in enumerate(("local", "scalar", "equivariant"))
        ]
        # Local contains the descriptor, trunk, normalization and energy head
        # shared with both message-passing variants.
        for state in states[1:]:
            for name, value in states[0].items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0, msg=name)

    def test_one_epoch_cpu_training_is_reproducible(self):
        for variant in ("local", "scalar", "equivariant"):
            with self.subTest(variant=variant):
                first_path = self.make_run(f"{variant}-first", variant=variant)
                second_path = self.make_run(f"{variant}-second", variant=variant)
                self.run_cli(first_path, ambient_seed=11)
                self.run_cli(second_path, ambient_seed=999)
                first = read_checkpoint(first_path / "checkpoints" / "last.pt")
                second = read_checkpoint(second_path / "checkpoints" / "last.pt")
                self.assertEqual(first["epoch"], 1)
                self.assert_same_weights(first["model_state_dict"], second["model_state_dict"])
                self.assertEqual(first["metrics"]["train_loss"], second["metrics"]["train_loss"])
                self.assertTrue(torch.equal(
                    first["rng_state"]["torch"], second["rng_state"]["torch"]
                ))


if __name__ == "__main__":
    unittest.main()
