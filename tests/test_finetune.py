from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from rune.cli.run import load_config, load_plugin, main
from rune.runtime.checkpoint import read_checkpoint
from rune.training.finetune import finetune
from rune.training.potential import GradientTrainer


class FinetuneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_run(self, name, model="nep", *, checkpoint=None, epochs=1):
        path = self.root / name
        path.mkdir()
        network = 'hidden_dims = [4]' if model == "nep" else '''hidden_dim = 8
blocks = 1
ffn_dim = 16
message_layers = 1
message_dim = 4
pair_dim = 4
message_groups = 2
vector_channels = 2'''
        config = f'''[model]
plugin = "rune.{model}"
[model.params]
elements = ["H"]
{network}
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
learning_rate = {0.0001 if checkpoint else 0.001}
[scheduler]
name = "{ "none" if checkpoint else "step" }"
step_size = 1
gamma = 0.5
[runtime]
device = "cpu"
seed = 73
gradient_accumulation_steps = 2
'''
        if checkpoint:
            config += f'\n[finetune]\ncheckpoint = "{checkpoint}"\n'
        (path / "rune.toml").write_text(config)
        frames = []
        for distance in (0.7, 0.85, 1.05):
            atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[10, 10, 10])
            atoms.calc = SinglePointCalculator(
                atoms, energy=distance**2, forces=np.zeros((2, 3)), stress=np.zeros(6)
            )
            frames.append(atoms)
        write(path / "train.xyz", frames, format="extxyz")
        return path

    def run_cli(self, *args):
        with redirect_stdout(io.StringIO()):
            main([str(arg) for arg in args])

    def test_weights_only_initialization_and_epoch_resume_for_both_models(self):
        for model in ("nep", "mobius"):
            with self.subTest(model=model):
                source = self.make_run(f"source-{model}", model, epochs=2)
                self.run_cli("train", source)
                checkpoint_path = source / "checkpoints" / "last.pt"
                original_bytes = checkpoint_path.read_bytes()
                original = read_checkpoint(checkpoint_path)
                self.assertTrue(original["optimizer_state_dict"]["state"])
                target = self.make_run(f"fine-{model}", model, checkpoint=checkpoint_path)
                config = load_config(target)
                before = copy.deepcopy(config)
                plugin = load_plugin(config["model"]["plugin"])
                params = plugin.resolve_params(config["model"]["params"])

                def inspect_initial(trainer):
                    self.assertEqual(trainer.current_epoch, 0)
                    self.assertEqual(trainer.best_metric, float("inf"))
                    self.assertFalse(trainer.optimizer.state)
                    self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.0001)
                    self.assertIsNone(trainer.scheduler)
                    self.assertTrue(trainer.q_scaler_initialized)
                    for name, value in trainer.model.state_dict().items():
                        torch.testing.assert_close(value, original["model_state_dict"][name], rtol=0, atol=0)
                    expected_rng = torch.Generator().manual_seed(73).get_state()
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    return trainer.model

                with patch.object(GradientTrainer, "fit", autospec=True, side_effect=inspect_initial):
                    with redirect_stdout(io.StringIO()):
                        finetune(target, config, plugin.build_model(params), plugin=plugin, model_params=params)
                self.assertEqual(config, before)
                self.run_cli("finetune", target)
                trained = read_checkpoint(target / "checkpoints" / "last.pt")
                self.assertEqual(trained["epoch"], 1)
                self.assertTrue(trained["run_config"]["runtime"]["resume_model_only"])
                self.assertEqual({int(state["step"]) for state in trained["optimizer_state_dict"]["state"].values()}, {2})
                torch.testing.assert_close(trained["model_state_dict"]["q_scaler"], original["model_state_dict"]["q_scaler"], rtol=0, atol=0)

                def inspect_resume(trainer):
                    self.assertEqual(trainer.current_epoch, 1)
                    self.assertEqual(trainer.best_metric, trained["best_metric"])
                    self.assertEqual({int(state["step"]) for state in trainer.optimizer.state.values()}, {2})
                    self.assertTrue(torch.equal(torch.get_rng_state(), trained["rng_state"]["torch"]))
                    return trainer.model

                with patch.object(GradientTrainer, "fit", autospec=True, side_effect=inspect_resume):
                    self.run_cli("finetune", target, "--resume", "checkpoints/last.pt", "--epochs", "2")
                self.run_cli("finetune", target, "--resume", "checkpoints/last.pt", "--epochs", "2")
                resumed = read_checkpoint(target / "checkpoints" / "last.pt")
                self.assertEqual(resumed["epoch"], 2)
                self.assertFalse(resumed["run_config"]["runtime"]["resume_model_only"])
                self.assertEqual({int(state["step"]) for state in resumed["optimizer_state_dict"]["state"].values()}, {4})
                self.assertEqual(checkpoint_path.read_bytes(), original_bytes)
                if model == "nep":
                    exported = plugin.load_model(target / "exports" / "nep.txt")
                    actual = plugin.load_model(target / "checkpoints" / "last.pt")
                    for name, value in actual.state_dict().items():
                        torch.testing.assert_close(exported.state_dict()[name], value)
                with self.assertRaisesRegex(FileExistsError, "new run directory"):
                    self.run_cli("finetune", target)

    def test_missing_source_unknown_fields_and_invalid_resume_mode(self):
        path = self.make_run("missing")
        with self.assertRaisesRegex(ValueError, "Fine-tuning requires"):
            self.run_cli("finetune", path)
        config_file = path / "rune.toml"
        original = config_file.read_text()
        config_file.write_text(original + '\n[finetune]\ncheckpoint = "missing.pt"\n')
        with self.assertRaises(FileNotFoundError):
            self.run_cli("finetune", path)
        for extra in ('\n[finetune]\ncheckpoint = 1\n', '\n[finetune]\ncheckpiont = "x.pt"\n'):
            config_file.write_text(original + extra)
            with self.assertRaises(ValueError):
                load_config(path)
        config_file.write_text(original + 'resume = "any.pt"\nresume_model_only = true\n')
        with self.assertRaisesRegex(ValueError, "restores full training state"):
            self.run_cli("finetune", path)

    def test_changed_model_parameters_are_rejected(self):
        source = self.make_run("source")
        self.run_cli("train", source)
        target = self.make_run("changed", checkpoint=source / "checkpoints" / "last.pt")
        config_file = target / "rune.toml"
        config_file.write_text(config_file.read_text().replace('rcut_radial = 3.0', 'rcut_radial = 3.5'))
        with self.assertRaisesRegex(ValueError, "model parameters differ"):
            self.run_cli("finetune", target)

    def test_inference_checkpoint_can_initialize_a_new_finetuning_stage(self):
        source = self.make_run("source")
        self.run_cli("train", source)
        full = read_checkpoint(source / "checkpoints" / "last.pt")
        artifact = self.root / "inference.pt"
        torch.save(
            {key: full[key] for key in ("plugin", "model_params", "model_state_dict")},
            artifact,
        )
        target = self.make_run("from-inference", checkpoint=artifact)
        self.run_cli("finetune", target)
        trained = read_checkpoint(target / "checkpoints" / "last.pt")
        self.assertEqual(trained["epoch"], 1)
        self.assertTrue(trained["optimizer_state_dict"]["state"])


if __name__ == "__main__":
    unittest.main()
