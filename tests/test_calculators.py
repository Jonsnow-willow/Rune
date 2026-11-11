from importlib import import_module
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError


def small_model(name, *, equivariant=False):
    plugin = import_module(f"rune.{name}.plugin")
    values = dict(
        elements=["H"], rcut_radial=3.5, rcut_angular=3.4,
        n_max_radial=1, n_max_angular=1, basis_size_radial=1,
        basis_size_angular=1, l_max=2, l_max_4body=0,
    )
    if name == "nep":
        values["hidden_dims"] = [4]
    else:
        values.update(hidden_dim=8, ffn_dim=12, blocks=1, message_layers=1,
                      message_dim=4, pair_dim=2, message_groups=2, vector_channels=2)
        if equivariant:
            values.update(message_mode="equivariant", equivariant_channels=2)
    return plugin, plugin.build_model(plugin.resolve_params(values))


class CalculatorTests(unittest.TestCase):
    def test_checkpoint_loader_preserves_float64_parameters_and_predictions(self):
        for name, equivariant in (("nep", False), ("mobius", False), ("mobius", True)):
            with self.subTest(model=name, equivariant=equivariant), tempfile.TemporaryDirectory() as directory:
                plugin, model = small_model(name, equivariant=equivariant)
                model = model.double().eval()
                with torch.no_grad():
                    # Include values not exactly representable in float32.
                    for parameter in model.parameters():
                        parameter.add_(1.23456789e-9)
                artifact = Path(directory) / "model.pt"
                torch.save(dict(plugin={"name": plugin.PLUGIN_NAME},
                                model_params=model.params,
                                model_state_dict=model.state_dict()), artifact)
                restored = plugin.load_model(artifact, device="cpu").eval()
                for key, expected in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
                calculator = plugin.build_calculator(artifact)
                atoms = Atoms("H2", positions=[[0, 0, 0], [0.9, 0.1, 0]], cell=[8, 8, 8])
                batch = calculator._build_batch(atoms, compute_forces=True, compute_virial=True)
                with torch.no_grad():
                    expected = model(batch)
                calculator.calculate(atoms, ["energy", "forces", "stress"])
                self.assertEqual(calculator.results["energy"], float(expected["energy"][0]))
                np.testing.assert_array_equal(calculator.results["forces"], expected["forces"].numpy())
                stress = -expected["virial"][0].numpy().reshape(3, 3) / atoms.get_volume()
                np.testing.assert_allclose(calculator.results["stress"][:3], np.diag(stress), rtol=0, atol=0)

    def test_double_precision_periodic_offsets_are_not_rounded_to_float32(self):
        spacing = 3.1234567890123
        atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]], cell=[spacing, 4.2, 5.3], pbc=True)
        for name in ("nep", "mobius"):
            with self.subTest(model=name):
                _, model = small_model(name)
                builder_type = import_module(f"rune.{name}.interface.batch").BatchBuilder
                batch = builder_type(model.params, torch.device("cpu"), torch.float64).build(
                    atoms, compute_forces=True, compute_virial=True,
                )
                for kind in ("radial", "angular"):
                    offsets = batch[f"{kind}_offsets"][batch[f"{kind}_neighbors"] >= 0]
                    self.assertEqual(sorted(offsets[:, 0].tolist()), [-spacing, spacing])

    def test_ase_stress_switch_and_unsupported_elements(self):
        for name in ("nep", "mobius"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                plugin, model = small_model(name)
                artifact = Path(directory) / "model.pt"
                torch.save(dict(plugin={"name": plugin.PLUGIN_NAME}, model_params=model.params,
                                model_state_dict=model.state_dict()), artifact)
                atoms = Atoms("H2", positions=[[0, 0, 0], [0.9, 0, 0]], cell=[8, 8, 8])
                atoms.calc = plugin.build_calculator(artifact, compute_stress=False)
                self.assertTrue(np.isfinite(atoms.get_potential_energy()))
                self.assertEqual(atoms.get_forces().shape, (2, 3))
                with self.assertRaises(PropertyNotImplementedError):
                    atoms.get_stress()
                atoms[0].symbol = "He"
                with self.assertRaisesRegex(ValueError, "not supported"):
                    atoms.get_potential_energy()


if __name__ == "__main__":
    unittest.main()
