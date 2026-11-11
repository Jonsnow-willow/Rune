"""Geometry and training checks for persistent Möbius direction states.

Run on a GPU with ``RUNE_TEST_DEVICE=cuda python -m unittest discover -s tests
-p test_mobius_equivariant.py``. The file can also be collected by pytest.
The hand-built neighbor graph stays fixed under differentiation and includes
periodic self images, so cell derivatives cannot be inferred from atom forces.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch

from rune.mobius.model import Mobius
from rune.mobius.plugin import PLUGIN_NAME, load_model, resolve_params


def _device():
    selected = torch.device(os.environ.get("RUNE_TEST_DEVICE", "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise unittest.SkipTest(
            "RUNE_TEST_DEVICE requests CUDA, but CUDA is unavailable."
        )
    return selected


def _params(**changes):
    values = {
        "elements": ["Fe", "C"],
        "hidden_dim": 8,
        "blocks": 1,
        "ffn_dim": 12,
        "message_layers": 2,
        "message_mode": "equivariant",
        "message_dim": 8,
        "message_groups": 2,
        "pair_dim": 3,
        "vector_channels": 2,
        "equivariant_channels": 3,
        "equivariant_l_max": None,
        "message_basis_size": 3,
        "rcut_radial": 4.8,
        "rcut_angular": 4.4,
        "n_max_radial": 1,
        "n_max_angular": 1,
        "basis_size_radial": 2,
        "basis_size_angular": 2,
        "l_max": 4,
        "l_max_4body": 2,
    }
    values.update(changes)
    if values["message_mode"] == "scalar":
        values.pop("equivariant_channels")
        values.pop("equivariant_l_max")
    return resolve_params(values)


def _model(device, **changes):
    torch.manual_seed(20261021)
    return Mobius(_params(**changes)).to(device=device, dtype=torch.float64)


def _batch(device, *, empty=False, dtype=torch.float64):
    cell = torch.tensor(
        [[3.9, 0.1, 0.2], [0.0, 4.1, 0.1], [0.1, 0.0, 4.3]],
        device=device,
        dtype=dtype,
    )
    fractional = torch.tensor(
        [[0.10, 0.13, 0.22], [0.82, 0.21, 0.37],
         [0.30, 0.75, 0.19], [0.68, 0.63, 0.81]],
        device=device,
        dtype=dtype,
    )
    positions = fractional @ cell
    count = len(positions)
    if empty:
        neighbors = torch.full((count, 1), -1, device=device, dtype=torch.long)
        offsets = positions.new_zeros((count, 1, 3))
    else:
        rows = []
        offset_rows = []
        for center in range(count):
            others = [neighbor for neighbor in range(count) if neighbor != center]
            shifts = [
                -torch.round(fractional[neighbor] - fractional[center]) @ cell
                for neighbor in others
            ]
            # Opposite periodic images of the same atom test the strain path:
            # their atom-position derivatives cancel, but their virials do not.
            rows.append(others + [center, center])
            offset_rows.append(torch.stack(shifts + [cell[0], -cell[0]]))
        neighbors = torch.tensor(rows, device=device, dtype=torch.long)
        offsets = torch.stack(offset_rows)
    return {
        "positions": positions,
        "types": torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long),
        "radial_neighbors": neighbors,
        "angular_neighbors": neighbors.clone(),
        "radial_offsets": offsets,
        "angular_offsets": offsets.clone(),
        "n_atoms_per_structure": torch.tensor([count], device=device),
    }


def _transform(batch, matrix):
    result = dict(batch)
    for key in ("positions", "radial_offsets", "angular_offsets"):
        result[key] = batch[key] @ matrix.T
    return result


def _energy(model, batch):
    return model({**batch, "compute_forces": False, "compute_virial": False})[
        "energy"
    ].sum()


def _geometry_reference(model, batch, *, create_graph=False):
    """Differentiate every geometry dependency, independently of force.py."""
    positions = batch["positions"].detach().clone().requires_grad_(True)
    strain = positions.new_zeros((3, 3), requires_grad=True)
    deformation = torch.eye(3, device=positions.device, dtype=positions.dtype) + strain
    moved = _transform({**batch, "positions": positions}, deformation)
    state = model.descriptor.compute_state(moved, track_geometry=True)
    edge = state.radial
    energies = model._atomic_energies(
        state.values,
        moved["types"],
        edge.center,
        edge.neighbor,
        edge.vectors,
        moments=state.angular.moments,
    )
    position_gradient, strain_gradient = torch.autograd.grad(
        energies.sum(),
        (positions, strain),
        create_graph=create_graph,
    )
    return {
        "energy": energies.sum().reshape(1),
        "forces": -position_gradient,
        "virial": -strain_gradient.reshape(1, 9),
    }


def _check_energy_force_and_virial_transform_together(device, parity):
    model = _model(device).eval()
    batch = _batch(device)
    axis = batch["positions"].new_tensor([1.0, -2.0, 3.0])
    axis = axis / axis.norm()
    cross = axis.new_zeros((3, 3))
    cross[0, 1], cross[0, 2] = -axis[2], axis[1]
    cross[1, 0], cross[1, 2] = axis[2], -axis[0]
    cross[2, 0], cross[2, 1] = -axis[1], axis[0]
    angle = axis.new_tensor(0.713)
    rotation = (
        torch.eye(3, device=device, dtype=axis.dtype)
        + angle.sin() * cross
        + (1 - angle.cos()) * (cross @ cross)
    ) * parity
    original = model(batch)
    rotated = model(_transform(batch, rotation))
    # Descriptor constants originate in float32; double inputs cannot undo
    # their rounding, so symmetry tolerances account for that existing limit.
    torch.testing.assert_close(rotated["energy"], original["energy"], rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(
        rotated["forces"], original["forces"] @ rotation.T, rtol=3e-6, atol=3e-7
    )
    expected_virial = rotation @ original["virial"].reshape(3, 3) @ rotation.T
    torch.testing.assert_close(
        rotated["virial"].reshape(3, 3), expected_virial, rtol=3e-6, atol=3e-7
    )
    torch.testing.assert_close(
        original["forces"].sum(0), axis.new_zeros(3), rtol=0, atol=2e-10
    )


def _check_periodic_force_and_full_strain_virial_finite_differences(device):
    model = _model(device).eval()
    batch = _batch(device)
    result = model(batch)
    step = 2e-5
    force_fd = torch.zeros_like(batch["positions"])
    for atom in range(len(force_fd)):
        for component in range(3):
            shift = torch.zeros_like(force_fd)
            shift[atom, component] = step
            with torch.no_grad():
                plus = _energy(model, {**batch, "positions": batch["positions"] + shift})
                minus = _energy(model, {**batch, "positions": batch["positions"] - shift})
            force_fd[atom, component] = -(plus - minus) / (2 * step)
    virial_fd = force_fd.new_zeros((3, 3))
    identity = torch.eye(3, device=device, dtype=force_fd.dtype)
    for row in range(3):
        for column in range(3):
            strain = torch.zeros_like(identity)
            strain[row, column] = step
            with torch.no_grad():
                plus = _energy(model, _transform(batch, identity + strain))
                minus = _energy(model, _transform(batch, identity - strain))
            virial_fd[row, column] = -(plus - minus) / (2 * step)
    torch.testing.assert_close(result["forces"], force_fd, rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(
        result["virial"].reshape(3, 3), virial_fd, rtol=2e-5, atol=3e-7
    )


def _without_angular_neighbors(batch):
    return {
        **batch,
        "angular_neighbors": torch.full_like(batch["angular_neighbors"], -1),
        "angular_offsets": torch.zeros_like(batch["angular_offsets"]),
    }


def _check_analytical_derivatives_match_complete_geometry_graph(
    device, order, *, empty_angular=False, **changes
):
    model = _model(device, equivariant_l_max=order, **changes).eval()
    batch = _batch(device)
    if empty_angular:
        batch = _without_angular_neighbors(batch)
    actual = model(batch)
    reference = _geometry_reference(model, batch)
    for key in ("energy", "forces", "virial"):
        torch.testing.assert_close(actual[key], reference[key], rtol=2e-7, atol=2e-8)


def _check_force_loss_parameter_gradients_match_complete_geometry_graph(
    device, *, empty_angular=False, **changes
):
    settings = {"l_max": 2, "equivariant_l_max": 2, **changes}
    model = _model(device, **settings).train()
    reference_model = copy.deepcopy(model)
    batch = _batch(device)
    if empty_angular:
        batch = _without_angular_neighbors(batch)
    target_force = batch["positions"].new_tensor(
        [[0.13, -0.27, 0.09], [-0.17, 0.21, 0.04],
         [0.06, 0.16, -0.22], [0.11, -0.02, 0.19]]
    )
    actual = model(batch)
    reference = _geometry_reference(reference_model, batch, create_graph=True)
    actual_loss = (actual["forces"] - target_force).square().mean()
    reference_loss = (reference["forces"] - target_force).square().mean()
    named_actual = dict(model.named_parameters())
    named_reference = dict(reference_model.named_parameters())
    actual_gradients = torch.autograd.grad(
        actual_loss, tuple(named_actual.values()), allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        reference_loss, tuple(named_reference.values()), allow_unused=True
    )
    for (name, parameter), actual_grad, reference_grad in zip(
        named_actual.items(), actual_gradients, reference_gradients
    ):
        # An energy offset can be absent from the force-only graph.
        if actual_grad is None:
            actual_grad = torch.zeros_like(parameter)
        if reference_grad is None:
            reference_grad = torch.zeros_like(parameter)
        torch.testing.assert_close(
            actual_grad, reference_grad, rtol=3e-6, atol=3e-8, msg=name
        )


def _check_direction_input_is_used_in_addition_to_invariant_descriptors(device):
    model = _model(device).eval()
    batch = _batch(device)
    state = model.descriptor.compute_state(batch)
    edge = state.radial
    arguments = (state.values, batch["types"], edge.center, edge.neighbor, edge.vectors)
    actual = model._atomic_energies(*arguments, moments=state.angular.moments)
    removed = model._atomic_energies(
        *arguments, moments=torch.zeros_like(state.angular.moments)
    )
    assert (actual - removed).abs().max().item() > 1e-7


def _assert_same_prediction(actual, expected):
    # CUDA neighbor scatter can accumulate in a different floating-point order
    # even when evaluating the same unchanged model twice.
    if expected.device.type != "cuda":
        rtol, atol = 0, 0
    elif expected.dtype == torch.float64:
        rtol, atol = 1e-12, 1e-12
    else:
        rtol, atol = 3e-6, 5e-7
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _assert_identical_state_dict(actual_model, expected_model):
    actual = actual_model.state_dict()
    expected = expected_model.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


def _check_inference_gradient_context_and_output_flags(device):
    model = _model(device).eval()
    batch = _batch(device)
    expected = model(batch)
    with torch.no_grad():
        actual = model(batch)
        forces_only = model({**batch, "compute_virial": False})
        virial_only = model({**batch, "compute_forces": False})
    for key in ("energy", "forces", "virial"):
        _assert_same_prediction(actual[key], expected[key])
    assert "virial" not in forces_only
    assert "forces" not in virial_only
    _assert_same_prediction(forces_only["forces"], expected["forces"])
    _assert_same_prediction(virial_only["virial"], expected["virial"])


def _check_empty_neighbors_keep_training_parameters_in_graph(device):
    model = _model(device, l_max=2, equivariant_l_max=2).train()
    result = model(_batch(device, empty=True))
    loss = result["energy"].square().sum() + result["forces"].square().sum()
    loss = loss + result["virial"].square().sum()
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert not missing, f"Parameters disconnected on empty neighbor lists: {missing}"
    for name, value in model.named_parameters():
        assert torch.isfinite(value.grad).all(), name
    assert torch.count_nonzero(result["forces"]).item() == 0
    assert torch.count_nonzero(result["virial"]).item() == 0


def _check_old_scalar_parameters_and_checkpoint_keys_still_work(device, layers):
    model = _model(device, message_mode="scalar", message_layers=layers).eval()
    old_params = dict(model.params)
    for name in ("message_mode", "equivariant_channels", "equivariant_l_max"):
        old_params.pop(name, None)
    restored = Mobius(old_params).to(device=device, dtype=torch.float64).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    _assert_identical_state_dict(restored, model)
    assert not any("equivariant" in name for name in model.state_dict())
    batch = _batch(device)
    original = model(batch)
    repeated = model(batch)
    result = restored(batch)
    for key in ("energy", "forces", "virial"):
        _assert_same_prediction(repeated[key], original[key])
        _assert_same_prediction(result[key], original[key])


def _check_equivariant_checkpoint_roundtrip(device, tmp_path):
    # Exercise the float32 precision used by the archived OAM checkpoints.
    model = _model(device).float().eval()
    path = tmp_path / "equivariant.pt"
    torch.save(
        {
            "plugin": {"name": PLUGIN_NAME},
            "model_params": model.params,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    restored = load_model(path, device=device).eval()
    assert restored.params["message_mode"] == "equivariant"
    assert restored.params["equivariant_l_max"] == 4
    _assert_identical_state_dict(restored, model)
    batch = _batch(device, dtype=torch.float32)
    original = model(batch)
    repeated = model(batch)
    result = restored(batch)
    for key in ("energy", "forces", "virial"):
        _assert_same_prediction(repeated[key], original[key])
        _assert_same_prediction(result[key], original[key])


class MobiusEquivariantTests(unittest.TestCase):
    def setUp(self):
        self.device = _device()

    def test_energy_force_and_virial_transform_together(self):
        for parity in (1, -1):
            with self.subTest(parity=parity):
                _check_energy_force_and_virial_transform_together(self.device, parity)

    def test_periodic_force_and_full_strain_virial_finite_differences(self):
        _check_periodic_force_and_full_strain_virial_finite_differences(self.device)

    def test_analytical_derivatives_match_complete_geometry_graph(self):
        for order in (1, 2, 4):
            with self.subTest(order=order):
                _check_analytical_derivatives_match_complete_geometry_graph(
                    self.device, order
                )
        with self.subTest(descriptors="all_high_body"):
            _check_analytical_derivatives_match_complete_geometry_graph(
                self.device, 4, l_max_5body=1, has_q_112=1,
                has_q_123=1, has_q_233=1, has_q_134=1,
            )
        with self.subTest(neighbors="radial_only"):
            _check_analytical_derivatives_match_complete_geometry_graph(
                self.device, 4, empty_angular=True
            )

    def test_force_loss_parameter_gradients_match_complete_geometry_graph(self):
        cases = [
            {},
            {"empty_angular": True},
            {
                "l_max": 4, "equivariant_l_max": 4, "l_max_5body": 1,
                "has_q_112": 1, "has_q_123": 1, "has_q_233": 1, "has_q_134": 1,
            },
        ]
        for changes in cases:
            with self.subTest(**changes):
                _check_force_loss_parameter_gradients_match_complete_geometry_graph(
                    self.device, **changes
                )

    def test_direction_input_is_used_in_addition_to_invariant_descriptors(self):
        _check_direction_input_is_used_in_addition_to_invariant_descriptors(self.device)

    def test_inference_gradient_context_and_output_flags(self):
        _check_inference_gradient_context_and_output_flags(self.device)

    def test_empty_neighbors_keep_training_parameters_in_graph(self):
        _check_empty_neighbors_keep_training_parameters_in_graph(self.device)

    def test_old_scalar_parameters_and_checkpoint_keys_still_work(self):
        for layers in (0, 2):
            with self.subTest(message_layers=layers):
                _check_old_scalar_parameters_and_checkpoint_keys_still_work(
                    self.device, layers
                )

    def test_equivariant_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_equivariant_checkpoint_roundtrip(self.device, Path(temporary))

    def test_equivariant_configuration_validation(self):
        cases = [
            ({"message_mode": "unknown"}, "message_mode"),
            ({"message_layers": 0}, "message_layers"),
            ({"equivariant_channels": 0}, "equivariant_channels"),
            ({"equivariant_channels": True}, "equivariant_channels"),
            ({"equivariant_l_max": 0}, "equivariant_l_max"),
            ({"equivariant_l_max": 5}, "equivariant_l_max"),
            ({"equivariant_l_max": 1.5}, "equivariant_l_max"),
            ({"l_max": 9, "equivariant_l_max": 8}, "l_max"),
        ]
        for changes, field in cases:
            with self.subTest(**changes), self.assertRaisesRegex(ValueError, field):
                _params(**changes)

    def test_direction_order_defaults_to_descriptor_order(self):
        self.assertEqual(_params(l_max=3)["equivariant_l_max"], 3)


if __name__ == "__main__":
    unittest.main()
