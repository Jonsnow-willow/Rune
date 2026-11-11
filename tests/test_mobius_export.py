"""Lossless native exports and independent polynomial value/gradient checks."""

from pathlib import Path
import tempfile
import unittest

import torch

from rune.mobius.descriptor import AngularDescriptor
from rune.mobius.export import _basis_terms, _invariant_terms, export_data
from rune.mobius.plugin import build_model, export_model, resolve_params


def _model(**changes):
    values = dict(elements=["Fe", "C"], hidden_dim=8, blocks=1, ffn_dim=12,
                  message_dim=8, message_groups=2, pair_dim=3, vector_channels=2,
                  n_max_radial=1, n_max_angular=1, basis_size_radial=2,
                  basis_size_angular=2, l_max=4, l_max_4body=2, l_max_5body=1,
                  has_q_112=1, has_q_123=1, has_q_233=1, has_q_134=1)
    values.update(changes)
    return build_model(resolve_params(values))


def _read_export(path):
    words = iter(Path(path).read_text().split())
    assert next(words) == "mobius1"
    count = int(next(words))
    elements = [next(words) for _ in range(count)]
    assert next(words) == "schema" and next(words) == "1"
    groups = []
    for name, convert in (("ints", int), ("reals", float)):
        assert next(words) == name
        size = int(next(words))
        groups.append({next(words): convert(next(words)) for _ in range(size)})
    assert next(words) == "tensors"
    tensors = {}
    for _ in range(int(next(words))):
        name, rank = next(words), int(next(words))
        shape = [int(next(words)) for _ in range(rank)]
        size = 1
        for dimension in shape:
            size *= dimension
        tensors[name] = torch.tensor([float(next(words)) for _ in range(size)], dtype=torch.float64).reshape(shape)
    assert next(words) == "END"
    assert next(words, None) is None
    return elements, *groups, tensors


class NativeExportTests(unittest.TestCase):
    def test_export_preserves_all_weights_scaler_elements_and_zbl(self):
        flexible = dict(rc_inner=0.0, rc_outer=0.0, flexible=True,
                        parameters=[[0.5, 1.8, .18, 3.2, .51, .94, .28, .40, .03, .20]] * 3)
        for mode in ("local", "scalar", "equivariant"):
            for dtype in (torch.float32, torch.float64):
                for zbl in (None, [1.0, 2.0], flexible):
                    with self.subTest(mode=mode, dtype=dtype, zbl=zbl):
                        changes = dict(message_layers=0 if mode == "local" else 2, zbl=zbl)
                        if mode == "equivariant":
                            changes.update(message_mode=mode, equivariant_channels=3)
                        model = _model(**changes).to(dtype=dtype)
                        model.set_descriptor_scaler(torch.linspace(.4, 1.6, model.input_dim))
                        before = {key: value.clone() for key, value in model.state_dict().items()}
                        with tempfile.TemporaryDirectory() as directory:
                            path = export_model(model, Path(directory) / "model")
                            self.assertEqual(path.name, "mobius.txt")
                            elements, ints, reals, tensors = _read_export(path)
                        self.assertEqual(elements, ["Fe", "C"])
                        self.assertEqual(ints["message_layers"], changes["message_layers"])
                        self.assertEqual(ints["message_mode"], int(mode == "equivariant"))
                        self.assertEqual(ints["zbl_enabled"], int(zbl is not None))
                        self.assertEqual(reals["rcut_radial"], model.params["rcut_radial"])
                        for key, value in before.items():
                            torch.testing.assert_close(tensors[key], value.double(), rtol=0, atol=0)
                            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_sparse_angular_polynomials_and_derivatives_all_orders(self):
        torch.manual_seed(138)
        for order in range(1, 9):
            angular = AngularDescriptor(2, 2, 3, 4., order, has_q_222=int(order >= 2)).double()
            # Testing arbitrary coordinates also catches an incorrect use of
            # unit-sphere identities in the polynomial derivative table.
            coordinates = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
            terms, _ = _basis_terms(angular)
            components = order * (order + 2)
            columns = []
            for component in range(components):
                selected = terms[terms[:, 0] == component]
                monomials = coordinates[:, None, :].pow(selected[:, 2:].long()).prod(-1)
                columns.append((monomials * selected[:, 1]).sum(-1))
            actual = torch.stack(columns, -1)
            expected = angular._compute_basis(coordinates, None)[:, :components]
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-10)
            sensitivity = torch.randn_like(actual)
            actual_gradient, = torch.autograd.grad((actual * sensitivity).sum(), coordinates, retain_graph=True)
            expected_gradient, = torch.autograd.grad((expected * sensitivity).sum(), coordinates)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-12, atol=2e-9)

    def test_all_invariant_polynomials_gradients_and_metric(self):
        model = _model(message_mode="equivariant", equivariant_channels=3).double()
        angular = model.descriptor.angular
        torch.manual_seed(813)
        moments = torch.randn(4, angular.n_desc, 24, dtype=torch.float64, requires_grad=True)
        terms, _ = _invariant_terms(angular)
        columns = []
        for output in range(angular.num_L):
            column = torch.zeros_like(moments[..., 0])
            for term in terms[terms[:, 0] == output]:
                value = torch.ones_like(column) * term[1]
                for index in term[3:3 + int(term[2])].long():
                    value = value * moments[..., index]
                column = column + value
            columns.append(column)
        actual = torch.stack(columns, -1)
        expected = angular._compute_q(moments)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
        sensitivity = torch.randn_like(actual)
        actual_gradient, = torch.autograd.grad((actual * sensitivity).sum(), moments, retain_graph=True)
        expected_gradient, = torch.autograd.grad((expected * sensitivity).sum(), moments)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-13, atol=2e-13)
        _, _, tensors = export_data(model)
        torch.testing.assert_close(tensors["desc.orthonormal_scale"],
                                   angular.orthonormal_scale(moments, 4), rtol=0, atol=0)

    def test_reject_nonfinite_and_low_precision_export(self):
        model = _model()
        with torch.no_grad():
            model.q_scaler[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            export_data(model)
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            export_data(_model().half())


if __name__ == "__main__":
    unittest.main()
