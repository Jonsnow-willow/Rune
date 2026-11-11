"""Lossless model export for the native GPUMD Möbius evaluator.

All floating-point arrays are written with 17 significant digits. The native
evaluator uses double precision; float32 weights are promoted without rounding
or fitting. Polynomial tables retain the descriptor's existing buffer values,
while the normalized direction metric uses its original double constants,
exactly as ``model.double()`` does.
"""

from __future__ import annotations

from pathlib import Path
import math

import torch

from .descriptor import Q123_TERMS, Q233_TERMS, Q134_TERMS, _complex_power_terms


def _basis_terms(angular):
    """Sparse real angular polynomials: component, coefficient, x/y/z powers."""
    rows, offsets = [], [0]
    component = 0
    for order in range(1, angular.L_max + 1):
        z_coefficients = getattr(angular, f"Z{order}").detach().cpu().double()
        for xy_power in range(order + 1):
            for imaginary in ([False] if xy_power == 0 else [False, True]):
                for xy_coefficient, xp, yp in _complex_power_terms(xy_power, imaginary):
                    for zp in range((order + xy_power) % 2, order - xy_power + 1, 2):
                        coefficient = xy_coefficient * z_coefficients[xy_power, zp].item()
                        if coefficient:
                            rows.append([component, coefficient, xp, yp, zp])
                component += 1
                offsets.append(len(rows))
    return torch.tensor(rows, dtype=torch.float64), torch.tensor(offsets, dtype=torch.float64)


def _invariant_terms(angular):
    """Sparse q polynomials: output channel, coefficient, degree, four indices."""
    channels = []
    c3 = angular.C3B.detach().cpu().double().tolist()
    for order in range(1, angular.L_max + 1):
        start, stop = order * order - 1, (order + 1) ** 2 - 1
        channels.append([(c3[k] * (1 if k == start else 2), (k, k)) for k in range(start, stop)])
    if angular.has_q_222:
        c = angular.C4B.detach().cpu().double().tolist()
        channels.append([
            (c[0], (3, 3, 3)), (c[1], (3, 4, 4)), (c[1], (3, 5, 5)),
            (c[2], (3, 6, 6)), (c[2], (3, 7, 7)), (c[3], (6, 5, 5)),
            (-c[3], (6, 4, 4)), (c[4], (4, 5, 7)),
        ])
    if angular.has_q_1111:
        c = angular.C5B.detach().cpu().double().tolist()
        channels.append([
            (c[0], (0, 0, 0, 0)), (c[1], (0, 0, 1, 1)),
            (c[1], (0, 0, 2, 2)), (c[2], (1, 1, 1, 1)),
            (2 * c[2], (1, 1, 2, 2)), (c[2], (2, 2, 2, 2)),
        ])
    if angular.has_q_112:
        c = angular.C4B2.detach().cpu().double().tolist()
        channels.append([
            (c[0], (0, 0, 3)), (c[1], (0, 1, 4)), (c[1], (0, 2, 5)),
            (c[2], (3, 1, 1)), (c[2], (3, 2, 2)), (c[3], (6, 1, 1)),
            (-c[3], (6, 2, 2)), (c[4], (1, 2, 7)),
        ])
    for enabled, terms in ((angular.has_q_123, Q123_TERMS),
                           (angular.has_q_233, Q233_TERMS),
                           (angular.has_q_134, Q134_TERMS)):
        if enabled:
            channels.append(terms)
    rows, offsets = [], [0]
    for output, terms in enumerate(channels):
        for coefficient, indices in terms:
            rows.append([output, coefficient, len(indices), *indices, *([-1] * (4 - len(indices)))])
        offsets.append(len(rows))
    return torch.tensor(rows, dtype=torch.float64), torch.tensor(offsets, dtype=torch.float64)


def export_data(model):
    """Build the native schema without changing model weights, device or dtype."""
    if any(value.dtype not in (torch.float32, torch.float64)
           for value in model.state_dict().values()):
        raise ValueError("Native Möbius export requires float32 or float64 weights and buffers.")
    params = model.params
    names = (
        "n_desc_radial", "n_desc_angular", "k_max_radial", "k_max_angular",
        "l_max", "hidden_dim", "blocks", "ffn_dim", "message_layers",
        "message_dim", "pair_dim", "message_groups", "vector_channels",
        "message_basis_size", "max_neighbors_radial", "max_neighbors_angular",
    )
    ints = {name: int(params[name]) for name in names}
    activations = {"tanh": 0, "silu": 1, "swish": 1, "gelu": 2}
    ints.update(
        n_types=model.n_elements, num_L=model.num_L, input_dim=model.input_dim,
        message_mode=int(model.message_mode == "equivariant"),
        equivariant_channels=model.equivariant_channels,
        equivariant_l_max=model.equivariant_l_max,
        activation=activations[model.activation.strip().lower()],
        zbl_enabled=int(model.zbl is not None),
        zbl_flexible=int(model.zbl is not None and model.zbl.flexible),
    )
    reals = {name: float(params[name]) for name in
             ("rcut_radial", "rcut_angular", "message_cutoff", "norm_eps")}
    reals.update(zbl_rc_inner=model.zbl.rc_inner if model.zbl else 0.0,
                 zbl_rc_outer=model.zbl.rc_outer if model.zbl else 0.0)
    if not all(math.isfinite(value) for value in reals.values()):
        raise ValueError("Cannot export nonfinite Möbius configuration values.")
    tensors = {name: value.detach().cpu().double().contiguous()
               for name, value in model.state_dict().items()}
    angular = model.descriptor.angular
    tensors["desc.basis_terms"], tensors["desc.basis_offsets"] = _basis_terms(angular)
    tensors["desc.invariant_terms"], tensors["desc.invariant_offsets"] = _invariant_terms(angular)
    reference = torch.empty((), dtype=torch.float64)
    tensors["desc.orthonormal_scale"] = angular.orthonormal_scale(reference, angular.L_max)
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Cannot export a Möbius model containing nonfinite values.")
    return ints, reals, tensors


def export_model(model, path):
    """Write a GPUMD ``mobius1`` model and return the resulting path."""
    path = Path(path)
    if path.name == "model":
        path = path.with_name("mobius.txt")
    elif path.name.startswith("model_epoch_"):
        path = path.with_name(path.name.replace("model_", "mobius_", 1) + ".txt")
    ints, reals, tensors = export_data(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"mobius1 {model.n_elements} {' '.join(model.elements)}\n")
        handle.write("schema 1\n")
        handle.write(f"ints {len(ints)}\n")
        for name, value in ints.items():
            handle.write(f"{name} {value}\n")
        handle.write(f"reals {len(reals)}\n")
        for name, value in reals.items():
            handle.write(f"{name} {value:.17g}\n")
        handle.write(f"tensors {len(tensors)}\n")
        for name, value in tensors.items():
            dimensions = " ".join(str(size) for size in value.shape)
            handle.write(f"{name} {value.ndim} {dimensions}\n")
            flat = value.reshape(-1).tolist()
            for start in range(0, len(flat), 8):
                handle.write(" ".join(format(number, ".17g") for number in flat[start:start + 8]) + "\n")
        handle.write("END\n")
    return path
