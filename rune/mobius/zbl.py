"""Optional ZBL short-range repulsion used by Mobius."""

from __future__ import annotations

import torch
import torch.nn as nn
from ase.data import atomic_numbers


K_C_SP = 14.399645
ZBL_A_INV_FACTOR = 2.134563
UNIVERSAL_ZBL_COEFFS = (
    (0.18175, 3.1998),
    (0.50986, 0.94229),
    (0.28022, 0.4029),
    (0.02817, 0.20162),
)


class ZBLRepulsion(nn.Module):
    """Universal or explicitly parameterized pairwise ZBL energy.

    Neighbor lists in RUNE are directed, so every edge contributes half of the
    screened pair energy.  ``forward`` returns per-center-atom energies.
    """

    def __init__(self, elements: list[str], config):
        super().__init__()
        self.n_elements = len(elements)
        normalized = self._normalize(config)
        self.rc_inner = normalized["rc_inner"]
        self.rc_outer = normalized["rc_outer"]
        self.flexible = normalized["flexible"]
        self.register_buffer(
            "atomic_numbers",
            torch.tensor([atomic_numbers[e] for e in elements], dtype=torch.float32),
        )
        self.register_buffer(
            "universal_coefficients",
            torch.tensor(UNIVERSAL_ZBL_COEFFS, dtype=torch.float32),
        )
        parameters = normalized.get("parameters")
        if self.flexible:
            expected = self.n_elements * (self.n_elements + 1) // 2
            tensor = torch.as_tensor(parameters, dtype=torch.float32).reshape(
                -1, 10
            )
            if tensor.shape[0] != expected:
                raise ValueError(
                    f"Flexible ZBL expects {expected} pair rows, got {tensor.shape[0]}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError("Flexible ZBL parameters must all be finite.")
            invalid_cutoffs = (tensor[:, 0] < 0.0) | (
                tensor[:, 0] >= tensor[:, 1]
            )
            if invalid_cutoffs.any():
                raise ValueError(
                    "Each flexible ZBL pair must satisfy 0 <= rc_inner < rc_outer."
                )
        else:
            tensor = torch.zeros((0, 10), dtype=torch.float32)
        self.register_buffer("pair_parameters", tensor)
        self.max_rc_outer = (
            float(tensor[:, 1].max()) if self.flexible else self.rc_outer
        )

    @staticmethod
    def _normalize(config) -> dict:
        if isinstance(config, dict):
            inner = float(config["rc_inner"])
            outer = float(config["rc_outer"])
            flexible = bool(config.get("flexible", inner == 0.0 and outer == 0.0))
            parameters = config.get("parameters")
        else:
            if len(config) != 2:
                raise ValueError("zbl must be [rc_inner, rc_outer] or a mapping.")
            inner, outer = map(float, config)
            flexible = inner == 0.0 and outer == 0.0
            parameters = None
        if flexible and parameters is None:
            raise ValueError("Flexible ZBL requires ten parameters per element pair.")
        if not flexible and not (0.0 <= inner < outer):
            raise ValueError("Universal ZBL requires 0 <= rc_inner < rc_outer.")
        return {
            "rc_inner": inner,
            "rc_outer": outer,
            "flexible": flexible,
            "parameters": parameters,
        }

    def forward(
        self,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        energy = vectors.new_zeros(types.shape[0])
        if center.numel() == 0:
            return energy

        distances = torch.clamp(torch.linalg.norm(vectors, dim=-1), min=1.0e-12)
        type_i, type_j = types[center], types[neighbor]
        numbers = self.atomic_numbers.to(dtype=vectors.dtype)
        z_i, z_j = numbers[type_i], numbers[type_j]
        a_inv = (z_i.pow(0.23) + z_j.pow(0.23)) * ZBL_A_INV_FACTOR
        prefactor = K_C_SP * z_i * z_j

        if self.flexible:
            low = torch.minimum(type_i, type_j)
            high = torch.maximum(type_i, type_j)
            pair_index = low * self.n_elements - low * (low - 1) // 2 + high - low
            parameters = self.pair_parameters.to(dtype=vectors.dtype)[pair_index]
            rc_inner, rc_outer = parameters[:, 0], parameters[:, 1]
            coefficients = parameters[:, 2:].reshape(-1, 4, 2)
        else:
            rc_inner = distances.new_full(distances.shape, self.rc_inner)
            rc_outer = distances.new_full(distances.shape, self.rc_outer)
            coefficients = self.universal_coefficients.to(
                dtype=vectors.dtype
            ).unsqueeze(0)

        screened = prefactor * torch.sum(
            coefficients[..., 0]
            * torch.exp(-coefficients[..., 1] * (distances * a_inv).unsqueeze(-1)),
            dim=-1,
        ) / distances
        width = torch.clamp(rc_outer - rc_inner, min=1.0e-12)
        cutoff = torch.where(
            distances < rc_inner,
            torch.ones_like(distances),
            torch.where(
                distances < rc_outer,
                0.5 * torch.cos(torch.pi * (distances - rc_inner) / width) + 0.5,
                torch.zeros_like(distances),
            ),
        )
        edge_energy = 0.5 * screened * cutoff
        energy.scatter_add_(0, center, edge_energy)
        return energy
