"""Optional ZBL short-range repulsion used by NEP."""

from __future__ import annotations

from math import isfinite

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
    """Universal or fitted ZBL energy with fixed, model-owned buffers.

    Directed neighbor lists count each pair twice, so each edge contributes
    half the pair energy. The owning model computes forces and virials.
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
        if self.flexible:
            num_pairs = self.n_elements * (self.n_elements + 1) // 2
            # GPUMD stores ten parameters per pair in a fixed 550-float array.
            if num_pairs * 10 > 550:
                raise ValueError(
                    "GPUMD flexible ZBL supports at most 10 elements (550 parameters)."
                )
            parameters = torch.tensor(normalized["parameters"], dtype=torch.float32)
            if parameters.numel() != num_pairs * 10:
                raise ValueError(
                    f"Flexible ZBL requires {num_pairs * 10} parameters ({num_pairs} pairs)."
                )
            parameters = parameters.reshape(num_pairs, 10)
            if not torch.isfinite(parameters).all():
                raise ValueError("Flexible ZBL parameters must all be finite.")
            inner, outer = parameters[:, 0], parameters[:, 1]
            if ((inner < 0.0) | (inner >= outer)).any():
                raise ValueError(
                    "Each flexible ZBL pair requires 0 <= rc_inner < rc_outer."
                )
        else:
            parameters = torch.zeros((0, 10), dtype=torch.float32)
        self.register_buffer("pair_parameters", parameters)

    @staticmethod
    def _normalize(config) -> dict:
        if isinstance(config, dict):
            inner = float(config["rc_inner"])
            outer = float(config["rc_outer"])
            flexible = bool(config.get("flexible", inner == 0.0 and outer == 0.0))
            parameters = config.get("parameters")
        else:
            if len(config) != 2:
                raise ValueError("ZBL config should be a dict or [rc_inner, rc_outer].")
            inner, outer = map(float, config)
            flexible = inner == 0.0 and outer == 0.0
            parameters = None
        if flexible and parameters is None:
            raise ValueError("Flexible ZBL requires fitted ZBL parameters.")
        if not isfinite(inner) or not isfinite(outer):
            raise ValueError("ZBL cutoffs must be finite.")
        if flexible:
            if inner != 0.0 or outer != 0.0:
                raise ValueError("Flexible ZBL requires rc_inner = rc_outer = 0.")
        elif not 0.0 <= inner < outer:
            raise ValueError("Universal ZBL requires 0 <= rc_inner < rc_outer.")
        return {
            "rc_inner": inner,
            "rc_outer": outer,
            "flexible": flexible,
            "parameters": parameters,
        }

    @property
    def max_rc_outer(self) -> float:
        """Largest active pair cutoff, for the owning model's neighbor check."""
        return float(self.pair_parameters[:, 1].max()) if self.flexible else self.rc_outer

    def forward(
        self,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-center atomic energies from the directed pair list."""
        if center.numel() == 0:
            return vectors.new_zeros(types.shape[0])

        distances = torch.clamp(torch.linalg.norm(vectors, dim=-1), min=1.0e-12)
        type_i = types[center]
        type_j = types[neighbor]
        atomic_numbers_tensor = self.atomic_numbers.to(
            device=vectors.device,
            dtype=vectors.dtype,
        )
        z_i = atomic_numbers_tensor[type_i]
        z_j = atomic_numbers_tensor[type_j]
        a_inv = (torch.pow(z_i, 0.23) + torch.pow(z_j, 0.23)) * ZBL_A_INV_FACTOR
        zizj = K_C_SP * z_i * z_j

        if self.flexible:
            t1 = torch.minimum(type_i, type_j)
            t2 = torch.maximum(type_i, type_j)
            zbl_index = t1 * self.n_elements - (t1 * (t1 - 1)) // 2 + (t2 - t1)
            parameters = self.pair_parameters.to(
                device=vectors.device,
                dtype=vectors.dtype,
            )[zbl_index]
            rc_inner = parameters[:, 0]
            rc_outer = parameters[:, 1]
            coefficients = parameters[:, 2:].reshape(-1, 4, 2)
        else:
            rc_inner = vectors.new_full(distances.shape, self.rc_inner)
            rc_outer = vectors.new_full(distances.shape, self.rc_outer)
            coefficients = self.universal_coefficients.to(
                device=vectors.device,
                dtype=vectors.dtype,
            ).view(1, 4, 2)

        x = distances * a_inv
        phi = torch.sum(
            coefficients[..., 0] * torch.exp(-coefficients[..., 1] * x.unsqueeze(-1)),
            dim=-1,
        )
        screened = zizj * phi / distances
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
        pair_energy = 0.5 * screened * cutoff
        energy = vectors.new_zeros(types.shape[0])
        energy.scatter_add_(0, center, pair_energy)
        return energy
