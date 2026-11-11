"""Static scalar layers used by the Mobius potential."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Return a constructor for a supported, export-friendly activation."""
    normalized = str(name).strip().lower()
    factories: dict[str, Callable[[], nn.Module]] = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
    }
    if normalized not in factories:
        choices = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported Mobius activation {name!r}; choose one of {choices}.")
    return factories[normalized]


class RMSNorm(nn.Module):
    """RMS normalization without centering.

    Keeping this operation explicit makes the future native evaluator contract
    unambiguous and avoids depending on a particular PyTorch ``nn.RMSNorm``
    version.
    """

    def __init__(self, dimension: int, eps: float = 1.0e-8):
        super().__init__()
        if dimension <= 0:
            raise ValueError("RMSNorm dimension must be positive.")
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values * scale * self.weight


class ResidualBlock(nn.Module):
    """Pre-normalized scalar residual feed-forward block."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        activation: str,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        make_activation = activation_factory(activation)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            make_activation(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.ffn(self.norm(hidden))


class ElementwiseLinearHead(nn.Module):
    """One small linear atomic-energy head per central element.

    The shared Mobius representation supplies ``hidden``.  Element type
    ``z_i`` selects a row of ``weight`` and one scalar from ``bias``:

    ``E_i = weight[z_i] dot hidden_i + bias[z_i]``.
    """

    def __init__(self, n_elements: int, hidden_dim: int):
        super().__init__()
        if n_elements <= 0:
            raise ValueError("ElementwiseLinearHead requires at least one element.")
        self.weight = nn.Parameter(torch.empty(n_elements, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(n_elements))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        selected_weight = self.weight[types]
        return torch.sum(selected_weight * hidden, dim=-1) + self.bias[types]
