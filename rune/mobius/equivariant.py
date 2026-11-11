"""Persistent angular states with invariant gates and fixed-shape messages."""

from __future__ import annotations

import torch
import torch.nn as nn

from .descriptor import chebyshev_basis
from .layers import RMSNorm, activation_factory


class DirectionEncoder(nn.Module):
    """Project normalized angular moments along the radial channel axis.

    Input moments have shape ``[atoms, n_desc, l_max * (l_max + 2)]``.
    Each order ``l=1,...,l_max`` has its own channel map, shared by all
    ``2*l+1`` angular components. The input basis must have a Euclidean
    invariant metric; raw NEP-style polynomial components do not.
    """

    def __init__(self, n_desc: int, channels: int, l_max: int):
        super().__init__()
        if n_desc <= 0 or channels <= 0 or l_max <= 0:
            raise ValueError("DirectionEncoder dimensions must be positive.")
        self.n_desc = int(n_desc)
        self.channels = int(channels)
        self.l_max = int(l_max)
        self.projections = nn.ModuleList(
            nn.Linear(self.n_desc, self.channels, bias=False)
            for _ in range(self.l_max)
        )

    def forward(self, normalized_moments: torch.Tensor) -> torch.Tensor:
        states = []
        for order, projection in enumerate(self.projections, start=1):
            start, stop = order * order - 1, (order + 1) ** 2 - 1
            moments = normalized_moments[..., start:stop]
            states.append(projection(moments.transpose(1, 2)).transpose(1, 2))
        return torch.cat(states, dim=-1)


class EquivariantBlock(nn.Module):
    r"""Update scalar and angular states using invariant edge gates.

    For directed edge ``i <- j``, ``vectors = R_j + shift - R_i`` and
    ``harmonics = Y(vectors / |vectors|)``. Angular states and harmonics
    share the same normalized real basis, packed in increasing order
    ``l=1,...,l_max``. The persistent state has shape ``[atoms, channels, M]``.

    The edge network sees scalar node features and the invariant probes
    ``H_i^l . Y_l(r_hat_ij)`` and ``H_j^l . Y_l(-r_hat_ij)``. It supplies
    signed scalar gates for

    ``H_i'^l = H_i^l + sum_j fc * (a_ij^l * W_l H_j^l + b_ij^l * Y_l)``.

    ``W_l`` mixes channels only. All components of an angular order receive
    the same gate, and no component-wise angular activation or bias is used.
    The scalar update sees grouped scalar messages and squared angular norms.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        message_dim: int,
        pair_dim: int,
        message_groups: int,
        equivariant_channels: int,
        l_max: int,
        n_elements: int,
        basis_size: int,
        cutoff: float,
        activation: str,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        if hidden_dim <= 0 or message_dim <= 0 or pair_dim <= 0:
            raise ValueError("Equivariant block dimensions must be positive.")
        if message_groups <= 0 or hidden_dim % message_groups != 0:
            raise ValueError(
                "message_groups must be positive and divide hidden_dim."
            )
        if equivariant_channels <= 0 or l_max <= 0:
            raise ValueError("Angular channels and l_max must be positive.")
        if n_elements <= 0 or basis_size <= 0 or cutoff <= 0.0:
            raise ValueError("Elements, radial basis size and cutoff must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.message_groups = int(message_groups)
        self.group_width = self.hidden_dim // self.message_groups
        self.equivariant_channels = int(equivariant_channels)
        self.l_max = int(l_max)
        self.n_elements = int(n_elements)
        self.basis_size = int(basis_size)
        self.cutoff = float(cutoff)

        make_activation = activation_factory(activation)
        self.input_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.center_context = nn.Linear(hidden_dim, message_dim, bias=False)
        self.neighbor_context = nn.Linear(hidden_dim, message_dim, bias=False)
        self.pair_embedding = nn.Embedding(n_elements * n_elements, pair_dim)
        angular_width = self.l_max * self.equivariant_channels
        edge_input_dim = basis_size + pair_dim + 2 * message_dim + 2 * angular_width
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, message_dim),
            make_activation(),
        )
        self.group_gate = nn.Linear(message_dim, self.message_groups)
        self.direction_gates = nn.Linear(message_dim, 2 * angular_width)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.direction_mix = nn.ModuleList(
            nn.Linear(
                self.equivariant_channels, self.equivariant_channels, bias=False
            )
            for _ in range(self.l_max)
        )
        self.message_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim + angular_width, message_dim),
            make_activation(),
            nn.Linear(message_dim, hidden_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        direction: torch.Tensor,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        vectors: torch.Tensor,
        harmonics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_atoms = hidden.shape[0]
        normalized = self.input_norm(hidden)

        if center.numel() == 0:
            # DDP needs every parameter in the graph even for isolated atoms.
            parameter_dependency = sum(
                parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()
            )
            scalar_message = hidden.new_zeros(
                (n_atoms, self.hidden_dim)
            ) + parameter_dependency
            new_direction = direction + parameter_dependency
        else:
            probes_center, probes_neighbor, mixed_states = [], [], []
            for order, mix in enumerate(self.direction_mix, start=1):
                start, stop = order * order - 1, (order + 1) ** 2 - 1
                state = direction[..., start:stop]
                basis = harmonics[:, None, start:stop]
                probes_center.append((state[center] * basis).sum(dim=-1))
                # Natural parity gives Y_l(-r_hat) = (-1)^l Y_l(r_hat).
                probes_neighbor.append(
                    ((-1) ** order) * (state[neighbor] * basis).sum(dim=-1)
                )
                # Precompute each channel map once per atom, before edge gathers.
                mixed_states.append(mix(state.transpose(1, 2)).transpose(1, 2))

            distances = torch.linalg.norm(vectors, dim=-1)
            radial_basis = chebyshev_basis(distances, self.cutoff, self.basis_size)
            pair_index = types[center] * self.n_elements + types[neighbor]
            edge_hidden = self.edge_mlp(
                torch.cat(
                    (
                        radial_basis,
                        self.pair_embedding(pair_index),
                        self.center_context(normalized)[center],
                        self.neighbor_context(normalized)[neighbor],
                        torch.cat(probes_center, dim=-1),
                        torch.cat(probes_neighbor, dim=-1),
                    ),
                    dim=-1,
                )
            )
            cutoff = torch.where(
                distances < self.cutoff,
                0.5 * torch.cos(torch.pi * distances / self.cutoff) + 0.5,
                torch.zeros_like(distances),
            )
            gates = torch.sigmoid(self.group_gate(edge_hidden)) * cutoff[:, None]
            values = self.value_projection(normalized)[neighbor]
            edge_values = (
                values.reshape(-1, self.message_groups, self.group_width)
                * gates.unsqueeze(-1)
            ).reshape(-1, self.hidden_dim)
            scalar_message = hidden.new_zeros((n_atoms, self.hidden_dim))
            scalar_message.scatter_add_(
                0, center[:, None].expand_as(edge_values), edge_values
            )

            # Signed, bounded gates permit both transport and cancellation.
            angular_gates = torch.tanh(self.direction_gates(edge_hidden)).reshape(
                -1, 2, self.l_max, self.equivariant_channels
            )
            updated_states = []
            for order, mixed in enumerate(mixed_states, start=1):
                start, stop = order * order - 1, (order + 1) ** 2 - 1
                transport = angular_gates[:, 0, order - 1, :, None]
                injection = angular_gates[:, 1, order - 1, :, None]
                edge_direction = cutoff[:, None, None] * (
                    transport * mixed[neighbor]
                    + injection * harmonics[:, None, start:stop]
                )
                increment = direction.new_zeros(
                    (n_atoms, self.equivariant_channels, 2 * order + 1)
                )
                increment.scatter_add_(
                    0, center[:, None, None].expand_as(edge_direction), edge_direction
                )
                updated_states.append(direction[..., start:stop] + increment)
            new_direction = torch.cat(updated_states, dim=-1)

        angular_invariants = []
        for order in range(1, self.l_max + 1):
            start, stop = order * order - 1, (order + 1) ** 2 - 1
            angular_invariants.append(
                new_direction[..., start:stop].square().sum(dim=-1)
            )
        update_input = torch.cat(
            (self.message_norm(scalar_message), *angular_invariants), dim=-1
        )
        return hidden + self.update(update_input), new_direction
