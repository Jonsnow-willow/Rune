"""CUDA-oriented scalar and low-rank vector message passing for Mobius."""

from __future__ import annotations

import torch
import torch.nn as nn

from .descriptor import chebyshev_basis
from .layers import RMSNorm, activation_factory


class MessageBlock(nn.Module):
    r"""A grouped scalar message plus an immediately contracted ``l=1`` path.

    For directed edge ``i <- j`` the scalar channels are divided into ``G``
    fixed groups and updated with

    ``m_ic = sum_j a_ij,g(c) (W_v h_j)_c``.

    The narrow vector branch constructs

    ``v_ia = sum_j a^v_ija s_ja r_hat_ij``

    and immediately reduces it to invariant scalars ``t_ia = v_ia dot v_ia``.
    No vector state survives the block, so the persistent representation stays
    scalar and fixed-shape.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        message_dim: int,
        pair_dim: int,
        message_groups: int,
        vector_channels: int,
        n_elements: int,
        basis_size: int,
        cutoff: float,
        activation: str,
        norm_eps: float = 1.0e-8,
    ):
        super().__init__()
        if message_groups <= 0:
            raise ValueError("message_groups must be positive.")
        if hidden_dim % message_groups != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"message_groups ({message_groups})."
            )
        if vector_channels < 0:
            raise ValueError("vector_channels cannot be negative.")
        if basis_size <= 0:
            raise ValueError("message basis_size must be positive.")
        if cutoff <= 0.0:
            raise ValueError("message cutoff must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.message_groups = int(message_groups)
        self.group_width = self.hidden_dim // self.message_groups
        self.vector_channels = int(vector_channels)
        self.basis_size = int(basis_size)
        self.cutoff = float(cutoff)
        self.n_elements = int(n_elements)

        make_activation = activation_factory(activation)
        self.input_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.pair_embedding = nn.Embedding(n_elements * n_elements, pair_dim)
        edge_input_dim = self.basis_size + pair_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, message_dim),
            make_activation(),
        )
        self.group_gate = nn.Linear(message_dim, self.message_groups)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if self.vector_channels:
            self.vector_gate = nn.Linear(message_dim, self.vector_channels)
            self.vector_source = nn.Linear(
                hidden_dim, self.vector_channels, bias=False
            )
            # A fixed-width channel mixing before contraction increases the
            # rank of directional correlations without a tensor product.
            self.vector_mix = nn.Linear(
                self.vector_channels, self.vector_channels, bias=False
            )
        else:
            self.vector_gate = None
            self.vector_source = None
            self.vector_mix = None

        update_input_dim = hidden_dim + self.vector_channels
        self.message_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.update = nn.Sequential(
            nn.Linear(update_input_dim, message_dim),
            make_activation(),
            nn.Linear(message_dim, hidden_dim),
        )

    def _edge_features(
        self,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.linalg.norm(vectors, dim=-1)
        safe_distances = torch.clamp(distances, min=1.0e-12)
        unit_vectors = vectors / safe_distances.unsqueeze(-1)
        radial_basis = chebyshev_basis(
            distances, self.cutoff, self.basis_size
        )
        pair_index = types[center] * self.n_elements + types[neighbor]
        pair_features = self.pair_embedding(pair_index)
        edge_hidden = self.edge_mlp(
            torch.cat((radial_basis, pair_features), dim=-1)
        )
        cutoff = torch.where(
            distances < self.cutoff,
            0.5 * torch.cos(torch.pi * distances / self.cutoff) + 0.5,
            torch.zeros_like(distances),
        )
        return edge_hidden, cutoff, unit_vectors

    def forward(
        self,
        hidden: torch.Tensor,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        n_atoms = hidden.shape[0]
        normalized = self.input_norm(hidden)

        if center.numel() == 0:
            # Keep every parameter in the graph so DDP can use its faster
            # static reduction path even for structures with no edges.
            parameter_dependency = sum(
                parameter.reshape(-1)[0] * 0.0
                for parameter in self.parameters()
            )
            scalar_message = hidden.new_zeros(
                (n_atoms, self.hidden_dim)
            ) + parameter_dependency
            vector_invariants = hidden.new_zeros(
                (n_atoms, self.vector_channels)
            ) + parameter_dependency
        else:
            edge_hidden, cutoff, unit_vectors = self._edge_features(
                types, center, neighbor, vectors
            )
            gates = torch.sigmoid(self.group_gate(edge_hidden))
            gates = gates * cutoff.unsqueeze(-1)

            values = self.value_projection(normalized)[neighbor]
            grouped_values = values.reshape(
                -1, self.message_groups, self.group_width
            )
            edge_values = (grouped_values * gates.unsqueeze(-1)).reshape(
                -1, self.hidden_dim
            )
            scalar_message = hidden.new_zeros((n_atoms, self.hidden_dim))
            scalar_message.scatter_add_(
                0, center.unsqueeze(-1).expand_as(edge_values), edge_values
            )

            if self.vector_channels:
                vector_gates = torch.sigmoid(self.vector_gate(edge_hidden))
                vector_gates = vector_gates * cutoff.unsqueeze(-1)
                source = self.vector_source(normalized)[neighbor]
                edge_vectors = (
                    vector_gates * source
                ).unsqueeze(-1) * unit_vectors.unsqueeze(1)
                vector_sum = hidden.new_zeros(
                    (n_atoms, self.vector_channels, 3)
                )
                vector_sum.scatter_add_(
                    0,
                    center.view(-1, 1, 1).expand_as(edge_vectors),
                    edge_vectors,
                )
                # Mix the channel axis independently for x/y/z.  This commutes
                # with spatial rotations and is therefore equivariant.
                mixed = self.vector_mix(vector_sum.transpose(1, 2)).transpose(1, 2)
                vector_invariants = mixed.square().sum(dim=-1)
            else:
                vector_invariants = hidden.new_zeros((n_atoms, 0))

        update_input = torch.cat(
            (self.message_norm(scalar_message), vector_invariants), dim=-1
        )
        return hidden + self.update(update_input)
