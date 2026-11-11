"""Mobius descriptor-native message-passing machine-learning potential."""

from __future__ import annotations

import torch
import torch.nn as nn
from ase.data import atomic_numbers

from .descriptor import Descriptor
from .equivariant import DirectionEncoder, EquivariantBlock
from .layers import ElementwiseLinearHead, RMSNorm, ResidualBlock
from .message import MessageBlock
from .zbl import ZBLRepulsion


def _structure_ids(
    n_atoms_per_structure: torch.Tensor, device: torch.device
) -> torch.Tensor:
    counts = n_atoms_per_structure.to(device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)


def _accumulate_direct_edge_derivatives(
    n_atoms: int,
    center: torch.Tensor,
    neighbor: torch.Tensor,
    vectors: torch.Tensor,
    gradient: torch.Tensor | None,
    *,
    compute_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert ``dE/d r_ij`` into atom forces and atomic virials."""
    forces = vectors.new_zeros((n_atoms, 3))
    virial = vectors.new_zeros((n_atoms, 9)) if compute_virial else None
    if gradient is None or center.numel() == 0:
        return forces, virial
    forces.scatter_add_(0, center.unsqueeze(-1).expand_as(gradient), gradient)
    forces.scatter_add_(0, neighbor.unsqueeze(-1).expand_as(gradient), -gradient)
    if virial is not None:
        pair_virial = -(gradient.unsqueeze(-1) * vectors.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(
            0, neighbor.unsqueeze(-1).expand_as(pair_virial), pair_virial
        )
    return forces, virial


class Mobius(nn.Module):
    r"""Compact-descriptor graph potential trained by RUNE.

    Mobius uses its own invariant descriptor ``q_i``, a shared element-conditioned
    residual representation and one linear energy head per central element.
    Scalar messages contract their transient direction branch immediately.
    Equivariant messages additionally retain angular states across layers.

    The forward result uses the RUNE model contract:

    ``energies``
        Per-atom energies, shape ``[N]``.
    ``energy``
        Per-structure total energies, shape ``[B]``.
    ``forces``
        Conservative forces, shape ``[N, 3]`` when requested.
    ``virial``
        Per-structure virials, shape ``[B, 9]`` when requested.

    Force construction retains all coordinate paths:

    ``dE/dR = E_q q_R + E_s_direct s_R + E_r_edge r_edge_R``.

    The optional middle term is the independent equivariant initialization
    path. It shares the analytical angular Jacobian with the q path. The last
    term differentiates the edge gates and geometry. All force terms remain
    differentiable with respect to model parameters.
    """

    model_family = "mobius"

    def __init__(self, params: dict):
        super().__init__()
        self.params = dict(params)
        self.elements = list(self.params["elements"])
        if not self.elements:
            raise ValueError("Mobius requires a non-empty elements list.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("Mobius elements must not contain duplicates.")
        invalid_elements = [
            element for element in self.elements if element not in atomic_numbers
        ]
        if invalid_elements:
            raise ValueError(f"Unknown chemical element symbols: {invalid_elements}.")
        self.n_elements = len(self.elements)
        configured_types = int(self.params.get("n_types", self.n_elements))
        if configured_types != self.n_elements:
            raise ValueError(
                "params['n_types'] must match the number of entries in "
                "params['elements']."
            )
        self.params["n_types"] = self.n_elements

        self.hidden_dim = int(self.params.get("hidden_dim", 96))
        self.blocks = int(self.params.get("blocks", 3))
        self.ffn_dim = int(self.params.get("ffn_dim", 2 * self.hidden_dim))
        self.activation = str(self.params.get("activation", "silu"))
        self.message_layers = int(self.params.get("message_layers", 2))
        self.message_mode = self.params.get("message_mode", "scalar")
        if self.message_mode not in ("scalar", "equivariant"):
            raise ValueError("message_mode must be 'scalar' or 'equivariant'.")
        self.equivariant_channels = int(self.params.get("equivariant_channels", 8))
        self.equivariant_l_max = self.params.get("equivariant_l_max")
        if self.equivariant_l_max is None:
            self.equivariant_l_max = int(self.params["l_max"])
        if self.message_mode == "equivariant":
            if self.message_layers < 1 or self.equivariant_channels < 1:
                raise ValueError("Equivariant messages require positive layers and channels.")
            if not 1 <= self.equivariant_l_max <= min(int(self.params["l_max"]), 8):
                raise ValueError("equivariant_l_max must be in 1..min(l_max, 8).")
            self.params.update(
                message_mode=self.message_mode,
                equivariant_channels=self.equivariant_channels,
                equivariant_l_max=self.equivariant_l_max,
            )
        self.message_dim = int(self.params.get("message_dim", self.hidden_dim))
        self.pair_dim = int(self.params.get("pair_dim", 16))
        self.message_groups = int(self.params.get("message_groups", 8))
        self.vector_channels = int(self.params.get("vector_channels", 8))
        self.norm_eps = float(self.params.get("norm_eps", 1.0e-8))
        if (
            min(
                self.hidden_dim,
                self.ffn_dim,
                self.message_dim,
                self.pair_dim,
            )
            <= 0
        ):
            raise ValueError(
                "Mobius hidden, FFN, message, and pair dimensions must be positive."
            )
        if self.blocks < 0 or self.message_layers < 0:
            raise ValueError("Mobius blocks and message_layers cannot be negative.")
        if self.message_layers and self.hidden_dim % self.message_groups != 0:
            raise ValueError("hidden_dim must be divisible by message_groups.")

        # Record resolved defaults so checkpoints are self-describing.
        self.params.update(
            {
                "model": "mobius",
                "hidden_dim": self.hidden_dim,
                "blocks": self.blocks,
                "ffn_dim": self.ffn_dim,
                "activation": self.activation,
                "message_layers": self.message_layers,
                "message_dim": self.message_dim,
                "pair_dim": self.pair_dim,
                "message_groups": self.message_groups,
                "vector_channels": self.vector_channels,
                "norm_eps": self.norm_eps,
            }
        )

        self.descriptor = Descriptor(self.params)
        self.n_desc_radial = int(self.params["n_desc_radial"])
        self.n_desc_angular = int(self.params["n_desc_angular"])
        self.num_L = int(self.descriptor.angular.num_L)
        self.input_dim = self.n_desc_radial + self.n_desc_angular * self.num_L

        scaler = self.params.get("descriptor_scaler")
        if scaler is None:
            scaler_tensor = torch.ones(self.input_dim, dtype=torch.float32)
        else:
            scaler_tensor = torch.as_tensor(scaler, dtype=torch.float32).reshape(-1)
            if scaler_tensor.numel() != self.input_dim:
                raise ValueError(
                    f"descriptor_scaler has {scaler_tensor.numel()} values; "
                    f"Mobius expects {self.input_dim}."
                )
        self.register_buffer("q_scaler", scaler_tensor.clone())
        self.params["descriptor_scaler"] = scaler_tensor.tolist()

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.element_embedding = nn.Embedding(self.n_elements, self.hidden_dim)
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(
                self.hidden_dim,
                self.ffn_dim,
                self.activation,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.blocks)
        )
        # Initialize shared readout weights before variant-specific modules so
        # the same seed gives all three variants the same common parameters.
        output_norm = RMSNorm(self.hidden_dim, eps=self.norm_eps)
        energy_heads = ElementwiseLinearHead(self.n_elements, self.hidden_dim)

        message_basis_size = int(
            self.params.get("message_basis_size", self.params["k_max_radial"])
        )
        message_cutoff = float(
            self.params.get("message_cutoff", self.params["rcut_radial"])
        )
        if message_cutoff > float(self.params["rcut_radial"]):
            raise ValueError(
                "message_cutoff cannot exceed rcut_radial because Mobius uses "
                "the radial neighbor list for message edges."
            )
        self.params["message_basis_size"] = message_basis_size
        self.params["message_cutoff"] = message_cutoff
        message_params = dict(
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            pair_dim=self.pair_dim,
            message_groups=self.message_groups,
            n_elements=self.n_elements,
            basis_size=message_basis_size,
            cutoff=message_cutoff,
            activation=self.activation,
            norm_eps=self.norm_eps,
        )
        if self.message_mode == "equivariant":
            self.direction_encoder = DirectionEncoder(
                self.n_desc_angular, self.equivariant_channels, self.equivariant_l_max
            )
            self.message_blocks = nn.ModuleList(
                EquivariantBlock(
                    **message_params,
                    equivariant_channels=self.equivariant_channels,
                    l_max=self.equivariant_l_max,
                )
                for _ in range(self.message_layers)
            )
        else:
            self.message_blocks = nn.ModuleList(
                MessageBlock(**message_params, vector_channels=self.vector_channels)
                for _ in range(self.message_layers)
            )
        # Preserve registration order for historical optimizer-state resumes.
        self.output_norm = output_norm
        # Public name: these are genuinely element-specific, but only the
        # lightweight readout is separate; the expensive representation is shared.
        self.energy_heads = energy_heads

        self.zbl = (
            ZBLRepulsion(self.elements, self.params["zbl"])
            if self.params.get("zbl") is not None
            else None
        )
        if self.zbl is not None and self.zbl.max_rc_outer > float(
            self.params["rcut_radial"]
        ):
            raise ValueError(
                "ZBL rc_outer cannot exceed rcut_radial because Mobius "
                "evaluates ZBL on the radial neighbor list."
            )

    def encode(
        self,
        descriptors: torch.Tensor,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        edge_vectors: torch.Tensor,
        *,
        moments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode descriptors into the final shared Mobius atom states."""
        hidden = self.input_projection(descriptors * self.q_scaler)
        hidden = hidden + self.element_embedding(types)
        for block in self.residual_blocks:
            hidden = block(hidden)
        if self.message_mode == "equivariant":
            if moments is None:
                raise ValueError("Equivariant encoding requires angular moments.")
            angular = self.descriptor.angular
            direction = self.direction_encoder(
                angular.normalized_moments(moments, self.equivariant_l_max)
            )
            harmonics = angular.normalized_basis(edge_vectors, self.equivariant_l_max)
            for message_block in self.message_blocks:
                hidden, direction = message_block(
                    hidden, direction, types, center, neighbor, edge_vectors, harmonics
                )
        else:
            for message_block in self.message_blocks:
                hidden = message_block(hidden, types, center, neighbor, edge_vectors)
        return hidden

    def _atomic_energies(
        self,
        descriptors: torch.Tensor,
        types: torch.Tensor,
        center: torch.Tensor,
        neighbor: torch.Tensor,
        edge_vectors: torch.Tensor,
        *,
        moments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate learned and optional ZBL per-atom energies."""
        hidden = self.encode(
            descriptors, types, center, neighbor, edge_vectors, moments=moments
        )
        energies = self.energy_heads(self.output_norm(hidden), types)
        if self.zbl is not None:
            energies = energies + self.zbl(types, center, neighbor, edge_vectors)
        return energies

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        compute_forces = bool(batch.get("compute_forces", True))
        compute_virial = bool(batch.get("compute_virial", True))
        need_derivatives = compute_forces or compute_virial
        state = self.descriptor.compute_state(batch, track_geometry=False)

        edge = state.radial
        if need_derivatives:
            with torch.enable_grad():
                descriptors = state.values
                if not descriptors.requires_grad:
                    descriptors = descriptors.detach().requires_grad_(True)
                edge_vectors = edge.vectors.detach().requires_grad_(True)
                moments = None
                inputs = (descriptors, edge_vectors)
                if self.message_mode == "equivariant":
                    # q uses the original moments. This sibling clone supplies
                    # an independent partial derivative for the direction path
                    # without cutting its parameter gradient to c_table.
                    moments = state.angular.moments.clone()
                    if not moments.requires_grad:
                        moments.requires_grad_(True)
                    inputs = (*inputs, moments)
                energies = self._atomic_energies(
                    descriptors,
                    batch["types"],
                    edge.center,
                    edge.neighbor,
                    edge_vectors,
                    moments=moments,
                )
                gradients = torch.autograd.grad(
                    energies.sum(),
                    inputs,
                    create_graph=self.training,
                    retain_graph=self.training,
                    allow_unused=True,
                )
                descriptor_gradient, edge_gradient = gradients[:2]
                moment_gradient = gradients[2] if moments is not None else None
            if descriptor_gradient is None:
                descriptor_gradient = torch.zeros_like(descriptors)
            descriptor_forces, descriptor_virial = self.descriptor.contract_forces(
                descriptor_gradient,
                state,
                compute_virial=compute_virial,
                moment_gradient=moment_gradient,
            )
            edge_forces, edge_virial = _accumulate_direct_edge_derivatives(
                batch["types"].shape[0],
                edge.center,
                edge.neighbor,
                edge_vectors,
                edge_gradient,
                compute_virial=compute_virial,
            )
            forces = descriptor_forces + edge_forces if compute_forces else None
            atomic_virial = descriptor_virial + edge_virial if compute_virial else None
        else:
            energies = self._atomic_energies(
                state.values,
                batch["types"],
                edge.center,
                edge.neighbor,
                edge.vectors,
                moments=state.angular.moments,
            )
            forces = None
            atomic_virial = None

        structure_ids = _structure_ids(batch["n_atoms_per_structure"], energies.device)
        n_structures = int(batch["n_atoms_per_structure"].numel())
        total_energy = energies.new_zeros(n_structures)
        total_energy.scatter_add_(0, structure_ids, energies)

        result = {"energies": energies, "energy": total_energy}
        if forces is not None:
            result["forces"] = forces
        if atomic_virial is not None:
            virial = atomic_virial.new_zeros((n_structures, 9))
            virial.scatter_add_(
                0,
                structure_ids.unsqueeze(-1).expand_as(atomic_virial),
                atomic_virial,
            )
            result["virial"] = virial
        return result

    def compute_descriptor_scaler(
        self, dataloader, device=None, max_batches: int | None = None
    ) -> torch.Tensor:
        """Estimate and store ``1 / (q_max - q_min)`` from a data loader."""
        q_min, q_max = self.compute_descriptor_min_max(
            dataloader, device=device, max_batches=max_batches
        )
        if q_min is not None:
            self.set_descriptor_scaler_from_min_max(q_min, q_max)
        return self.q_scaler.detach().cpu().clone()

    def compute_descriptor_min_max(
        self, dataloader, device=None, max_batches: int | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return componentwise raw descriptor extrema."""
        if device is None:
            device = next(self.parameters()).device
        previous_mode = self.training
        self.eval()
        q_min = q_max = None
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    moved = {
                        key: value.to(device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in batch.items()
                    }
                    values = self.descriptor.compute_state(
                        moved, track_geometry=False
                    ).values
                    if values.numel() == 0:
                        continue
                    current_min = torch.amin(values, dim=0)
                    current_max = torch.amax(values, dim=0)
                    q_min = (
                        current_min
                        if q_min is None
                        else torch.minimum(q_min, current_min)
                    )
                    q_max = (
                        current_max
                        if q_max is None
                        else torch.maximum(q_max, current_max)
                    )
        finally:
            self.train(previous_mode)
        return q_min, q_max

    def set_descriptor_scaler_from_min_max(
        self, q_min: torch.Tensor, q_max: torch.Tensor
    ) -> torch.Tensor:
        """Set the persistent descriptor scaler from componentwise extrema."""
        q_min = torch.as_tensor(
            q_min, device=self.q_scaler.device, dtype=self.q_scaler.dtype
        ).reshape(-1)
        q_max = torch.as_tensor(
            q_max, device=self.q_scaler.device, dtype=self.q_scaler.dtype
        ).reshape(-1)
        if q_min.numel() != self.input_dim or q_max.numel() != self.input_dim:
            raise ValueError(
                f"Descriptor extrema must each contain {self.input_dim} values."
            )
        extent = q_max - q_min
        scaler = torch.ones_like(extent)
        valid = extent > 1.0e-12
        scaler[valid] = extent[valid].reciprocal()
        return self.set_descriptor_scaler(scaler)

    def set_descriptor_scaler(self, scaler: torch.Tensor) -> torch.Tensor:
        """Set and persist an explicitly supplied descriptor scale vector."""
        scaler = torch.as_tensor(
            scaler, device=self.q_scaler.device, dtype=self.q_scaler.dtype
        ).reshape(-1)
        if scaler.numel() != self.input_dim:
            raise ValueError(f"Descriptor scaler must contain {self.input_dim} values.")
        if not torch.isfinite(scaler).all():
            raise ValueError("Descriptor scaler must contain only finite values.")
        with torch.no_grad():
            self.q_scaler.copy_(scaler)
        self.params["descriptor_scaler"] = scaler.detach().cpu().tolist()
        return self.q_scaler.detach().cpu().clone()
