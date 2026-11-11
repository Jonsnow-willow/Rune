import re

import torch
import torch.nn as nn

from .descriptor import Descriptor
from .zbl import ZBLRepulsion


def _structure_ids(n_atoms_per_structure, device):
    counts = n_atoms_per_structure.to(device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)


def _apply_structure_deformation(vectors, deformation, atom_structure_ids):
    per_atom_deformation = deformation[atom_structure_ids]
    if vectors.ndim == 2:
        return torch.einsum("ni,nji->nj", vectors, per_atom_deformation)
    if vectors.ndim == 3:
        return torch.einsum("nmi,nji->nmj", vectors, per_atom_deformation)
    raise ValueError(f"Expected vectors with ndim 2 or 3, got {vectors.ndim}.")


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


class NEP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = dict(params)
        self.descriptor = Descriptor(self.params)

        self.n_desc_radial = params["n_desc_radial"]
        self.n_desc_angular = params["n_desc_angular"]
        self.L_max = params["l_max"]
        self.l_max_4body = params.get("l_max_4body", 2)
        self.l_max_5body = params.get("l_max_5body", 0)
        self.has_q_222 = int(params.get("has_q_222", self.l_max_4body > 0))
        self.has_q_1111 = int(params.get("has_q_1111", self.l_max_5body > 0))
        self.has_q_112 = int(params.get("has_q_112", 0))
        self.has_q_123 = int(params.get("has_q_123", 0))
        self.has_q_233 = int(params.get("has_q_233", 0))
        self.has_q_134 = int(params.get("has_q_134", 0))
        self.num_L = self.descriptor.angular.num_L
        self.elements = params["elements"]
        self.n_elements = len(self.elements)

        input_dim = self.n_desc_radial + self.n_desc_angular * self.num_L
        hidden_dims = params["hidden_dims"]

        self.element_mlps = nn.ModuleDict()
        for _, element in enumerate(self.elements):
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.Tanh())
            for j in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[j - 1], hidden_dims[j]))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dims[-1], 1, bias=False))
            self.element_mlps[element] = nn.Sequential(*layers)

        self._init_descriptor_scaler(input_dim)
        self._configure_zbl()
        self.shared_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        compute_forces = batch.get("compute_forces", True)
        compute_virial = batch.get("compute_virial", True)
        state = self.descriptor.compute_state(batch, track_geometry=False)
        need_derivatives = compute_forces or compute_virial

        edge = state.angular
        if need_derivatives:
            with torch.enable_grad():
                descriptors = state.values
                if not descriptors.requires_grad:
                    descriptors = descriptors.detach().requires_grad_(True)
                edge_vectors = edge.vectors.detach().requires_grad_(True)
                e_atom = self._atomic_energies(
                    descriptors, batch["types"], edge.center, edge.neighbor, edge_vectors
                )
                # Independent inputs retain the two terms of dE/dR exactly once.
                descriptor_gradient, edge_gradient = torch.autograd.grad(
                    e_atom.sum(),
                    (descriptors, edge_vectors),
                    create_graph=self.training,
                    retain_graph=self.training,
                    allow_unused=True,
                )
            if descriptor_gradient is None:
                descriptor_gradient = torch.zeros_like(descriptors)
            descriptor_forces, descriptor_virial = self.descriptor.contract_forces(
                descriptor_gradient, state, compute_virial=compute_virial
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
            e_atom = self._atomic_energies(
                state.values, batch["types"], edge.center, edge.neighbor, edge.vectors
            )
            forces = None
            atomic_virial = None

        structure_ids = _structure_ids(
            batch["n_atoms_per_structure"],
            device=e_atom.device,
        )
        n_structures = int(batch["n_atoms_per_structure"].numel())
        e_total = e_atom.new_zeros(n_structures)
        e_total.scatter_add_(0, structure_ids, e_atom)

        result = {
            "energies": e_atom,
            "energy": e_total,
        }
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

    def _atomic_energies(self, descriptors, types, center, neighbor, edge_vectors):
        """Evaluate learned and optional ZBL per-atom energies."""
        scaled = descriptors * self.q_scaler
        e_atom = scaled.new_zeros(scaled.shape[0])
        for type_index, element in enumerate(self.elements):
            mask = types == type_index
            if mask.any():
                e_atom[mask] = (
                    self.element_mlps[element](scaled[mask]).squeeze(-1)
                    - self.shared_bias
                )
        if self.zbl is not None:
            e_atom = e_atom + self.zbl(types, center, neighbor, edge_vectors)
        return e_atom

    def _forward_autograd_reference(self, batch):
        """Reference force/virial path used by numerical tests and debugging."""
        positions = batch["positions"]
        types = batch["types"]

        if not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)
            batch = batch.copy()
            batch["positions"] = positions
        else:
            batch = batch.copy()

        strain = None
        descriptor_batch = batch
        if batch.get("compute_virial", True):
            n_structures = int(batch["n_atoms_per_structure"].numel())
            strain = torch.zeros(
                (n_structures, 3, 3),
                device=positions.device,
                dtype=positions.dtype,
                requires_grad=True,
            )
            deformation = (
                torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(
                    0
                )
                + strain
            )
            atom_structure_ids = _structure_ids(
                batch["n_atoms_per_structure"],
                device=positions.device,
            )
            descriptor_batch = batch.copy()
            descriptor_batch["positions"] = _apply_structure_deformation(
                positions,
                deformation,
                atom_structure_ids,
            )
            descriptor_batch["radial_offsets"] = _apply_structure_deformation(
                batch["radial_offsets"].to(
                    device=positions.device, dtype=positions.dtype
                ),
                deformation,
                atom_structure_ids,
            )
            descriptor_batch["angular_offsets"] = _apply_structure_deformation(
                batch["angular_offsets"].to(
                    device=positions.device, dtype=positions.dtype
                ),
                deformation,
                atom_structure_ids,
            )

        descriptor_state = self.descriptor.compute_state(
            descriptor_batch,
            track_geometry=True,
        )
        edge = descriptor_state.angular
        e_atom = self._atomic_energies(
            descriptor_state.values, types, edge.center, edge.neighbor, edge.vectors
        )

        n_atoms_per_structure = batch["n_atoms_per_structure"].detach().cpu().tolist()
        atom_idx = 0
        e_total_list = []

        forces = None
        total_energy = e_atom.sum()
        if batch.get("compute_forces", True):
            position_gradient = torch.autograd.grad(
                outputs=total_energy,
                inputs=positions,
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )[0]
            forces = (
                -position_gradient
                if position_gradient is not None
                else torch.zeros_like(positions)
            )

        for n_atoms_in_structure in n_atoms_per_structure:
            e_structure = e_atom[atom_idx : atom_idx + n_atoms_in_structure].sum()
            e_total_list.append(e_structure)
            atom_idx += n_atoms_in_structure

        e_total = torch.stack(e_total_list)
        virial = None
        if strain is not None:
            virial = torch.autograd.grad(
                outputs=e_total.sum(),
                inputs=strain,
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if virial is None:
                virial = torch.zeros_like(strain)
            virial = -virial.reshape(strain.shape[0], -1)

        result = {
            "energies": e_atom,
            "energy": e_total,
        }
        if forces is not None:
            result["forces"] = forces
        if virial is not None:
            result["virial"] = virial
        return result


    @classmethod
    def from_nep_txt(cls, filepath, device=None):
        def _floats(lines, start):
            for line in lines[start:]:
                for token in line.split():
                    yield float(token)

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip() != ""]

        idx = 0
        tokens = lines[idx].split()
        idx += 1
        header = tokens[0].lower()
        match = re.fullmatch(r"nep(\d+)(?:_(.*))?", header)
        if match is None:
            raise ValueError(
                "Invalid nep.txt: first line should start with nep4 or nep5."
            )
        version = int(match.group(1))
        suffix = match.group(2) or ""
        has_zbl = "zbl" in suffix.split("_")
        if version != 4:
            raise NotImplementedError("Only nep4 is supported by the RUNE loader.")
        num_types = int(tokens[1])
        elements = tokens[2:]
        if len(elements) != num_types:
            raise ValueError("Invalid nep.txt: element count mismatch.")

        zbl = None
        if has_zbl:
            tokens = lines[idx].split()
            idx += 1
            if tokens[0] != "zbl" or len(tokens) not in (3, 4):
                raise ValueError(
                    "Invalid nep.txt: zbl line not found after nep4_zbl header."
                )
            zbl = {
                "rc_inner": float(tokens[1]),
                "rc_outer": float(tokens[2]),
                "flexible": float(tokens[1]) == 0.0 and float(tokens[2]) == 0.0,
            }
            if len(tokens) == 4:
                zbl["typewise_cutoff_factor"] = float(tokens[3])
                raise NotImplementedError(
                    "Typewise ZBL cutoffs are not supported in RUNE."
                )

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "cutoff":
            raise ValueError("Invalid nep.txt: cutoff line not found.")
        if len(tokens) == 5:
            rcut_radial = float(tokens[1])
            rcut_angular = float(tokens[2])
            max_neighbors_radial = int(tokens[3])
            max_neighbors_angular = int(tokens[4])
        else:
            # cutoff rc_r1 rc_a1 rc_r2 rc_a2 ... MN_radial MN_angular
            expected = num_types * 2 + 3
            if len(tokens) != expected:
                raise ValueError("Invalid nep.txt: cutoff line length mismatch.")
            rc_radial = [float(tokens[1 + i * 2]) for i in range(num_types)]
            rc_angular = [float(tokens[2 + i * 2]) for i in range(num_types)]
            if len(set(rc_radial)) != 1 or len(set(rc_angular)) != 1:
                raise NotImplementedError("Per-type cutoffs are not supported in RUNE.")
            rcut_radial = rc_radial[0]
            rcut_angular = rc_angular[0]
            max_neighbors_radial = int(tokens[-2])
            max_neighbors_angular = int(tokens[-1])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "n_max":
            raise ValueError("Invalid nep.txt: n_max line not found.")
        n_max_radial = int(tokens[1])
        n_max_angular = int(tokens[2])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "basis_size":
            raise ValueError("Invalid nep.txt: basis_size line not found.")
        basis_size_radial = int(tokens[1])
        basis_size_angular = int(tokens[2])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "l_max":
            raise ValueError("Invalid nep.txt: l_max line not found.")
        l_max_values = [int(value) for value in tokens[1:]]
        if not 1 <= len(l_max_values) <= 7:
            raise ValueError("Invalid nep.txt: l_max should contain 1 to 7 values.")
        l_max_values.extend([1, 0, 0, 0, 0, 0][len(l_max_values) - 1 :])
        l_max, has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233, has_q_134 = (
            l_max_values
        )
        l_max_4body = 2 if has_q_222 > 0 else 0
        l_max_5body = 1 if has_q_1111 > 0 else 0

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "ANN" or len(tokens) != 3:
            raise ValueError("Invalid nep.txt: expected ANN width 0.")
        num_neurons = int(tokens[1])
        if not 1 <= num_neurons <= 200:
            raise ValueError("Invalid nep.txt: ANN width must be in [1, 200].")
        if int(tokens[2]) != 0:
            raise ValueError(
                "RUNE only supports single-hidden-layer NEP text models (ANN width 0)."
            )

        n_desc_radial = n_max_radial + 1
        n_desc_angular = n_max_angular + 1
        k_max_radial = basis_size_radial + 1
        k_max_angular = basis_size_angular + 1

        params = {
            "elements": elements,
            "rcut_radial": rcut_radial,
            "rcut_angular": rcut_angular,
            "n_desc_radial": n_desc_radial,
            "n_desc_angular": n_desc_angular,
            "k_max_radial": k_max_radial,
            "k_max_angular": k_max_angular,
            "l_max": l_max,
            "l_max_4body": l_max_4body,
            "l_max_5body": l_max_5body,
            "has_q_222": int(has_q_222 > 0),
            "has_q_1111": int(has_q_1111 > 0),
            "has_q_112": int(has_q_112 > 0),
            "has_q_123": int(has_q_123 > 0),
            "has_q_233": int(has_q_233 > 0),
            "has_q_134": int(has_q_134 > 0),
            "max_neighbors_radial": max_neighbors_radial,
            "max_neighbors_angular": max_neighbors_angular,
            "hidden_dims": [num_neurons],
            "n_types": num_types,
        }
        if zbl is not None and not zbl["flexible"]:
            params["zbl"] = zbl

        model = cls(params)
        num_L = model.num_L
        dim = n_desc_radial + n_desc_angular * num_L

        stream = _floats(lines, idx)

        with torch.no_grad():
            for t, element in enumerate(elements):
                w0 = torch.tensor(
                    [next(stream) for _ in range(num_neurons * dim)],
                    dtype=torch.float32,
                ).reshape(num_neurons, dim)
                b0 = torch.tensor(
                    [next(stream) for _ in range(num_neurons)], dtype=torch.float32
                )
                w1 = torch.tensor(
                    [next(stream) for _ in range(num_neurons)], dtype=torch.float32
                )

                mlp = model.element_mlps[element]
                mlp[0].weight.copy_(w0)
                mlp[0].bias.copy_(-b0)
                mlp[-1].weight.copy_(w1.view(1, -1))

            shared_bias = torch.tensor(next(stream), dtype=torch.float32)
            model.shared_bias.copy_(shared_bias)

            c_radial = torch.empty(
                num_types, num_types, n_desc_radial, k_max_radial, dtype=torch.float32
            )
            for n in range(n_desc_radial):
                for k in range(k_max_radial):
                    for t1 in range(num_types):
                        for t2 in range(num_types):
                            c_radial[t1, t2, n, k] = next(stream)
            model.descriptor.radial.c_table.copy_(c_radial)

            c_angular = torch.empty(
                num_types, num_types, n_desc_angular, k_max_angular, dtype=torch.float32
            )
            for n in range(n_desc_angular):
                for k in range(k_max_angular):
                    for t1 in range(num_types):
                        for t2 in range(num_types):
                            c_angular[t1, t2, n, k] = next(stream)
            model.descriptor.angular.c_table.copy_(c_angular)

            q_scaler = torch.tensor(
                [next(stream) for _ in range(dim)], dtype=torch.float32
            )
            model.q_scaler.copy_(q_scaler)

            if zbl is not None and zbl["flexible"]:
                zbl_parameters = torch.tensor(
                    list(stream),
                    dtype=torch.float32,
                )
                zbl["parameters"] = zbl_parameters.detach().cpu().reshape(-1).tolist()
                model.params["zbl"] = zbl
                model._configure_zbl()

        if device is not None:
            model.to(device)

        return model

    def save_nep_txt(self, filepath):
        if len(self.params["hidden_dims"]) != 1:
            raise ValueError(
                "RUNE nep.txt export requires exactly one NEP hidden layer."
            )
        if self.params["max_neighbors_radial"] > 819:
            raise ValueError("GPUMD nep.txt export requires max_neighbors_radial <= 819.")
        with open(filepath, "w") as f:
            zbl = self.zbl
            header = "nep4_zbl" if zbl is not None else "nep4"
            f.write(f"{header} {len(self.elements)} " + " ".join(self.elements) + "\n")
            if zbl is not None:
                f.write(f"zbl {zbl.rc_inner} {zbl.rc_outer}\n")
            f.write(
                f"cutoff {self.params['rcut_radial']} "
                f"{self.params['rcut_angular']} "
                f"{self.params['max_neighbors_radial']} "
                f"{self.params['max_neighbors_angular']}\n"
            )
            f.write(
                f"n_max {int(self.params['n_desc_radial']) - 1} "
                f"{int(self.params['n_desc_angular']) - 1}\n"
            )
            f.write(
                f"basis_size {int(self.params['k_max_radial']) - 1} "
                f"{int(self.params['k_max_angular']) - 1}\n"
            )
            l_max_values = [
                int(self.L_max),
                2 if self.has_q_222 else 0,
                self.has_q_1111,
                self.has_q_112,
                self.has_q_123,
                self.has_q_233,
                self.has_q_134,
            ]
            while len(l_max_values) > 3 and l_max_values[-1] == 0:
                l_max_values.pop()
            f.write("l_max " + " ".join(str(value) for value in l_max_values) + "\n")
            f.write(f"ANN {int(self.params['hidden_dims'][0])} 0\n")

            for element in self.elements:
                mlp = self.element_mlps[element]
                for idx, layer in enumerate(mlp):
                    if isinstance(layer, nn.Linear):
                        weights = layer.weight.data.flatten().tolist()
                        for val in weights:
                            f.write(f"{val:18.10e}\n")

                        if layer.bias is not None:
                            bias = layer.bias.data
                            if idx == 0:
                                bias = -bias
                            bias = bias.flatten().tolist()
                            for val in bias:
                                f.write(f"{val:18.10e}\n")

            bias_value = self.shared_bias.data.item()
            f.write(f"{bias_value:18.10e}\n")

            n_max_radial = int(self.params["n_desc_radial"])
            n_max_angular = int(self.params["n_desc_angular"])
            k_max_radial = int(self.params["k_max_radial"])
            k_max_angular = int(self.params["k_max_angular"])
            n_types = len(self.elements)

            radial_params = self.descriptor.radial.c_table.data
            for n in range(n_max_radial):
                for k in range(k_max_radial):
                    for t1 in range(n_types):
                        for t2 in range(n_types):
                            val = radial_params[t1, t2, n, k].item()
                            f.write(f"{val:18.10e}\n")

            angular_params = self.descriptor.angular.c_table.data
            for n in range(n_max_angular):
                for k in range(k_max_angular):
                    for t1 in range(n_types):
                        for t2 in range(n_types):
                            val = angular_params[t1, t2, n, k].item()
                            f.write(f"{val:18.10e}\n")

            scaler = self.q_scaler.detach().cpu().reshape(-1)
            for val in scaler.tolist():
                f.write(f"{val:18.10e}\n")

            if zbl is not None and zbl.flexible:
                for val in zbl.pair_parameters.detach().cpu().reshape(-1).tolist():
                    f.write(f"{val:18.10e}\n")

    def _init_descriptor_scaler(self, input_dim):
        scaler_tensor = torch.ones(input_dim, dtype=torch.float32)
        self.register_buffer("q_scaler", scaler_tensor)

    def _configure_zbl(self):
        self.zbl = (
            ZBLRepulsion(self.elements, self.params["zbl"])
            if self.params.get("zbl") is not None
            else None
        )
        if self.zbl is not None and not self.zbl.max_rc_outer <= self.params["rcut_angular"]:
            raise ValueError(
                "ZBL rc_outer cannot exceed rcut_angular because NEP "
                "evaluates ZBL on the angular neighbor list."
            )

    def compute_descriptor_scaler(self, dataloader, device=None, max_batches=None):
        """
        Estimate scaler = 1 / (max - min) for each descriptor dimension.
        """
        q_min, q_max = self.compute_descriptor_min_max(
            dataloader, device=device, max_batches=max_batches
        )
        if q_min is not None:
            self.set_descriptor_scaler_from_min_max(q_min, q_max)
        return self.q_scaler.detach().cpu().clone()

    def compute_descriptor_min_max(self, dataloader, device=None, max_batches=None):
        if device is None:
            device = next(self.parameters()).device
        prev_mode = self.training
        self.eval()
        q_min = None
        q_max = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                descriptors = self.descriptor(batch)
                g_total = self._combine_descriptors(descriptors)
                if g_total.numel() == 0:
                    continue
                batch_min = torch.amin(g_total, dim=0)
                batch_max = torch.amax(g_total, dim=0)
                if q_min is None:
                    q_min = batch_min
                    q_max = batch_max
                else:
                    q_min = torch.minimum(q_min, batch_min)
                    q_max = torch.maximum(q_max, batch_max)
        self.train(prev_mode)
        return q_min, q_max

    def set_descriptor_scaler_from_min_max(self, q_min, q_max):
        if q_min is not None:
            diff = q_max - q_min
            scaler = torch.ones_like(diff)
            valid = diff > 1.0e-12
            scaler[valid] = 1.0 / diff[valid]
            tensor = scaler.flatten()
            with torch.no_grad():
                self.q_scaler.copy_(tensor)
        return self.q_scaler.detach().cpu().clone()

    def _combine_descriptors(self, descriptors):
        g_radial = descriptors["g_radial"]
        g_angular = descriptors["g_angular"]
        n_atoms = g_radial.shape[0]
        if n_atoms == 0:
            return g_radial.new_zeros(
                (0, self.n_desc_radial + self.n_desc_angular * self.num_L)
            )
        g_angular_flat = g_angular.permute(0, 2, 1).reshape(n_atoms, -1)
        return torch.cat([g_radial, g_angular_flat], dim=-1)
