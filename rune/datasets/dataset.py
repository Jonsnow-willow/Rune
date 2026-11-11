import numpy as np
import torch
from ase.data import atomic_numbers, chemical_symbols
from torch.utils.data import Dataset
from .neighbor import build_neighbor_list


def collate_fn(batch):
    atom_offset = 0
    types_list = []
    positions_list = []
    radial_neighbors_list = []
    angular_neighbors_list = []
    radial_offsets_list = []
    angular_offsets_list = []

    n_atoms_per_structure = []

    energy_list = []
    is_energy = []
    forces_list = []
    virial_list = []
    is_virial = []

    for item in batch:
        types_list.append(item["types"])
        positions_list.append(item["positions"])
        n_atoms_per_structure.append(item["n_atoms"])

        nbs_rad = item["radial_neighbors"].clone()
        valid_rad = nbs_rad != -1
        nbs_rad[valid_rad] += atom_offset
        radial_neighbors_list.append(nbs_rad)
        radial_offsets_list.append(item["radial_offsets"])

        nbs_ang = item["angular_neighbors"].clone()
        valid_ang = nbs_ang != -1
        nbs_ang[valid_ang] += atom_offset
        angular_neighbors_list.append(nbs_ang)
        angular_offsets_list.append(item["angular_offsets"])

        e = item.get("energy", None)
        energy_list.append(e)
        is_energy.append(e is not None)

        f = item.get("forces", None)
        forces_list.append(f)

        v = item.get("virial", None)
        virial_list.append(v)
        is_virial.append(v is not None)

        atom_offset += item["n_atoms"]

    types = torch.cat(types_list, dim=0)
    positions = torch.cat(positions_list, dim=0)
    radial_neighbors = torch.cat(radial_neighbors_list, dim=0)
    angular_neighbors = torch.cat(angular_neighbors_list, dim=0)
    radial_offsets = torch.cat(radial_offsets_list, dim=0)
    angular_offsets = torch.cat(angular_offsets_list, dim=0)

    result = {
        "types": types,  # [N_atoms_total]
        "positions": positions,  # [N_atoms_total, 3]
        "n_atoms_per_structure": torch.tensor(n_atoms_per_structure, dtype=torch.long),
        "batch_size": len(batch),  # int
        "radial_neighbors": radial_neighbors,  # [N_atoms_total, max_neighbors_radial]
        "angular_neighbors": angular_neighbors,  # [N_atoms_total, max_neighbors_angular]
        "radial_offsets": radial_offsets,  # [N_atoms_total, max_neighbors_radial, 3]
        "angular_offsets": angular_offsets,  # [N_atoms_total, max_neighbors_angular, 3]
    }

    if any(is_energy):
        result["energy"] = torch.stack(
            [
                e if e is not None else torch.tensor(0.0, dtype=torch.float32)
                for e in energy_list
            ]
        )
        result["is_energy"] = torch.tensor(is_energy, dtype=torch.bool)

    has_forces = [f is not None for f in forces_list]
    if all(has_forces):
        result["forces"] = torch.cat(forces_list, dim=0)
    elif any(has_forces):
        raise ValueError("Cannot collate a mixed batch with some missing force labels.")

    if any(is_virial):
        result["virial"] = torch.stack(
            [
                v if v is not None else torch.zeros(9, dtype=torch.float32)
                for v in virial_list
            ]
        )
        result["is_virial"] = torch.tensor(is_virial, dtype=torch.bool)

    return result


class StructureDataset(Dataset):
    def __init__(self, frames, model_params, require_forces=True):
        self.frames = frames
        self.model_params = model_params
        self.require_forces = require_forces
        self.elements = model_params["elements"]
        self.cutoff_radial = model_params["rcut_radial"]
        self.cutoff_angular = model_params["rcut_angular"]
        self.max_neighbors_radial = model_params["max_neighbors_radial"]
        self.max_neighbors_angular = model_params["max_neighbors_angular"]
        element_atomic_numbers = [atomic_numbers[element] for element in self.elements]
        self.z2id = {z: idx for idx, z in enumerate(element_atomic_numbers)}
        self.id2z = {idx: z for idx, z in enumerate(element_atomic_numbers)}

        dataset_atomic_numbers = sorted(
            {int(z) for atoms in self.frames for z in atoms.get_atomic_numbers()}
        )
        missing_atomic_numbers = [
            z for z in dataset_atomic_numbers if z not in self.z2id
        ]
        if missing_atomic_numbers:
            missing_symbols = [
                chemical_symbols[z] if z < len(chemical_symbols) else f"Z{z}"
                for z in missing_atomic_numbers
            ]
            raise ValueError(
                "Training set contains elements not listed in model_params['elements']. "
                f"Missing elements: {missing_symbols}. "
                f"Provided elements (type id order): {self.elements}."
            )

        self.data = [self._build_sample(atoms) for atoms in self.frames]

    def _build_sample(self, atoms):
        n_atoms = len(atoms)
        atomic_number_list = atoms.get_atomic_numbers()
        types = [self.z2id[int(z)] for z in atomic_number_list]
        types = torch.tensor(types, dtype=torch.long)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        (
            neighbors_rad,
            offsets_rad,
            neighbors_ang,
            offsets_ang,
        ) = build_neighbor_list(
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            cutoff_radial=self.cutoff_radial,
            cutoff_angular=self.cutoff_angular,
            max_neighbors_radial=self.max_neighbors_radial,
            max_neighbors_angular=self.max_neighbors_angular,
        )

        result = {
            "n_atoms": n_atoms,  # int
            "types": types,  # [N_atoms]
            "positions": positions,  # [N_atoms, 3]
            "radial_neighbors": torch.from_numpy(neighbors_rad).long(),
            "angular_neighbors": torch.from_numpy(neighbors_ang).long(),
            "radial_offsets": torch.from_numpy(offsets_rad).float(),
            "angular_offsets": torch.from_numpy(offsets_ang).float(),
        }

        energy = _get_label(atoms, "energy")
        if energy is not None:
            result["energy"] = torch.tensor(
                float(np.asarray(energy)), dtype=torch.float32
            )

        forces = _get_label(atoms, "forces")
        if forces is None:
            if self.require_forces:
                raise ValueError(
                    f"Forces data is missing for structure with {n_atoms} atoms"
                )
        else:
            if not hasattr(forces, "__len__") or len(forces) != n_atoms:
                raise ValueError(
                    f"Forces shape mismatch: expected {n_atoms} atoms, got forces shape {getattr(forces, 'shape', len(forces) if hasattr(forces, '__len__') else 'unknown')}"
                )

            forces = torch.tensor(forces, dtype=torch.float32)  # [N_atoms, 3]
            if forces.shape != (n_atoms, 3):
                raise ValueError(
                    f"Forces shape incorrect: expected ({n_atoms}, 3), got {forces.shape}"
                )

            result["forces"] = forces

        virial = _get_label(atoms, "virial")
        stress = _get_label(atoms, "stress")
        if virial is not None:
            virial = np.asarray(virial, dtype=np.float64).reshape(-1)
            if virial.size == 6:
                virial = virial[[0, 5, 4, 5, 1, 3, 4, 3, 2]]
            if virial.size != 9:
                raise ValueError(
                    f"Virial must contain 6 or 9 values, got {virial.size}."
                )
            result["virial"] = torch.tensor(virial, dtype=torch.float32)
        elif stress is not None:
            stress = np.asarray(stress, dtype=np.float64).reshape(-1)
            if len(stress) == 6:
                virial = -stress[[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
            else:
                virial = -stress.reshape(-1) * atoms.get_volume()
            if virial.size != 9:
                raise ValueError(
                    f"Stress must contain 6 or 9 values, got {stress.size}."
                )
            result["virial"] = torch.tensor(virial, dtype=torch.float32)  # [9]

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _get_label(atoms, name):
    """Read an extxyz label without relying on Wizard's legacy XYZ wrapper."""
    value = atoms.info.get(name)
    if value is not None:
        return value
    value = atoms.arrays.get(name)
    if value is not None:
        return value
    calculator = getattr(atoms, "calc", None)
    results = getattr(calculator, "results", {}) if calculator is not None else {}
    return results.get(name)
