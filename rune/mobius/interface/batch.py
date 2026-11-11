from __future__ import annotations

import numpy as np
import torch
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import neighbor_list


def _pack_neighbors(
    centers: np.ndarray,
    neighbors: np.ndarray,
    offsets: np.ndarray,
    n_atoms: int,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    packed_neighbors = np.full((n_atoms, max_neighbors), -1, dtype=np.int64)
    packed_offsets = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float64)
    if centers.size == 0:
        return packed_neighbors, packed_offsets

    order = np.argsort(centers, kind="stable")
    centers = centers[order]
    neighbors = neighbors[order]
    offsets = offsets[order]
    counts = np.bincount(centers, minlength=n_atoms)
    largest_count = int(counts.max())
    if largest_count > max_neighbors:
        atom_index = int(np.argmax(counts))
        raise ValueError(
            f"Neighbor count {largest_count} for atom {atom_index} exceeds "
            f"max_neighbors={max_neighbors}. Increase max_neighbors_radial "
            "or max_neighbors_angular."
        )

    starts = np.cumsum(counts) - counts
    slots = np.arange(centers.size) - np.repeat(starts, counts)
    packed_neighbors[centers, slots] = neighbors
    packed_offsets[centers, slots] = offsets
    return packed_neighbors, packed_offsets


def _build_neighbor_list(
    atoms,
    *,
    cutoff_radial: float,
    cutoff_angular: float,
    max_neighbors_radial: int,
    max_neighbors_angular: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the two padded Möbius neighbor lists with ASE's cell-list backend."""
    cutoff_radial = float(cutoff_radial)
    cutoff_angular = float(cutoff_angular)
    if cutoff_radial <= 0.0 or cutoff_angular <= 0.0:
        raise ValueError("Möbius radial and angular cutoffs must be positive.")
    if max_neighbors_radial < 1 or max_neighbors_angular < 1:
        raise ValueError(
            "Möbius radial and angular max-neighbor counts must be positive."
        )

    n_atoms = len(atoms)
    if n_atoms == 0:
        radial_neighbors = np.full((0, max_neighbors_radial), -1, dtype=np.int64)
        radial_offsets = np.zeros((0, max_neighbors_radial, 3), dtype=np.float64)
        angular_neighbors = np.full((0, max_neighbors_angular), -1, dtype=np.int64)
        angular_offsets = np.zeros((0, max_neighbors_angular, 3), dtype=np.float64)
        return (
            radial_neighbors,
            radial_offsets,
            angular_neighbors,
            angular_offsets,
        )

    cutoff = max(cutoff_radial, cutoff_angular)
    centers, neighbors, image_shifts = neighbor_list("ijS", atoms, cutoff)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    offsets = np.asarray(image_shifts, dtype=np.float64) @ cell
    positions = np.asarray(atoms.positions, dtype=np.float64)
    vectors = positions[neighbors] + offsets - positions[centers]
    distance_squared = np.einsum("ij,ij->i", vectors, vectors)

    radial_mask = distance_squared < cutoff_radial * cutoff_radial
    angular_mask = distance_squared < cutoff_angular * cutoff_angular
    radial_neighbors, radial_offsets = _pack_neighbors(
        centers[radial_mask],
        neighbors[radial_mask],
        offsets[radial_mask],
        n_atoms,
        max_neighbors_radial,
    )
    angular_neighbors, angular_offsets = _pack_neighbors(
        centers[angular_mask],
        neighbors[angular_mask],
        offsets[angular_mask],
        n_atoms,
        max_neighbors_angular,
    )
    return (
        radial_neighbors,
        radial_offsets,
        angular_neighbors,
        angular_offsets,
    )


class BatchBuilder:
    """Convert one ASE structure directly to a Möbius inference batch."""

    def __init__(self, params: dict, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.elements = list(params["elements"])
        self.cutoff_radial = float(params["rcut_radial"])
        self.cutoff_angular = float(params["rcut_angular"])
        self.max_neighbors_radial = int(params["max_neighbors_radial"])
        self.max_neighbors_angular = int(params["max_neighbors_angular"])
        self.z_to_type = {
            atomic_numbers[element]: type_index
            for type_index, element in enumerate(self.elements)
        }
        self._cached_numbers: tuple[int, ...] | None = None
        self._cached_types: torch.Tensor | None = None
        self._cached_counts: torch.Tensor | None = None

    def build(
        self,
        atoms,
        *,
        compute_forces: bool,
        compute_virial: bool,
    ) -> dict:
        types, counts = self._types_and_counts(atoms)
        (
            radial_neighbors,
            radial_offsets,
            angular_neighbors,
            angular_offsets,
        ) = _build_neighbor_list(
            atoms,
            cutoff_radial=self.cutoff_radial,
            cutoff_angular=self.cutoff_angular,
            max_neighbors_radial=self.max_neighbors_radial,
            max_neighbors_angular=self.max_neighbors_angular,
        )
        return {
            "types": types,
            "positions": torch.as_tensor(
                np.asarray(atoms.positions),
                dtype=self.dtype,
                device=self.device,
            ),
            "n_atoms_per_structure": counts,
            "radial_neighbors": torch.as_tensor(
                radial_neighbors, dtype=torch.long, device=self.device
            ),
            "angular_neighbors": torch.as_tensor(
                angular_neighbors, dtype=torch.long, device=self.device
            ),
            "radial_offsets": torch.as_tensor(
                radial_offsets, dtype=self.dtype, device=self.device
            ),
            "angular_offsets": torch.as_tensor(
                angular_offsets, dtype=self.dtype, device=self.device
            ),
            "compute_forces": bool(compute_forces),
            "compute_virial": bool(compute_virial),
        }

    def _types_and_counts(self, atoms) -> tuple[torch.Tensor, torch.Tensor]:
        numbers = tuple(int(number) for number in atoms.numbers)
        if numbers == self._cached_numbers:
            assert self._cached_types is not None
            assert self._cached_counts is not None
            return self._cached_types, self._cached_counts

        missing = sorted(set(numbers).difference(self.z_to_type))
        if missing:
            symbols = [
                chemical_symbols[number]
                if 0 <= number < len(chemical_symbols)
                else f"Z{number}"
                for number in missing
            ]
            raise ValueError(
                f"Atoms contain elements not supported by this Möbius model: {symbols}. "
                f"Model elements (type-id order): {self.elements}."
            )

        self._cached_numbers = numbers
        self._cached_types = torch.tensor(
            [self.z_to_type[number] for number in numbers],
            dtype=torch.long,
            device=self.device,
        )
        self._cached_counts = torch.tensor(
            [len(numbers)], dtype=torch.long, device=self.device
        )
        return self._cached_types, self._cached_counts
