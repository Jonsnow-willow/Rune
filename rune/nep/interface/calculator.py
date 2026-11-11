import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator as ASECalculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.stress import full_3x3_to_voigt_6_stress

from ..plugin import load_model
from .batch import BatchBuilder


class Calculator(ASECalculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        artifact,
        device="cpu",
        compute_stress: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.model = load_model(artifact, device=self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.compute_stress = bool(compute_stress)
        self.implemented_properties = ["energy", "free_energy", "forces"]
        if self.compute_stress:
            self.implemented_properties.append("stress")
        model_dtype = next(self.model.parameters()).dtype
        self.batch_builder = BatchBuilder(
            self.model.params,
            device=self.device,
            dtype=model_dtype,
        )

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        properties = properties or ["energy"]
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("Calculator requires an ASE Atoms object.")

        requested = set(properties)
        compute_forces = "forces" in requested
        compute_virial = "stress" in requested
        if compute_virial and not self.compute_stress:
            raise PropertyNotImplementedError(
                "Stress is disabled for this Calculator. "
                "Construct it with compute_stress=True for NPT or stress calculations."
            )
        if compute_virial:
            volume = atoms.get_volume()
            if volume <= 0:
                raise ValueError("Stress calculation requires a positive cell volume.")

        batch = self._build_batch(
            atoms,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        with torch.no_grad():
            prediction = self.model(batch)

        energy = float(prediction["energy"][0].detach().cpu())
        self.results = {
            "energy": energy,
            "free_energy": energy,
        }
        if compute_forces:
            self.results["forces"] = prediction["forces"].detach().cpu().numpy()
        if compute_virial:
            virial = prediction["virial"][0].detach().cpu().numpy().reshape(3, 3)
            stress = -np.asarray(virial) / volume
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

    def _build_batch(
        self,
        atoms,
        *,
        compute_forces: bool = True,
        compute_virial: bool = False,
    ):
        return self.batch_builder.build(
            atoms,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
