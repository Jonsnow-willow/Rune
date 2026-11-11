# Inference

[Documentation](README.md) · [Model cards](../models/README.md)

The repository includes three trained Möbius-OAM models. Run examples from the
project root after [installing RUNE](installation.md).

| Model | Weight file | Hidden-state interaction |
| --- | --- | --- |
| Local | `models/mobius-oam-loc/model.pt` | Local invariant representation |
| MP | `models/mobius-oam-mp/model.pt` | Scalar messages with transient directional features |
| EQ | `models/mobius-oam-eq/model.pt` | Persistent angular states between message blocks |

All three use the same ASE interface. Each model card records its element order,
training route, checksums, and evaluation results. Changing the filename selects
a separately trained model; editing the architecture of a loaded checkpoint
would change its fitted function.

## ASE energy, forces, and stress

```python
from ase.io import read
from rune.mobius.plugin import build_calculator

atoms = read("models/example.extxyz", index=0)
atoms.calc = build_calculator("models/mobius-oam-eq/model.pt", device="cpu")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress(voigt=False)

print("Energy (eV):", energy)
print("Forces (eV/Å):", forces)
print("Stress (eV/Å³):", stress)
```

Replace `models/example.extxyz` with your ASE-readable structure file;
`index=0` selects its first frame. Change `mobius-oam-eq` to `mobius-oam-loc`
or `mobius-oam-mp` to use another trained model.

`get_stress()` returns ASE's six-component Voigt representation by default;
`voigt=False` returns a 3 × 3 tensor. RUNE's model output is a total virial, and
the Calculator converts it to stress as `-virial / cell_volume`. Stress requires
a cell with positive volume. For a molecule without a cell, disable stress:

```python
from ase.build import molecule
from rune.mobius.plugin import build_calculator

atoms = molecule("H2O")
atoms.calc = build_calculator(
    "models/mobius-oam-eq/model.pt", device="cpu", compute_stress=False,
)
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

The molecule example demonstrates the interface; it is not evidence of model
accuracy for molecular chemistry. The model's encoded elements and aggregate
benchmark results do not establish accuracy for every composition or condition.

For GPU prediction, use `device="cuda"` with a CUDA-enabled PyTorch environment.
The loader preserves the stored tensor precision. Unsupported elements raise
an error rather than being silently mapped to another element.

## Loading a model directly

```python
from rune.mobius.plugin import load_model

model = load_model("models/mobius-oam-eq/model.pt", device="cpu")
print(model.params)
```

Use the Calculator for ordinary ASE structures. A direct model call expects a
RUNE tensor batch; the model-owned `rune.mobius.interface.batch.BatchBuilder`
constructs it when lower-level integration is needed. The root `rune` package
exposes package metadata; model capabilities live in their explicit modules.

The bundled `model.pt` files contain model parameters and tensor states, without
optimizer or random-number-generator state. They can initialize a new
[fine-tuning run](training.md#fine-tune-a-trained-model), but cannot resume an
old training run's optimization history. Full training checkpoints can also be
loaded for inference. Load checkpoints only from sources you trust, because
legacy training checkpoints contain Python-serialized state.

## NEP inference

RUNE also loads NEP4 checkpoints and GPUMD `nep.txt` files through its NEP plugin.
For example, after training a NEP run in `runs/nep`:

```python
from ase.io import read
from rune.nep.plugin import build_calculator

atoms = read("models/example.extxyz", index=0)
atoms.calc = build_calculator("runs/nep/exports/nep.txt", device="cpu")
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

The NEP must support the elements in your structure. No pretrained NEP weight is
bundled here. See [configuration](configuration.md) for a NEP training input and
[deployment](deployment.md#nep-export) for text export. Möbius `mobius.txt` is
read by the native GPUMD adapter, not by the Python Möbius loader.
