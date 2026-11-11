# mobius-oam-mp

Möbius MP trained on OMat24 and fine-tuned on sAlex and MPTrj. The actual
float32 inference weight is [model.pt](model.pt).

## Use

After [installing RUNE](../../README.md), run this Python example from the
repository root. The interface accepts any ASE structure within the model's
supported element set:

```python
from ase.io import read
from rune.mobius.plugin import build_calculator

atoms = read("models/example.extxyz", index=0)
atoms.calc = build_calculator("models/mobius-oam-mp/model.pt", device="cpu")
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()           # eV/angstrom
stress = atoms.get_stress()           # eV/angstrom^3
```

For native GPUMD deployment, export the weight to a text potential:

```python
from rune.mobius.plugin import load_model, export_model

model = load_model("models/mobius-oam-mp/model.pt", device="cpu")
export_model(model, "mobius.txt")
```

Compile GPUMD from source with the Möbius CUDA adapter enabled, then select
`mobius.txt` in your simulation inputs. The [deployment documentation](../../docs/deployment.md)
provides the build and run instructions.

## Architecture

MP adds grouped scalar messages between neighboring atoms. Directional aggregates are contracted to invariant scalars within a message block; the persistent hidden state is scalar.

| Setting | Value |
|---|---|
| Trainable parameters | 3,327,499 |
| Elements | 89; ordered list in the [catalog](../README.md#elements) |
| Descriptor dimensions | 103 |
| Radial / angular cutoff | 6 / 5 Å |
| Hidden / feed-forward width | 128 / 256 |
| Local residual blocks | 4 |
| Message blocks | 2 |
| Directional state | 8 transient channels; contracted within each block |
| ZBL inner / outer cutoff | 1 / 2 Å |

Exact resolved model parameters are stored in `model.pt` as `model_params`.
The model predicts scalar energies and conservative forces/virials; it does not
expose independently controllable spin or charge degrees of freedom.

## Training

| Stage | Data and completed epochs | Optimizer and schedule | Effective batch |
|---|---|---|---:|
| Pretraining | OMat24; 6 epochs | AdamW, 1e-3; cosine over 40 epochs, minimum 1e-5 | 256 |
| Fine-tuning | sAlex ×1 + MPTrj ×8; one mixed epoch | New AdamW, constant 1e-4 | 256 |

Pretraining used 16 GPUs with 16 structures per rank; fine-tuning used 8 GPUs
with 32 structures per rank. Both stages used accumulation 1, seed 42, weight
decay 1e-4 and energy/force/virial loss weights 1/1/0.1. 

The historical [pretraining recipe](recipes/pretrain/rune.toml) and
[fine-tuning recipe](recipes/finetune/rune.toml) preserve numerical settings and
use relative data/cache/parent paths.  

The supplied `model.pt` can initialize a [new fine-tuning stage](../../docs/training.md). It omits optimizer,
scheduler, RNG and run configuration, so it does not resume the full original
training state.

## Matbench Discovery

| Metric | Value |
|---|---:|
| CPS ↑ | 0.772360 |
| F1, unique prototypes ↑ | 0.828043 |
| Geometry RMSD, unitless ↓ | 0.072268 |
| κ_SRME ↓ | 0.467413 |
| WBM structures / phonon materials | 256,963 / 103 |
| Unconverged relaxations | 319 |
| Geometry unmatched, penalized | 14,621 |
| Materials with imaginary modes, penalized | 0 |

[All models](../README.md)
