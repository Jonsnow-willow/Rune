<p align="center">
  <img src="docs/assets/rune-logo.png" alt="RUNE logo" width="600">
</p>

# RUNE

RUNE trains, fine-tunes, and deploys machine-learned interatomic potentials.
It supports NEP4 and Möbius models, energy/force/virial training, ASE prediction,
and native GPUMD deployment.

Möbius provides three architectures: **Local** uses local invariant features,
**MP** exchanges scalar features with transient directional messages, and
**EQ** carries angular states between message blocks. This repository includes
trained OAM weights for all three, with model cards and training configurations.

## Setup

Use Python 3.10 or newer and run the commands and Python examples below from
the repository root. Create an environment and install RUNE:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
rune --help
```

For development, use `python -m pip install -e .`. The equivalent module CLI is
`python -m rune.cli.run`; with the dependencies installed, it also runs directly
from this source checkout.
See [environment setup](docs/installation.md) for Windows and CUDA details.

## Möbius-OAM models

OAM denotes OMat24 pretraining followed by fine-tuning on sAlex and MPTrj.
All three models encode the same 89 elements and completed one mixed
fine-tuning epoch. Each model directory contains `model.pt`, a model card,
and pretraining/fine-tuning configurations under `recipes/`.
These model files and configurations are repository assets, separate from the
installed Python package.

| Model | Parameters | Hidden-state interaction | OMat24 pretraining | Matbench CPS (local) |
| --- | ---: | --- | --- | ---: |
| [Local](models/mobius-oam-loc/README.md) | 2,977,803 | Local invariant features | 6 cosine epochs | 0.680395 |
| [MP](models/mobius-oam-mp/README.md) | 3,327,499 | Scalar messages, transient vectors | 6 cosine epochs | 0.772360 |
| [EQ](models/mobius-oam-eq/README.md) | 5,001,403 | Persistent angular states | 6 cosine epochs | 0.840992 |

The [model catalog](models/README.md) lists supported elements and weight
details. Training, validation, and benchmark analysis are described in the
[paper](paper/README.md).

## Use a trained model

The weights are in `models/`; no separate download is needed for these examples.
Create a silicon cell with ASE and evaluate it with the EQ model:

```python
from ase.build import bulk
from rune.mobius.plugin import build_calculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = build_calculator("models/mobius-oam-eq/model.pt", device="cpu")

print("Energy (eV):", atoms.get_potential_energy())
print("Forces (eV/Å):", atoms.get_forces())
print("Stress (eV/Å³):", atoms.get_stress(voigt=False))
```

Change `mobius-oam-eq` to `mobius-oam-loc` or `mobius-oam-mp` to use either
of the other trained models. Replace `atoms` with your own ASE structure,
for example loaded with `ase.io.read`. Stress requires a cell with positive volume; the
[inference guide](docs/inference.md) also covers molecules, GPU prediction,
and NEP files.

## Train and fine-tune

A run is a directory containing `rune.toml` and references to labeled data.
Prepare a configuration and your own labeled data using the
[training guide](docs/training.md), then start or continue a run:

```bash
python -m rune.cli.run train runs/my-model
python -m rune.cli.run train runs/my-model --resume checkpoints/last.pt --epochs 100
```

For fine-tuning, copy the selected model's training configuration into a new
run directory. For EQ:

```bash
mkdir -p runs/eq-finetune
cp models/mobius-oam-eq/recipes/finetune/rune.toml runs/eq-finetune/rune.toml
```

Keep `[model]` unchanged, set `finetune.checkpoint` to
`"../../models/mobius-oam-eq/model.pt"`, and replace the historical `[data]`
section with your own files and batch settings. Choose epochs, learning rate,
loss weights, and runtime device using the
[fine-tuning guide](docs/training.md#fine-tune-a-trained-model), then launch:

```bash
python -m rune.cli.run finetune runs/eq-finetune
```

Fine-tuning starts a new optimization history from the selected weights.
Use `--resume` for a full checkpoint from your own run; `--epochs` is the total
target epoch count.

## Deploy in GPUMD

The models are distributed as `.pt` weights. Export a native text potential
from the RUNE project root:

```python
from rune.mobius.plugin import load_model, export_model

model = load_model("models/mobius-oam-eq/model.pt", device="cpu")
export_model(model, "runs/gpumd-eq/mobius.txt")
```

Then compile GPUMD with the Möbius CUDA adapter and select `potential mobius.txt`
in `run.in`, following the [deployment guide](docs/deployment.md). NEP4 exports
use the standard `nep.txt` path and do not require the Möbius adapter. Native
inference does not require Python or PyTorch. The exporter creates the output
directory automatically.

## Authors:

| Name                  | contact                           |
| --------------------- | --------------------------------- |
| Jiahui Liu            | liujiahui@zgci.ac.cn              |

## Documentation and report

| Topic | Entry |
| --- | --- |
| Environment and prediction | [Setup](docs/installation.md) · [Inference](docs/inference.md) |
| Training and `rune.toml` | [Training](docs/training.md) · [Configuration](docs/configuration.md) |
| Export and GPUMD compilation | [Deployment](docs/deployment.md) |
| Model architecture | [Architecture](docs/architecture.md) |
| Trained weights and results | [Models](models/README.md) |
| Paper, validation, and benchmarks | [Paper](docs/paper.pdf) |
| Full guide | [Documentation](docs/README.md) |
