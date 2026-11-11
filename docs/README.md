# RUNE documentation

RUNE trains and fine-tunes interatomic potentials and uses them through ASE or
GPUMD. The guides below cover the Python interface, trained Möbius models,
and training configurations. Commands are written for the project root.

| Task | Guide |
| --- | --- |
| Install RUNE and choose CPU or CUDA | [Installation](installation.md) |
| Predict energies, forces, and stress | [Inference](inference.md) |
| Train, fine-tune, and continue a run | [Training](training.md) |
| Set up `rune.toml`, NEP4, or Möbius | [Configuration](configuration.md) |
| Export a potential and run it in GPUMD | [Deployment](deployment.md) |
| Understand the software and Local/MP/EQ architectures | [Architecture](architecture.md) |

The [model directory](../models/README.md) contains trained `.pt` weights,
model cards, historical training configurations, and benchmark summaries.
The [deployment guide](deployment.md) explains how to export native text potentials.
Use the [ASE interface](inference.md) for prediction and the
[training guide](training.md) to prepare runs with your own labeled data.

The [technical report](../paper/paper.md) describes the software, model family,
training routes, validation, and benchmark results. A [PDF](../paper/paper.pdf) is generated
from the same report source.

## Package organization

```text
rune/
├── cli/        configuration, arguments, and dispatch
├── runtime/    devices, distributed execution, checkpoints, and logging
├── datasets/   structures, labels, neighbors, and batches
├── training/   training and fine-tuning entry points and shared trainer
├── nep/        NEP4 model, derivatives, serialization, and ASE interface
└── mobius/     Möbius model, derivatives, serialization, ASE and GPUMD interfaces
```

A run follows `rune.toml` → model plugin → constructed model → shared trainer.
Each model folder owns its mathematical operations and external interfaces.
[The plugin contract](configuration.md#model-plugin-interface) explains how to
add a model without adding model-name switches to the shared training code.
