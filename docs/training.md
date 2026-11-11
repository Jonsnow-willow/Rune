# Training and fine-tuning

[Documentation](README.md) · [Configuration](configuration.md)

RUNE fits energy, force, and virial labels with the same CLI for NEP4 and
Möbius. A run directory contains one `rune.toml` and references to its data.
Run all commands below from the project root after
[setting up the environment](installation.md).

## Train on your data

Create a directory such as `runs/nep` with `rune.toml` and your labeled data.
The [complete NEP4 configuration](configuration.md#a-complete-nep4-run) can be
used as the starting input. Select the elements and numerical settings for
your system before training:

```bash
python -m rune.cli.run train runs/nep
```

Add `--device cuda` to use a CUDA GPU instead of the configured device.
An existing run with checkpoints must be continued explicitly with `--resume`.
Running `python -m rune.cli.run` without a subcommand trains the current
directory using its `rune.toml`.

### Data conventions

The standard data input is extended XYZ. Energy is a total structure energy in
eV, forces are in eV/Å, and positions/cell vectors are in Å. Virial is a total
structure quantity in eV. When stress labels are supplied instead, they use ASE
units of eV/Å³ and are converted by `virial = -stress * volume`.

| Label | Shape or meaning |
| --- | --- |
| `energy` | Scalar total energy for a structure |
| `forces` | One three-component vector per atom |
| `virial` | Nine tensor entries or six components in ASE Voigt order |
| `stress` | Nine tensor entries or six components in ASE Voigt order |

The six-component order is `xx, yy, zz, yz, xz, xy`. Use consistent energy
references and stress signs across your sources. Forces are required when the
force-loss weight is nonzero. Omit `data.test_file` to train without a test
loader; an explicitly configured missing test file is an error. For data without
virial/stress labels, set `loss.virial = 0.0`.

`data_format = "auto"` selects eager or indexed extended-XYZ loading according
to file size. `eager` and `lazy_xyz` select those paths explicitly. Lazy indices
are stored under `data.index_dir`, which defaults to `.rune_index` in the run.

## Fine-tune a trained model

The three OAM weights are usable starting points for a new optimization stage.
The model directories include historical training configurations with the
corresponding architectures. For EQ, copy its configuration into a new run:

```bash
mkdir -p runs/eq-finetune
cp models/mobius-oam-eq/recipes/finetune/rune.toml runs/eq-finetune/rune.toml
```

Supply your own labeled data at `data/train.xyz` under the project root.
In the copied `rune.toml`, replace the existing `[finetune]` and `[data]`
sections with:

```toml
[finetune]
checkpoint = "../../models/mobius-oam-eq/model.pt"

[data]
train_file = "../../data/train.xyz"
# test_file = "../../data/test.xyz"
batch_size = 5
```

These paths apply to `runs/eq-finetune`. Replacing the whole `[data]` section
also removes the historical mixed-data repetition settings. Uncomment
`test_file` if you provide validation data. Keep the full `[model]` section:
the plugin, exact element order, and resolved parameters must match the source
weight. Set `runtime.device = "auto"` for automatic device selection, and
choose epochs, learning rate, batch size, and loss weights for your task.
Then launch:

```bash
python -m rune.cli.run finetune runs/eq-finetune
```

For another architecture, start from the historical
[Local configuration](../models/mobius-oam-loc/recipes/finetune/rune.toml) or
[MP configuration](../models/mobius-oam-mp/recipes/finetune/rune.toml), and set
`finetune.checkpoint` to its corresponding `model.pt`. Use a new run directory
and adjust all relative paths to that location.

Fine-tuning loads all weights and the saved descriptor scaler, then creates a
new optimizer, scheduler, epoch count, best score, and random state using the
configured seed. It does not automatically freeze layers, refit energy
references, or choose new loss weights. Leave `runtime.resume_model_only` at
its default `false`; the fine-tuning entry selects weights-only initialization
when it starts a new stage.

The bundled inference weights do not contain the optimizer/RNG state needed to
resume their historical training. They initialize a new stage. Full checkpoints
written by your new stage can subsequently be resumed.

## Mixed training data

`train_file` can be a list and `train_repeats` specifies a positive integer
number of presentations for each source. For example, in a run directory with
`alex.xyz` and `mptrj.xyz`:

```toml
[data]
train_file = ["alex.xyz", "mptrj.xyz"]
train_repeats = [1, 8]
data_format = "lazy_xyz"
batch_size = 16
num_workers = 4

[runtime]
gradient_accumulation_steps = 2
```

One mixed epoch presents each selected frame from the first source once and
from the second source eight times, shuffled together. Repeats reuse dataset
objects and indices without copying XYZ files. `frame_stride` and
`max_train_frames` apply separately to each source before repetition. The test
dataset remains a single unrepeated file. This example explains the sampling
mechanism; select source proportions for your own training objective.

## Distributed execution

Use `torchrun` to launch one process per GPU:

```bash
torchrun --nproc_per_node=2 -m rune.cli.run train runs/nep --device cuda
torchrun --nproc_per_node=2 -m rune.cli.run finetune runs/eq-finetune --device cuda
```

`data.batch_size` is per process. With eight GPUs, batch size 16 per process,
and accumulation 2, an ordinary optimizer update uses 256 structures. The final
update can be smaller; the distributed sampler can pad the shuffled dataset
prefix so each rank receives the same number of frames. Schedulers advance at
epoch boundaries.

## Continue a run

```bash
python -m rune.cli.run finetune runs/eq-finetune --resume checkpoints/last.pt --epochs 2
python -m rune.cli.run train runs/nep --resume checkpoints/last.pt --epochs 100
```

`--epochs` is the total target, not the number of additional epochs. `--resume`
(or `runtime.resume`) takes precedence over `finetune.checkpoint` and restores
the model, optimizer, scheduler, completed epoch, best score, and RNG state.
Strict continuation rejects incompatible plugin, model-parameter, and scheduler
settings. Checkpoints are saved at epoch boundaries; unfinished-epoch recovery
is not supported.

To inspect a checkpoint without loading the run configuration:

```bash
python -m rune.cli.run report runs/eq-finetune --checkpoint checkpoints/last.pt
```

## Outputs and checkpoint retention

A run writes:

```text
runs/eq-finetune/
├── rune.toml
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── exports/
│   └── mobius.txt
└── logs/
    └── metrics.csv
```

Full checkpoints contain the plugin identity, resolved model parameters, run
configuration, model/optimizer/scheduler states, epoch metrics, best score,
and RNG state. To retain historical checkpoints as well:

```toml
[runtime]
save_every = 1
keep_every = 5
```

`keep_every` defaults to `0` (disabled). A value of 5 retains
`epoch-0005.pt`, `epoch-0010.pt`, and so on, plus the final epoch on normal
completion. For 13 epochs, this keeps epochs 5, 10, and 13. The interval uses the
cumulative epoch number after resuming. These full-state checkpoints can also
be selected with `--resume`.

Historical checkpoints are not overwritten or automatically deleted. Branching
from an earlier checkpoint into a run with a colliding history requires a new
run directory. `save_every` controls updates of `last.pt`; zero disables periodic
updates, but normal completion still writes `last.pt`. `best.pt` tracks the best
score. Checkpoints are published atomically, so a failed write does not replace
the previous complete checkpoint.

Models with an export capability write a final native text potential after
training; `runtime.export_every` also enables periodic exports. NEP4 writes
`nep.txt`, and Möbius writes `mobius.txt`. See
[deployment](deployment.md) to use those files in GPUMD.
