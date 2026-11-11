# Configuration reference

[Documentation](README.md) · [Training guide](training.md)

All commands are run from the project root. Dataset and checkpoint paths inside
`rune.toml` are resolved relative to that run directory unless absolute.
Unknown section names, common fields, and model parameters are rejected.

## A complete NEP4 run

Each run is a directory containing one `rune.toml` and its datasets:

```text
my-run/
├── rune.toml
├── train.xyz
└── test.xyz       # optional
```

A compact NEP4 configuration is:

```toml
[model]
plugin = "rune.nep"

[model.params]
elements = ["H", "He"]
rcut_radial = 5.0
rcut_angular = 4.0
n_max_radial = 4
n_max_angular = 4
basis_size_radial = 8
basis_size_angular = 8
l_max = 4
max_neighbors_radial = 100
max_neighbors_angular = 30
hidden_dims = [30]

[data]
train_file = "train.xyz"
test_file = "test.xyz"
batch_size = 5

[optimizer]
name = "adamw"
epochs = 500
learning_rate = 1.0e-3

[scheduler]
name = "none"

[loss]
energy = 1.0
forces = 1.0
virial = 0.1

[runtime]
device = "auto"
```

## NEP4 parameters

RUNE's current NEP configuration limits are listed below. The descriptor ranges
follow the [GPUMD training parameter parser at a8416f3](https://github.com/brucefan1983/GPUMD/blob/a8416f3466bc35741a33669a948cc3569d2c6be7/src/main_nep/parameters.cu);
the ANN width limit is set by RUNE.

| Parameter | Accepted values |
| --- | --- |
| `elements` | Unique chemical symbols from H through Pu |
| Cutoffs (Å) | Finite values with `3 <= rcut_angular <= rcut_radial <= 100` |
| `n_max_radial`, `n_max_angular` | Integers in `0..12`, `0..8` |
| `basis_size_radial`, `basis_size_angular` | Integers in `0..16`, `0..12` |
| `l_max` | Integer in `2..8` |
| `l_max_4body`, `l_max_5body` | `0/2`, `0/1`, respectively |
| `has_q_112`, `has_q_123`, `has_q_233`, `has_q_134` | Integer switches `0/1` |
| `hidden_dims` | A one-element list with an integer width in `1..200` |

`q_123` and `q_233` require `l_max >= 3`; `q_134` requires `l_max >= 4`.
The angular descriptor count, `(n_max_angular + 1) * (l_max + enabled higher-order
terms)`, must not exceed 90. Together with the radial limit, this caps the total
descriptor count at 103. Neighbor limits must be positive integers; GPUMD text
export additionally requires `max_neighbors_radial <= 819`.

RUNE currently imports and exports single-hidden-layer NEP4 models. It rejects
nonzero second-layer widths in an `ANN width second_width` text header. GPUMD's
training parser accepts two layers, while the ordinary MD reader at the pinned
revision still uses the single-layer layout.

Optional ZBL accepts `[rc_inner, rc_outer]` or a mapping with those fields.
Universal ZBL requires finite `0 <= rc_inner < rc_outer <= rcut_angular`;
the inner cutoff remains independently configurable. Flexible ZBL uses a mapping
with zero inner and outer header cutoffs and `parameters` containing ten values
per unordered element pair. Parameters must be finite, and each pair must satisfy
the same cutoff ordering and angular-neighbor coverage. GPUMD's 550-value storage
limits flexible ZBL to ten elements; universal ZBL has no such pair-table limit.
Existing text models retain their cutoff values on import; the new-configuration
cutoff range above is not imposed on them.

NEP owns a `ZBLRepulsion(nn.Module)` in `rune/nep/zbl.py`, using the same
component interface and buffer names as Möbius while keeping both model folders
independent. The component manages ZBL parameters and returns per-atom energies;
the NEP model combines learned and ZBL energies, computes forces and virials,
and handles text serialization. Both derivative paths use the same ZBL energy
formula on the angular neighbor list.

ZBL checkpoint buffers now use `zbl.atomic_numbers`, `zbl.universal_coefficients`,
and `zbl.pair_parameters`; disabled ZBL registers no buffers. Checkpoints with
the former root-level ZBL keys require explicit conversion before loading,
including old checkpoints created without ZBL. No automatic migration is applied.
The `nep.txt` format is unchanged. Flexible ZBL export writes the current module
buffer values at their tensor precision, matching the parameters used in prediction.

## Möbius architectures

For Möbius, select its plugin and add its network fields:

```toml
[model]
plugin = "rune.mobius"

[model.params]
elements = ["H", "He"]
hidden_dim = 96
blocks = 3
ffn_dim = 192
message_layers = 2
message_dim = 64
pair_dim = 16
message_groups = 8
vector_channels = 8
```

This is the existing scalar message-passing branch: each message block creates
a temporary vector branch and contracts it to invariants. For **Möbius Local**,
set `message_layers = 0`; the shared element-conditioned residual network still
uses `blocks` local updates (three by default). Local requires its own trained
weights; removing message blocks from a trained checkpoint does not preserve
that model's predictions or benchmark results.

For the **persistent equivariant branch**, use:

```toml
[model]
plugin = "rune.mobius"

[model.params]
elements = ["Fe", "C"]
message_mode = "equivariant"
message_layers = 2
equivariant_channels = 8
equivariant_l_max = 4
l_max = 4
```

The scalar local encoder is shared with the other branches. Normalized angular
descriptor moments initialize direction states, and each message block updates
both scalar and direction states. Invariant projections onto both ends of a bond
and scalar node context determine edge gates; squared direction norms feed back
into the scalar energy network. All forces and virials derive from the same
energy, including the additional angular-moment derivative path. See
[the architecture guide](architecture.md#eq-persistent-angular-state-with-invariant-gates)
for the equations and implementation mapping.

`equivariant_channels` defaults to `8` and must be positive.
`equivariant_l_max` defaults to the descriptor's `l_max` and must lie in
`1..min(l_max, 8)`; `message_layers` must be at least one in equivariant mode.
These are software defaults, not an optimized accuracy or speed prescription.
`vector_channels` controls only the scalar branch's temporary vectors.
Equivariant-only fields are rejected unless `message_mode = "equivariant"`.
The default scalar mode omits the new default fields from resolved parameters,
preserving existing Local/scalar checkpoint identities. Changing to the
equivariant branch changes the architecture and requires a separate model.

The equivariant branch supports energy/force/virial fitting and RUNE checkpoints.
All three branches also have a native CUDA GPUMD evaluator with lossless weight
export; see [GPUMD deployment](deployment.md). The native evaluator reuses
per-model scratch buffers and skips initialization only for fully overwritten
outputs. The native evaluator also caches a 1 Å skin neighbor list, uses per-center moment reductions,
and evaluates normalization and energy heads with warp reductions.
The training derivative interface does not provide
arbitrary-order coordinate derivatives.

Unknown common fields and unknown model parameters are rejected instead of
being silently ignored.

## Shared sections and defaults

The model plugin resolves `[model.params]`. The following common settings are
handled by the training workflow. Omitted settings use the listed defaults.

| Section | Settings and defaults |
| --- | --- |
| `data` | `train_file = "train.xyz"`, optional `test_file`, `batch_size = 5`, `num_workers = 0`, `shuffle = true` |
| `data` loading | `data_format = "auto"` (`eager` and `lazy_xyz` also supported), `lazy_threshold_mb = 512`, `frame_stride = 1`, optional `max_train_frames` / `max_test_frames` |
| `data` workers | `pin_memory = "auto"`, `persistent_workers = "auto"`, `prefetch_factor = 2` |
| `data` indexing | `index_dir = ".rune_index"`, `cache_index = true` |
| `optimizer` | AdamW: `epochs = 500`, `learning_rate = 0.001`, `weight_decay = 0.0001`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8` |
| `loss` | `energy = 1.0`, `forces = 1.0`, `virial = 0.1` |
| `runtime` | `device = "auto"`, `seed = 42`, `gradient_accumulation_steps = 1`, optional `gradient_clip_norm` |
| `runtime` checkpoints | `save_every = 1`, `keep_every = 0`, `export_every = 0`; optional `resume` path |
| `runtime` initialization | `resume_model_only = false`, `resume_reset_best = false`, `compute_descriptor_scaler_once = true`, optional `descriptor_scaler_max_batches` |
| `runtime` logging | `progress_log_interval = 0` |

`train_file` may be a list. `train_repeats` supplies one positive integer per
training source; omission means one copy of each. The test dataset is a single
file and is not repeated. See [mixed data](training.md#mixed-training-data).

### Learning-rate schedules

| `scheduler.name` | Parameters |
| --- | --- |
| `none` (default) | Constant learning rate |
| `cosine` | `min_learning_rate = 0.0`; the period uses `optimizer.epochs` |
| `step` | `step_size = 100`, `gamma = 0.5` |
| `plateau` | `patience = 20`, `factor = 0.5`, `min_learning_rate = 0.0` |

Schedulers advance at epoch boundaries. Changing a schedule changes training
history; full-state resume checks its compatibility. A new fine-tuning stage
selects its own optimizer and scheduler. For fine-tuning, set
`[finetune] checkpoint = "path/to/model.pt"`; do not manually enable
`runtime.resume_model_only`. The historical
[Local](../models/mobius-oam-loc/recipes/finetune/rune.toml),
[MP](../models/mobius-oam-mp/recipes/finetune/rune.toml), and
[EQ](../models/mobius-oam-eq/recipes/finetune/rune.toml) configurations preserve
the corresponding trained model parameters. Follow the
[fine-tuning guide](training.md#fine-tune-a-trained-model) to copy one into a
new run directory and update its checkpoint, data, and training settings.

## Model plugin interface

Every model directory contains a `plugin.py` with an action-explicit contract:

```python
PLUGIN_NAME = "my-model"

def resolve_params(values: dict) -> dict: ...
def build_model(params: dict): ...
def get_ddp_options(params: dict) -> dict: ...
```

`resolve_params()` applies model defaults, validates user values, and returns a
serializable dictionary. `build_model()` constructs a `torch.nn.Module`.
`get_ddp_options()` returns only keyword arguments for
`DistributedDataParallel`.

A model may expose additional capabilities when they exist:

```python
def load_model(path, device=None): ...
def export_model(model, path): ...
def build_calculator(artifact, **kwargs): ...
```

NEP and Möbius are independent model folders. Möbius owns its descriptor and
force contraction and does not import `rune.nep`.

## Adding a model

To develop another built-in model:

1. create a complete directory such as `rune/example/`;
2. implement `PLUGIN_NAME`, `resolve_params()`, `build_model()`, and
   `get_ddp_options()` in `plugin.py`;
3. keep the model's numerical code and external interfaces inside that
   directory;
4. set `plugin = "rune.example"` in `rune.toml`;
5. start it through the same CLI.

Shared code must not need a model-name switch. Model packages must not import
their sibling model packages.
