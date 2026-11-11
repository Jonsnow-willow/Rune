# Installation

[Documentation](README.md) · [Use a model](inference.md)

RUNE requires Python 3.10 or newer. Installation includes PyTorch, NumPy, ASE,
and tomli on Python 3.10. The three trained Möbius models and their training
configurations live under `models/` in the repository; they are not included in
the installed wheel. Run the commands and examples below from the project root.

## Set up the environment

Create an environment and install RUNE:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
rune --help
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
For editable development, use `python -m pip install -e .` instead of
`python -m pip install .`.

Keep the environment active. `rune` and `python -m rune.cli.run` invoke the
same CLI; for example, `python -m rune.cli.run --help` shows the same commands.
With the dependencies installed, the module CLI and Python API can also run
directly from the source checkout.

## CPU and CUDA

The Python inference examples use `device="cpu"`. For a CUDA GPU, install
the PyTorch build appropriate to your driver and platform in the same
environment as the other dependencies. Check PyTorch's view of the device with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Select `device="cuda"` in the ASE interface or `--device cuda` on a training
command. PyTorch training and inference do not require GPUMD. Native GPUMD
inference uses the separate CUDA build described in
[deployment](deployment.md).

## First prediction

The model files are already in the checkout. Create a silicon cell with ASE
and evaluate it with the EQ model:

```python
from ase.build import bulk
from rune.mobius.plugin import build_calculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = build_calculator("models/mobius-oam-eq/model.pt", device="cpu")

print("Energy (eV):", atoms.get_potential_energy())
print("Forces (eV/Å):", atoms.get_forces())
print("Stress (eV/Å³):", atoms.get_stress(voigt=False))
```

Replace `atoms` with your own ASE structure, for example loaded with
`ase.io.read`. See [inference](inference.md) for GPU prediction and model choices.

## Tests

After installing RUNE, run the suite from the project root. The tests use
`unittest` and the runtime dependencies:

```bash
python -m unittest discover -s tests -v
```

Native integration tests require a separately built GPUMD binary selected by
`RUNE_GPUMD_BINARY`. CPU-only environments skip those tests. Validation and
benchmark results are discussed in the [technical report](../paper/paper.md).
See [training](training.md) for training and continuation instructions.
