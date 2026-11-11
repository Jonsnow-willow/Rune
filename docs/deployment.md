# Export and GPUMD deployment

[Documentation](README.md) · [Inference with ASE](inference.md)

The model directories distribute `model.pt` weights. To deploy one in GPUMD,
export `mobius.txt` yourself, compile GPUMD with the Möbius CUDA adapter, and
prepare your simulation inputs. GPUMD reads the exported text potential;
the `.pt` file is used by RUNE's Python interface.

Run the preparation commands below from the RUNE project root, using the
Python environment configured for [ASE inference](inference.md). Export works
on CPU. Compile GPUMD with a CUDA toolchain targeting your NVIDIA GPU, then
run simulations on that GPU. Native inference does not require Python,
PyTorch, or LibTorch.

## Export a checkpoint

Generate a text potential from the bundled EQ weight:

```bash
python - <<'PY'
from rune.mobius.plugin import load_model, export_model

model = load_model("models/mobius-oam-eq/model.pt", device="cpu")
export_model(model, "runs/gpumd-eq/mobius.txt")
PY
```

The exporter creates the output directory. Select another model by changing
the input and output paths:

| Model | Input weight | Suggested output |
| --- | --- | --- |
| Local | `models/mobius-oam-loc/model.pt` | `runs/gpumd-loc/mobius.txt` |
| MP | `models/mobius-oam-mp/model.pt` | `runs/gpumd-mp/mobius.txt` |
| EQ | `models/mobius-oam-eq/model.pt` | `runs/gpumd-eq/mobius.txt` |

The same API accepts a full Möbius training checkpoint, including one from
your own fine-tuning run. Export preserves the model values; it does not
distill the model or fit a different potential.

## Compile GPUMD with Möbius

Use a separate GPUMD checkout and install the adapter sources. The integration
targets official revision `fd805e1b01f31138039b64d6b3342a31cd8649aa`:

```bash
git clone --no-checkout https://github.com/brucefan1983/GPUMD.git runs/GPUMD
git -C runs/GPUMD checkout fd805e1b01f31138039b64d6b3342a31cd8649aa
python -m rune.mobius.install_gpumd runs/GPUMD
```

The installer copies the native sources from `rune/mobius/src/` into this
checkout and adds registration and build hooks. It does not compile GPUMD.
Build the executable yourself with a CUDA toolchain suitable for the selected
GPUMD revision and deployment GPU:

```bash
make -C runs/GPUMD/src -f makefile_mobius gpumd
```

Local, MP and EQ use the same evaluator; model dimensions and weights are read
from `mobius.txt` at runtime. The enabled build retains other potential types
and uses FP64 arithmetic and cuBLAS for the Möbius evaluator.

Alternatively, after installing the adapter sources, build with CMake:

```bash
cmake -S runs/GPUMD -B runs/GPUMD/build-mobius \
  -DGPUMD_MOBIUS=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build runs/GPUMD/build-mobius --target gpumd -j 4
```

Architecture `80` selects the A100; select the architecture of your GPU.
With `GPUMD_MOBIUS` disabled, the native evaluator is not compiled.

## Run a simulation

Prepare your GPUMD structure and `run.in` alongside the exported potential.
For the EQ example, the simulation directory should contain:

```text
runs/gpumd-eq/
├── mobius.txt
├── model.xyz
└── run.in
```

Select the potential in `run.in` with:

```text
potential mobius.txt
```

Complete the other GPUMD inputs for your system, ensemble, temperature, time
step, and requested outputs, then run the built executable in that simulation
directory. The evaluator calculates atomic energies, conservative forces, and
virials, including periodic images in small cells and the configured ZBL term.

Partial-system potential assignments are not supported. Exceeding a configured
neighbor limit produces an error rather than silently truncating the list.
MP and EQ reject heat-current observables (`compute_hac`, `compute_hnemd`,
`compute_hnemdec`, `compute_shc`, and `compute ... jp`), because conservative
forces and total virial do not establish the required multi-hop atomic
heat-current partition. Validation and performance results are discussed in
the [technical report](../paper/paper.md).

## NEP export

The bundled OAM models are Möbius models and export to `mobius.txt`. If you
have trained a NEP4 model, export its checkpoint to `nep.txt` instead:

```bash
python - <<'PY'
from rune.nep.plugin import load_model, export_model

model = load_model("runs/nep/checkpoints/best.pt", device="cpu")
export_model(model, "runs/nep/nep.txt")
PY
```

Place that file with your GPUMD inputs and select `potential nep.txt`. NEP4 uses
the standard GPUMD reader and does not require the Möbius adapter. RUNE currently
supports the single-hidden-layer NEP4 text layout; see
[NEP configuration](configuration.md#nep4-parameters) for supported dimensions,
ZBL settings, and checkpoint compatibility.
