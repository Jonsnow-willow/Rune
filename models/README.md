# Möbius-OAM models

Three trained Möbius models are included in this repository. Each directory
contains a `model.pt` weight, a model card, and two training configurations.
OAM denotes OMat24 pretraining followed by fine-tuning on sAlex and MPTrj.

| Model | Parameters | Pretraining completed / planned | Matbench CPS |
|---|---:|---|---:|
| [mobius-oam-loc](mobius-oam-loc/README.md) | 2,977,803 | 6  cosine epochs | 0.680395 |
| [mobius-oam-mp](mobius-oam-mp/README.md) | 3,327,499 | 6  cosine epochs | 0.772360 |
| [mobius-oam-eq](mobius-oam-eq/README.md) | 5,001,403 | 6  cosine epochs | 0.840992 |

All three completed one mixed fine-tuning epoch with sAlex once and MPTrj
eight times. Here MP means message passing; it is distinct from the MPTrj
dataset and from the separately trained MPTrj model.

Local uses local many-body invariant descriptors and a shared scalar network.
MP adds grouped scalar messages and contracts transient directional aggregates
within each block. EQ carries angular directional state between message blocks.
The models differ in capacity and training schedule; their results do not isolate
an architectural effect. See the [architecture documentation](../docs/architecture.md).

## Predict

After [installing RUNE](../README.md), run from the repository root:

```python
from ase.io import read
from rune.mobius.plugin import build_calculator

atoms = read("models/example.extxyz", index=0)
atoms.calc = build_calculator("models/mobius-oam-eq/model.pt", device="cpu")
print("Energy (eV):", atoms.get_potential_energy())
print("Forces (eV/angstrom):", atoms.get_forces())
print("Stress (eV/angstrom^3):", atoms.get_stress())
```

Replace `eq` with `loc` or `mp` to use another model. [example.extxyz](example.extxyz)
contains three synthetic fixtures for checking interfaces; it is not a scientific
test dataset. The model cards give Python and native deployment usage.

The distributed weights retain all learned prediction parameters from the
selected epoch-1 checkpoints. They support inference and initialization of a
new fine-tuning stage; optimizer, scheduler and RNG state are omitted. They do
not support exact resumption of the original training run. For GPUMD, generate
`mobius.txt` from `model.pt` and compile GPUMD with the Möbius CUDA adapter as
described in the [deployment guide](../docs/deployment.md).

## Elements

All three encode these 89 elements, in this order:

```text
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar
K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf
Ta W Re Os Ir Pt Au Hg Tl Pb Bi Ac Th Pa U Np Pu
```

Encoding an element does not establish accuracy for every composition, phase
or thermodynamic condition containing it. No independent spin or charge state
is exposed by the models.
