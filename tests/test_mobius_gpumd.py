"""End-to-end native CUDA checks against the Python energy and its derivatives.

Set RUNE_GPUMD_BINARY to the separately built GPUMD executable. Optional
RUNE_GPUMD_TEST_ROOT retains every input/output for auditing. The native harness
uses the selected CUDA device architecture; RUNE_GPUMD_TEST_CUDA_ARCH can override
it (for example sm_80), including with CPU-only PyTorch. Without a binary, the
integration checks are skipped; the installer still runs on CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.io import write
from ase.neighborlist import neighbor_list
import torch

from rune.mobius.export import export_model
from rune.mobius.install_gpumd import install_gpumd
from rune.mobius.model import Mobius
from rune.mobius.plugin import load_model, resolve_params


def model_for(mode, **changes):
    config = dict(elements=["Fe", "C"], hidden_dim=8, ffn_dim=12, blocks=1,
                  message_layers=0 if mode == "local" else 2, message_dim=8,
                  message_groups=2, pair_dim=3, vector_channels=2,
                  message_basis_size=3, rcut_radial=3.8, rcut_angular=3.5,
                  n_max_radial=1, n_max_angular=1,
                  basis_size_radial=2, basis_size_angular=2,
                  l_max=4, l_max_4body=2, l_max_5body=1,
                  has_q_112=1, has_q_123=1, has_q_233=1, has_q_134=1,
                  max_neighbors_radial=128, max_neighbors_angular=128)
    if mode == "equivariant":
        config.update(message_mode="equivariant", equivariant_channels=3,
                      equivariant_l_max=4)
    config.update(changes)
    torch.manual_seed(20261021)
    model = Mobius(resolve_params(config)).double().eval()
    model.set_descriptor_scaler(torch.linspace(.7, 1.2, model.input_dim, dtype=torch.float64))
    return model


def exact_batch(atoms, params):
    """Use float64 periodic shifts; the production ASE builder stores float32."""
    positions = np.asarray(atoms.positions, dtype=np.float64)
    batch = dict(positions=torch.tensor(positions),
                 types=torch.tensor([params["elements"].index(s) for s in atoms.symbols]),
                 n_atoms_per_structure=torch.tensor([len(atoms)]))
    for name in ("radial", "angular"):
        centers, neighbors, shifts = neighbor_list("ijS", atoms, params[f"rcut_{name}"])
        counts = np.bincount(centers, minlength=len(atoms))
        width = max(1, int(counts.max()))
        indices = np.full((len(atoms), width), -1, dtype=np.int64)
        offsets = np.zeros((len(atoms), width, 3), dtype=np.float64)
        slots = np.zeros(len(atoms), dtype=int)
        for i, j, shift in zip(centers, neighbors, shifts):
            indices[i, slots[i]] = j
            offsets[i, slots[i]] = shift @ np.asarray(atoms.cell)
            slots[i] += 1
        batch[f"{name}_neighbors"] = torch.tensor(indices)
        batch[f"{name}_offsets"] = torch.tensor(offsets)
    return batch


@contextmanager
def run_directory(name):
    root = os.environ.get("RUNE_GPUMD_TEST_ROOT")
    if root:
        path = Path(tempfile.mkdtemp(prefix=name + "-", dir=root))
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix="mobius-gpumd-" + name) as value:
            yield Path(value)


class InstallerTests(unittest.TestCase):
    def test_optional_install_is_idempotent(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "src/force").mkdir(parents=True)
            (root / "src/main_gpumd").mkdir()
            (root / "src/force/force.cu").write_text(
                '#include "force.cuh"\nconst std::vector<std::string>& tokens;\n'
                '  if (strcmp(potential_name, "tersoff_1989") == 0) {}\n')
            (root / "src/force/force.cuh").write_text("class Force { public:\n  Force(void);\n};\n")
            (root / "src/main_gpumd/run.cu").write_text(
                'void Run::parse_one_keyword(const std::vector<std::string>& tokens)\n{\n}\n')
            (root / "CMakeLists.txt").write_text("add_executable(gpumd source.cu)\n")
            install_gpumd(root)
            initial = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            install_gpumd(root)
            self.assertEqual(initial, {p.relative_to(root): p.read_bytes()
                                       for p in root.rglob("*") if p.is_file()})
            self.assertIn('option(GPUMD_MOBIUS "Build the RUNE Möbius potential" OFF)',
                          (root / "CMakeLists.txt").read_text())
            self.assertIn("check_mobius_heat_current", (root / "src/main_gpumd/run.cu").read_text())


@unittest.skipUnless(os.environ.get("RUNE_GPUMD_BINARY"), "RUNE_GPUMD_BINARY is not configured")
class NativeGPUMDTests(unittest.TestCase):
    def test_cached_neighbor_rebuilds_and_geometry_frames(self):
        # Direct frames cover cell/PBC/type changes that fixed-cell NVE cannot.
        source = Path(os.environ["RUNE_GPUMD_BINARY"]).resolve().parent
        nvcc = shutil.which("nvcc")
        objects = sorted((source / "force/mobius").glob("*.o"))
        objects += [source / p for p in ("force/neighbor.o", "force/potential.o", "model/box.o")]
        if not nvcc or len(objects) != 9 or not all(p.exists() for p in objects):
            self.skipTest("Neighbor harness requires nvcc and the matching GPUMD build objects")
        architecture = os.environ.get("RUNE_GPUMD_TEST_CUDA_ARCH")
        if not architecture:
            if not torch.cuda.is_available():
                self.skipTest("Set RUNE_GPUMD_TEST_CUDA_ARCH with CPU-only PyTorch")
            major, minor = torch.cuda.get_device_capability()
            architecture = f"sm_{major}{minor}"
        repository = Path(__file__).resolve().parents[1]
        with run_directory("neighbor-cache") as path:
            binary = path / "neighbor-check"
            command = [nvcc, "-std=c++14", "-O3", f"-arch={architecture}", "-I", str(source),
                       str(repository / "tests/native/mobius_neighbor.cu"),
                       *map(str, objects), "-lcublas", "-o", str(binary)]
            build = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=120)
            (path / "build.log").write_text(build.stdout)
            self.assertEqual(build.returncode, 0, build.stdout)
            check = subprocess.run([sys.executable, str(repository / "tests/mobius_neighbor_check.py"),
                                    "--repo", str(repository), "--binary", str(binary),
                                    "--output", str(path / "frames")],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=180)
            (path / "check.log").write_text(check.stdout)
            self.assertEqual(check.returncode, 0, check.stdout)

    def run_model(self, model, atoms, name, steps=1, extra="", expect_failure=False,
                  velocities=None, time_step=.001, frames=None):
        with run_directory(name) as path:
            export_model(model, path / "mobius.txt")
            # Explicit velocities make the short MD test deterministic (A/fs).
            atoms = atoms.copy()
            atoms.set_array("vel", np.zeros((len(atoms), 3)) if velocities is None
                            else np.asarray(velocities, dtype=float))
            write(path / "model.xyz", atoms, format="extxyz")
            (path / "run.in").write_text(
                f"potential mobius.txt\nensemble nve\ntime_step {time_step}\n" + extra +
                "dump_xyz 1 native.xyz precision double force potential virial\n"
                f"run {steps}\n")
            result = subprocess.run([os.environ["RUNE_GPUMD_BINARY"]], cwd=path,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=120)
            (path / "gpumd.log").write_text(result.stdout)
            if expect_failure:
                self.assertNotEqual(result.returncode, 0, result.stdout)
                return result.stdout
            self.assertEqual(result.returncode, 0, result.stdout[-6000:])
            lines = (path / "native.xyz").read_text().splitlines()
            self.assertEqual(len(lines), steps * (len(atoms)+2))
            errors = dict(energy=0., force=0., virial=0.)
            for step in range(steps):
                start = step*(len(atoms)+2)+2
                data = np.array([[float(v) for v in line.split()[1:]]
                                 for line in lines[start:start+len(atoms)]])
                frame = atoms.copy()
                frame.positions = data[:, :3]
                if frames is not None:
                    frames.append(frame)
                reference = model(exact_batch(frame, model.params))
                expected_e = reference["energies"].detach().numpy().reshape(-1)
                expected_f = reference["forces"].detach().numpy()
                expected_v = reference["virial"].detach().numpy().reshape(3,3).T
                actual_e, actual_f = data[:,6], data[:,3:6]
                actual_v = data[:,7:16].sum(axis=0).reshape(3,3)
                for key, actual, expected in (("energy",actual_e,expected_e),
                                               ("force",actual_f,expected_f),
                                               ("virial",actual_v,expected_v)):
                    errors[key] = max(errors[key], float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual, expected, rtol=2.e-9, atol=2.e-9,
                                               err_msg=f"{name} {key} step {step}")
            (path / "errors.json").write_text(json.dumps(errors, indent=2)+"\n")
            return errors

    @staticmethod
    def small_atoms():
        return Atoms("FeCFeC", scaled_positions=[[.10,.13,.22],[.82,.21,.37],
                      [.30,.75,.19],[.68,.63,.81]],
                     cell=[[4.1,.1,.2],[.0,4.3,.1],[.1,.0,4.5]], pbc=True)

    def test_three_modes_triclinic_periodic_images(self):
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                self.run_model(model_for(mode), self.small_atoms(), mode+"-small")

    def test_three_modes_large_cell_gpu_bins(self):
        atoms = Atoms("FeCFeCFeC", positions=[[.2,.3,.4],[19.2,.6,.4],[2.,1.,.7],
                      [8.,7.,9.],[9.8,7.2,8.1],[14.,14.,14.]],
                      cell=[[20.,.2,.4],[0.,20.5,.3],[.2,0.,21.]], pbc=True)
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                self.run_model(model_for(mode), atoms, mode+"-large")

    def test_empty_edges_and_runtime_model_dimensions(self):
        atoms = Atoms("FeC", positions=[[2.,2.,2.],[12.,12.,12.]], cell=[30.,30.,30.], pbc=True)
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                model = model_for(mode, hidden_dim=16, ffn_dim=24, message_dim=12,
                                  blocks=2, activation="tanh")
                self.run_model(model, atoms, mode+"-empty")

    def test_angular_cutoff_above_radial_and_zbl(self):
        model = model_for("equivariant", rcut_radial=3., rcut_angular=4.2,
                          zbl=[.8,2.8], equivariant_l_max=2)
        self.run_model(model, self.small_atoms(), "angular-zbl")

    def test_short_nve_matches_reference_each_step(self):
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                self.run_model(model_for(mode), self.small_atoms(), mode+"-nve", steps=10)

    def test_non_warp_aligned_network_channels(self):
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                self.run_model(model_for(mode, hidden_dim=36, ffn_dim=53,
                    message_dim=15, message_groups=3, blocks=2),
                    self.small_atoms(), mode+"-warp-tail")

    def test_descriptor_derivatives_across_angular_orders(self):
        for order in (1, 2, 6, 8):
            with self.subTest(order=order):
                model = model_for("equivariant", l_max=order,
                    equivariant_l_max=min(order, 3),
                    l_max_4body=2 if order >= 2 else 0,
                    has_q_112=int(order >= 2), has_q_123=int(order >= 3),
                    has_q_233=int(order >= 3), has_q_134=int(order >= 4))
                self.run_model(model, self.small_atoms(), f"derivative-order-{order}")

    def test_radial_channels_and_basis_group_boundaries(self):
        # Exercise partial subwarps, multiple channels per lane, and long basis
        # recurrences in the same FP64 energy/force/virial reference check.
        for channels in (2, 3, 5, 17, 33):
            with self.subTest(channels=channels):
                model = model_for("local", n_max_radial=channels-1,
                    n_max_angular=channels-1, basis_size_radial=20,
                    basis_size_angular=18)
                self.run_model(model, self.small_atoms(), f"radial-channels-{channels}")

    def test_neighbors_enter_cutoff_within_skin_displacement(self):
        # The pair starts outside the true cutoff but inside the skin. Each
        # atom moves only ~0.3 A, so a 1 A candidate skin remains reusable
        # while radial/message edges must change from absent to present.
        atoms = Atoms("FeC", positions=[[15.,15.,15.], [19.2,15.,15.]],
                      cell=[40.,40.,40.], pbc=True)
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                model = model_for(mode)
                frames = []
                self.run_model(model, atoms, mode+"-cutoff-within-skin", steps=60,
                    velocities=[[.5,0.,0.],[-.5,0.,0.]], time_step=.01, frames=frames)
                counts = [len(neighbor_list("i", frame, model.params["rcut_radial"]))
                          for frame in [atoms, *frames]]
                self.assertEqual(counts[0], 0)
                self.assertEqual(counts[-1], 2)

    def test_workspace_reuse_with_changing_and_empty_neighbors(self):
        # A fast atom passes two separated atoms, then leaves both cutoffs.
        # This exercises 0 -> 2 -> 4 -> 2 -> 0 directed edges in one evaluator;
        # retained accumulators must not leak into later, isolated configurations.
        # The imposed velocity is a numerical regression, not a physical MD setup.
        atoms = Atoms("FeCFe", positions=[[15.,15.,15.], [20.,15.,15.], [25.,16.3,15.]],
                      cell=[40.,40.,40.], pbc=True)
        velocities = [[0.,0.,0.], [0.,0.,0.], [-30.,0.,0.]]
        for mode in ("local", "scalar", "equivariant"):
            with self.subTest(mode=mode):
                model = model_for(mode, blocks=2)
                frames = []
                self.run_model(model, atoms, mode+"-workspace-reuse", steps=48,
                               velocities=velocities, time_step=.01, frames=frames)
                counts = {}
                for name in ("radial", "angular"):
                    counts[name] = [len(neighbor_list("i", frame, model.params[f"rcut_{name}"]))
                                    for frame in [atoms, *frames]]
                    self.assertEqual(counts[name][0], 0)
                    self.assertGreaterEqual(max(counts[name]), 4)
                    self.assertGreaterEqual(len(set(counts[name])), 3)
                    self.assertEqual(counts[name][-3:], [0, 0, 0])

    def test_mixed_pbc_and_additional_model_options(self):
        small = self.small_atoms()
        small.pbc = [True, False, True]
        self.run_model(model_for("local", activation="gelu"), small, "mixed-small-gelu")
        large = self.small_atoms()
        large.set_cell([[20.,.2,.4],[0.,20.5,.3],[.2,0.,21.]])
        large.pbc = [True, False, True]
        self.run_model(model_for("scalar", vector_channels=0), large, "mixed-large-vector0")
        coefficients = [.18175,3.1998,.50986,.94229,.28022,.4029,.02817,.20162]
        zbl = dict(rc_inner=0., rc_outer=0., flexible=True,
                   parameters=[[.5,2.8,*coefficients]]*3)
        self.run_model(model_for("equivariant", zbl=zbl), small, "flexible-zbl")

    def test_message_heat_current_is_rejected(self):
        for mode in ("scalar", "equivariant"):
            with self.subTest(mode=mode):
                output = self.run_model(model_for(mode), self.small_atoms(), mode+"-heat",
                    extra="compute_hnemd 1 0.001 0 0\n", expect_failure=True)
                self.assertIn("do not yet support heat-current/HNEMD", output)
        output = self.run_model(model_for("equivariant"),self.small_atoms(),"eq-compute-jp",
            extra="compute 1 1 jp\n",expect_failure=True)
        self.assertIn("do not yet support heat-current/HNEMD",output)

    def test_neighbor_capacity_is_rejected(self):
        output = self.run_model(model_for("local",max_neighbors_radial=1),self.small_atoms(),
                                "neighbor-capacity",expect_failure=True)
        self.assertIn("neighbor capacity exceeded",output)

    @unittest.skipUnless(os.environ.get("RUNE_GPUMD_CHECKPOINT"),"No trained checkpoint selected")
    def test_trained_checkpoint_weights_and_runtime_dimensions(self):
        model=load_model(os.environ["RUNE_GPUMD_CHECKPOINT"],device="cpu").double().eval()
        self.run_model(model,self.small_atoms(),"trained-checkpoint")

    def test_positions_outside_multiple_periods(self):
        atoms=Atoms("FeCFeC",positions=[[1.,1.,1.],[64.7,1.,1.],[8.,8.,8.],[-55.8,8.,8.]],
                    cell=[20.9,20.9,20.9],pbc=True)
        self.run_model(model_for("equivariant"),atoms,"unwrapped-large")

    def test_existing_gpumd_lj_potential_still_runs(self):
        upstream = Path(os.environ["RUNE_GPUMD_BINARY"]).resolve().parents[1]
        source = upstream / "potentials/lj/Ar_10A.txt"
        if not source.exists():
            self.skipTest("Official GPUMD Ar LJ fixture is absent")
        with run_directory("stock-lj") as path:
            shutil.copyfile(source,path/"lj.txt")
            atoms=Atoms("Ar4",positions=[[1.,1.,1.],[5.,1.,1.],[1.,5.,1.],[1.,1.,5.]],
                        cell=[30.,30.,30.],pbc=True)
            atoms.set_array("vel",np.zeros((4,3)))
            write(path/"model.xyz",atoms,format="extxyz")
            (path/"run.in").write_text("potential lj.txt\nensemble nve\ntime_step 0.001\n"
                "dump_xyz 1 native.xyz precision double force potential virial\nrun 5\n")
            result=subprocess.run([os.environ["RUNE_GPUMD_BINARY"]],cwd=path,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            (path/"gpumd.log").write_text(result.stdout)
            self.assertEqual(result.returncode,0,result.stdout)
            lines=(path/"native.xyz").read_text().splitlines()
            self.assertEqual(len(lines),30)
            for start in range(2,len(lines),6):
                values=np.array([[float(v) for v in line.split()[1:]] for line in lines[start:start+4]])
                self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
