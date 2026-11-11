"""Direct native E/F/virial and neighbor-cache regression; no MD integration.

python mobius_neighbor_harness.py --repo /path/to/Rune \
    --binary /path/to/mobius_neighbor_harness --output /path/to/new/result-folder

Add --prepare-only to export inputs and compute CPU references without GPU work.
The default three small, fixed models exercise the same compiled implementation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
import torch


def frames_for_test():
    records = []
    atoms = Atoms("FeCFeC", positions=[[5., 5., 5.], [8.9, 5., 5.],
                  [17., 17., 17.], [22., 17., 17.]], cell=[30., 30., 30.], pbc=True)

    def add(label, action):
        records.append((label, action, atoms.copy()))

    add("initial_empty_with_cutoff_and_skin_candidates", "build")
    atoms.positions[1, 0] = 8.7
    add("enter_radial_cutoff_without_rebuild", "reuse")
    atoms.positions[1, 0] = 8.45
    add("enter_angular_cutoff_without_rebuild", "reuse")
    atoms.positions[1, 0] = 8.95
    add("leave_both_cutoffs_without_rebuild", "reuse")
    atoms.positions[3, 0] = 20.7
    add("new_pair_enters_from_outside_skin", "build")
    atoms.set_chemical_symbols("CFeCFe")
    add("types_change_without_geometry_change", "reuse")
    atoms.set_cell([31., 31., 31.])
    add("box_change", "build")
    atoms.set_pbc([True, False, True])
    add("pbc_change", "build")
    atoms.positions[0, 0] += 3*31.
    atoms.positions[1, 0] -= 2*31.
    atoms.positions[2, 2] += 4*31.
    atoms.positions[3, 2] -= 3*31.
    add("independent_multiple_box_translations", "reuse")
    atoms.wrap()
    atoms.positions[0, 0] = 30.9
    atoms.positions[1, 0] = 3.2
    add("boundary_crossing_setup", "build")
    atoms.positions[0, 0] = 31.1
    add("cross_periodic_boundary_without_rebuild", "reuse")
    atoms.wrap()
    add("explicit_coordinate_rewrap", "reuse")
    atoms.set_cell([[31., .3, .5], [.1, 30., .2], [.4, .2, 32.]])
    atoms.set_pbc(True)
    add("triclinic_box_change", "build")
    atoms.positions += [.05, .03, .02]
    add("triclinic_small_displacement", "reuse")
    atoms.set_cell([30., 30., 30.])
    atoms.positions = [[3., 3., 3.], [12., 3., 3.], [3., 12., 3.], [3., 3., 12.]]
    add("return_to_empty_edges", "build")
    atoms.positions += .05
    add("reuse_empty_candidate_list", "reuse")
    atoms.set_cell([[4.1, .1, .2], [0., 4.3, .1], [.1, 0., 4.5]])
    atoms.set_scaled_positions([[.10, .13, .22], [.82, .21, .37],
                               [.30, .75, .19], [.68, .63, .81]])
    add("small_box_multiple_periodic_images", "small")
    atoms.positions[1] += [.02, -.01, .01]
    add("small_box_repeated_call", "small")
    atoms.set_cell([30., 30., 30.])
    atoms.set_pbc(False)
    atoms.positions = [[3., 3., 3.], [12., 3., 3.], [3., 12., 3.], [3., 3., 12.]]
    add("fully_nonperiodic_empty_edges", "small")
    atoms.set_pbc(True)
    add("return_to_large_box_after_small_path", "build")
    add("reuse_after_small_path_rebuild", "reuse")
    return records


def write_frames(path, records, elements):
    with path.open("w") as handle:
        handle.write(f"mobius_neighbor_frames_v1 {len(records[0][2])} {len(records)}\n")
        for index, (_, _, atoms) in enumerate(records):
            handle.write(f"frame {index}\n")
            handle.write(" ".join(str(int(v)) for v in atoms.pbc)+"\n")
            handle.write(" ".join(f"{v:.17g}" for v in np.asarray(atoms.cell).ravel())+"\n")
            for symbol, position in zip(atoms.symbols, atoms.positions):
                handle.write(str(elements.index(symbol))+" "+
                             " ".join(f"{v:.17g}" for v in position)+"\n")


def read_results(path, n, frames):
    lines = path.read_text().splitlines()
    assert lines[0].split() == ["mobius_neighbor_results_v1", str(n), str(frames)]
    assert len(lines) == 1+frames*(n+1)
    records = []
    for index in range(frames):
        start = 1+index*(n+1)
        header = lines[start].split()
        assert header[:2] == ["frame", str(index)] and len(header) == 5
        data = np.asarray([[float(v) for v in line.split()]
                           for line in lines[start+1:start+1+n]])
        assert data.shape == (n, 13) and np.isfinite(data).all()
        records.append((np.asarray(header[2:], dtype=int), data))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=["local", "scalar", "equivariant"],
                        default=["local", "scalar", "equivariant"])
    args = parser.parse_args()
    if not args.prepare_only and args.binary is None:
        parser.error("--binary is required unless --prepare-only is set")
    sys.path.insert(0, str(args.repo.resolve()))
    from rune.mobius.export import export_model
    spec = importlib.util.spec_from_file_location("mobius_gpumd_reference",
                                                 args.repo / "tests/test_mobius_gpumd.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = frames_for_test()
    summary = {"precision": "float64", "rtol": 2e-9, "atol": 2e-9,
               "frames_per_mode": len(frames), "prepared_only": args.prepare_only, "modes": {}}
    delta_for = {"build": [1, 0, 0], "reuse": [0, 1, 0], "small": [0, 0, 1]}
    # Raw GPUMD virial order: xx yy zz xy xz yz yx zx zy.
    rows = np.asarray([0, 1, 2, 0, 0, 1, 1, 2, 2])
    cols = np.asarray([0, 1, 2, 1, 2, 2, 0, 0, 1])
    for mode in args.modes:
        folder = args.output / mode
        folder.mkdir(exist_ok=True)
        model = helpers.model_for(mode, blocks=2).cpu()
        export_model(model, folder / "mobius.txt")
        write_frames(folder / "frames.txt", frames, model.params["elements"])
        references = []
        for label, action, atoms in frames:
            reference = model(helpers.exact_batch(atoms, model.params))
            expected_v = reference["virial"].detach().numpy().reshape(3, 3).T
            references.append((reference["energies"].detach().numpy().reshape(-1),
                               reference["forces"].detach().numpy(), expected_v[rows, cols]))
        np.savez(folder / "reference.npz", energy=np.asarray([v[0] for v in references]),
                 force=np.asarray([v[1] for v in references]),
                 total_virial=np.asarray([v[2] for v in references]))
        manifest = [{"label": label, "expected_action": action,
                     "radial_edges": len(neighbor_list("i", atoms, model.params["rcut_radial"])),
                     "angular_edges": len(neighbor_list("i", atoms, model.params["rcut_angular"]))}
                    for label, action, atoms in frames]
        assert [v["radial_edges"] for v in manifest[:5]] == [0, 2, 2, 0, 2]
        assert [v["angular_edges"] for v in manifest[:5]] == [0, 0, 2, 0, 0]
        (folder / "frames.json").write_text(json.dumps(manifest, indent=2)+"\n")
        if args.prepare_only:
            summary["modes"][mode] = {"prepared": True, "frames": manifest}
            continue
        run = subprocess.run([str(args.binary.resolve()), str(folder / "mobius.txt"),
                              str(folder / "frames.txt"), str(folder / "native.txt")],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        (folder / "harness.log").write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f"{mode} native harness failed ({run.returncode}): {run.stdout[-4000:]}")
        actual = read_results(folder / "native.txt", len(frames[0][2]), len(frames))
        previous = np.zeros(3, dtype=int)
        mode_errors = {"energy": 0., "force": 0., "total_virial": 0.}
        for index, ((label, action, _), (stats, data), expected) in enumerate(zip(frames, actual, references)):
            np.testing.assert_array_equal(stats-previous, delta_for[action],
                                          err_msg=f"{mode} {label}: wrong cache path")
            previous = stats
            frame_errors = {}
            for key, native, reference in zip(mode_errors, (data[:, 0], data[:, 1:4], data[:, 4:].sum(axis=0)), expected):
                error = float(np.max(np.abs(native-reference)))
                mode_errors[key] = max(mode_errors[key], error)
                frame_errors[key] = error
                np.testing.assert_allclose(native, reference, rtol=2e-9, atol=2e-9,
                                           err_msg=f"{mode} frame {index} {label} {key}")
            manifest[index].update(stats=stats.tolist(), errors=frame_errors)
        summary["modes"][mode] = {"passed": True, "errors": mode_errors,
                                 "final_stats": previous.tolist(), "frames": manifest}
        (folder / "verification.json").write_text(json.dumps(summary["modes"][mode], indent=2)+"\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "frames_per_mode": len(frames),
                      "modes": {key: {k: v for k, v in value.items() if k != "frames"}
                                for key, value in summary["modes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
