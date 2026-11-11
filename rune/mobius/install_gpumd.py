"""Add the optional native Möbius evaluator to a separate GPUMD source copy.

Usage: python -m rune.mobius.install_gpumd /path/to/GPUMD-copy
Then configure that copy with ``-DGPUMD_MOBIUS=ON``. This command neither
builds nor replaces an installed GPUMD executable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


def _once(text, marker, anchor, replacement):
    if marker in text:
        return text
    if text.count(anchor) != 1:
        raise ValueError(f"Unsupported GPUMD source: expected one {anchor!r}.")
    return text.replace(anchor, replacement, 1)


def install_gpumd(path):
    root = Path(path).resolve()
    native = Path(__file__).with_name("src")
    paths = {
        "force": root / "src/force/force.cu",
        "header": root / "src/force/force.cuh",
        "run": root / "src/main_gpumd/run.cu",
        "cmake": root / "CMakeLists.txt",
    }
    texts = {key: value.read_text(encoding="utf-8") for key, value in paths.items()}
    force = texts["force"]
    force = _once(force, '#include "mobius/mobius.cuh"', '#include "force.cuh"',
                  '#include "force.cuh"\n#ifdef GPUMD_MOBIUS\n#include "mobius/mobius.cuh"\n#endif')
    filename = "tokens[1].c_str()" if "const std::vector<std::string>& tokens" in force else "param[1]"
    anchor = '  if (strcmp(potential_name, "tersoff_1989") == 0) {'
    force = _once(force, 'strcmp(potential_name, "mobius1")', anchor,
        '#ifdef GPUMD_MOBIUS\n'
        '  if (strcmp(potential_name, "mobius1") == 0) {\n'
        f'    potential.reset(new Mobius({filename}, number_of_atoms));\n'
        '    if (mobius_heat_current_requested_ &&\n'
        '        !static_cast<Mobius*>(potential.get())->supports_heat_current())\n'
        '      PRINT_INPUT_ERROR("Möbius message models do not yet support heat-current/HNEMD observables.");\n'
        '  } else\n#endif\n' + anchor)
    if "void Force::check_mobius_heat_current" not in force:
        force += '''
#ifdef GPUMD_MOBIUS
void Force::check_mobius_heat_current(bool requested)
{
  mobius_heat_current_requested_ = mobius_heat_current_requested_ || requested;
  if (!mobius_heat_current_requested_) return;
  for (const auto& potential : potentials) {
    const auto* model = dynamic_cast<const Mobius*>(potential.get());
    if (model && !model->supports_heat_current())
      PRINT_INPUT_ERROR("Möbius message models do not yet support heat-current/HNEMD observables.");
  }
}
#endif
'''
    header = _once(texts["header"], "mobius_heat_current_requested_", "  Force(void);",
        '  Force(void);\n#ifdef GPUMD_MOBIUS\n'
        '  void check_mobius_heat_current(bool requested);\n'
        '  bool mobius_heat_current_requested_ = false;\n#endif')
    run = texts["run"]
    marker = "// Möbius multi-hop heat-current boundary."
    insertion = '''
#ifdef GPUMD_MOBIUS
  // Möbius multi-hop heat-current boundary.
  bool mobius_heat = tokens[0] == "compute_hac" || tokens[0] == "compute_hnemd" ||
                           tokens[0] == "compute_hnemdec" || tokens[0] == "compute_shc";
  if (tokens[0] == "compute")
    for (const auto& token : tokens) mobius_heat = mobius_heat || token == "jp";
  force.check_mobius_heat_current(mobius_heat);
#endif
'''
    if marker in run:
        run = re.sub(r"#ifdef GPUMD_MOBIUS\n  // Möbius multi-hop heat-current boundary\.[\s\S]*?#endif\n",
                     insertion.lstrip("\n"), run, count=1)
    else:
        pattern = r"(void Run::parse_one_keyword\([^{}]+\)\s*\{)"
        matches = list(re.finditer(pattern, run))
        if len(matches) != 1:
            raise ValueError("Unsupported GPUMD Run::parse_one_keyword signature.")
        position = matches[0].end()
        run = run[:position] + insertion + run[position:]
    cmake = texts["cmake"]
    if 'option(GPUMD_MOBIUS ' not in cmake:
        cmake += '''
# Optional RUNE Möbius native CUDA evaluator.
option(GPUMD_MOBIUS "Build the RUNE Möbius potential" OFF)
if(GPUMD_MOBIUS)
  file(GLOB MOBIUS_SOURCES CONFIGURE_DEPENDS src/force/mobius/*.cu)
  target_sources(gpumd PRIVATE ${MOBIUS_SOURCES})
  target_compile_definitions(gpumd PRIVATE GPUMD_MOBIUS=1)
endif()
'''
    source_files = sorted(native.glob("*.cu")) + sorted(native.glob("*.cuh"))
    required = {"mobius.cu", "mobius.cuh", "kernels.cuh", "network.cu", "descriptor.cu",
                "message.cu", "equivariant.cu", "force.cu"}
    if not required.issubset({p.name for p in source_files}):
        raise ValueError("The complete native Möbius source set is required before installation.")
    # Validate every patch before changing the explicitly selected source copy.
    for key, value in {"force": force, "header": header, "run": run, "cmake": cmake}.items():
        if value != texts[key]:
            paths[key].write_text(value, encoding="utf-8")
    target = root / "src/force/mobius"
    target.mkdir(exist_ok=True)
    for source in source_files:
        destination = target / source.name
        if not destination.exists() or source.read_bytes() != destination.read_bytes():
            # The source checkout can have older timestamps than compiled
            # objects; changed content must get a fresh destination timestamp.
            shutil.copyfile(source, destination)
    (root / "src/makefile_mobius").write_text(
        "# Use only in an independent GPUMD source copy.\n"
        "include makefile\n"
        "CFLAGS += -DGPUMD_MOBIUS=1\n"
        "MOBIUS_OBJECTS = $(patsubst %.cu,%.o,$(wildcard force/mobius/*.cu))\n"
        "$(MOBIUS_OBJECTS) force/force.o: $(wildcard force/mobius/*.cuh)\n"
        "gpumd: $(MOBIUS_OBJECTS)\n", encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpumd_source", type=Path)
    args = parser.parse_args()
    root = install_gpumd(args.gpumd_source)
    print(f"Installed optional Möbius sources in {root}.")
    print("Configure an independent build with -DGPUMD_MOBIUS=ON.")
    print("Alternatively, in its src directory: make -f makefile_mobius gpumd.")


if __name__ == "__main__":
    main()
