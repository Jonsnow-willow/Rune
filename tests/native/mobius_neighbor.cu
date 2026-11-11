// Standalone correctness driver. Link to the same objects as the tested GPUMD.
// Usage: mobius_neighbor_harness MODEL INPUT OUTPUT
// Input: magic N frames; then frame index, PBC[3], ASE row-cell[9],
//        and N rows of type_index x y z. One Mobius survives all frames.
// Output: magic N frames; then frame index builds reuses small_calls,
//         and N rows of E Fx Fy Fz Vxx Vyy Vzz Vxy Vxz Vyz Vyx Vzx Vzy.
#include <cuda_runtime.h>
#include "force/mobius/mobius.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void expect(std::istream& stream, const char* expected) {
  std::string value;
  require(bool(stream >> value) && value == expected, "Unexpected input token");
}
double finite_value(std::istream& stream) {
  double value;
  require(bool(stream >> value) && std::isfinite(value), "Invalid input number");
  return value;
}
void write_finite(std::ostream& stream, double value) {
  require(std::isfinite(value), "Non-finite native result");
  stream << ' ' << value;
}
}

int main(int argc, char** argv) {
  try {
    require(argc == 4, "Usage: mobius_neighbor_harness MODEL INPUT OUTPUT");
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    require(bool(input) && bool(output), "Cannot open harness input/output");
    expect(input, "mobius_neighbor_frames_v1");
    int n = 0, frames = 0;
    require(bool(input >> n >> frames) && n > 0 && frames > 0, "Invalid frame shape");
    Mobius model(argv[1], n);
    GPU_Vector<int> types(n);
    GPU_Vector<double> position(3*size_t(n)), potential(n), force(3*size_t(n)), virial(9*size_t(n));
    std::vector<int> host_types(n);
    std::vector<double> host_position(3*size_t(n)), host_potential(n), host_force(3*size_t(n)), host_virial(9*size_t(n));
    output << std::setprecision(17) << "mobius_neighbor_results_v1 " << n << ' ' << frames << '\n';
    for (int frame = 0; frame < frames; ++frame) {
      expect(input, "frame");
      int index = -1;
      require(bool(input >> index) && index == frame, "Unexpected frame index");
      Box box;
      require(bool(input >> box.pbc_x >> box.pbc_y >> box.pbc_z), "Missing PBC flags");
      require((box.pbc_x == 0 || box.pbc_x == 1) && (box.pbc_y == 0 || box.pbc_y == 1) &&
                  (box.pbc_z == 0 || box.pbc_z == 1), "Invalid PBC flags");
      // ASE stores lattice vectors by row; GPUMD stores them by column.
      for (int lattice = 0; lattice < 3; ++lattice)
        for (int axis = 0; axis < 3; ++axis)
          box.cpu_h[3*axis+lattice] = finite_value(input);
      for (int atom = 0; atom < n; ++atom) {
        require(bool(input >> host_types[atom]), "Missing atom type");
        for (int axis = 0; axis < 3; ++axis)
          host_position[axis*n+atom] = finite_value(input);
      }
      box.get_inverse();
      box.set_is_orthogonal();
      types.copy_from_host(host_types.data());
      position.copy_from_host(host_position.data());
      // GPUMD potentials accumulate into these arrays; each frame starts empty.
      potential.fill(0.0);
      force.fill(0.0);
      virial.fill(0.0);
      model.compute(box, types, position, potential, force, virial);
      mobius_native::cuda_check(cudaDeviceSynchronize());
      potential.copy_to_host(host_potential.data());
      force.copy_to_host(host_force.data());
      virial.copy_to_host(host_virial.data());
      const auto& stats = model.neighbor_stats();
      output << "frame " << frame << ' ' << stats.candidate_builds << ' '
             << stats.candidate_reuses << ' ' << stats.small_box_calls << '\n';
      for (int atom = 0; atom < n; ++atom) {
        write_finite(output, host_potential[atom]);
        for (int axis = 0; axis < 3; ++axis) write_finite(output, host_force[axis*n+atom]);
        for (int component = 0; component < 9; ++component)
          write_finite(output, host_virial[component*n+atom]);
        output << '\n';
      }
      output.flush();
      require(bool(output), "Cannot write harness output");
    }
    std::string trailing;
    require(!(input >> trailing), "Trailing input data");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Neighbor harness: " << error.what() << '\n';
    return 1;
  }
}
