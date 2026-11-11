// Descriptor values and their analytical coordinate derivatives.
// Sparse polynomials are exported from the Python descriptor, keeping the
// NEP-style raw moment metric separate from the normalized direction states.
#include "kernels.cuh"
#include <cmath>

namespace mobius_native {

struct DescriptorCache {
  Tensor basis, angular_radial, radial_derivative, angular_derivative;
};

namespace {
constexpr int threads = 128;
constexpr double pi = 3.1415926535897932384626433832795;

__device__ double power(double x, int exponent) {
  double value = 1.0;
  for (int k = 0; k < exponent; ++k) value *= x;
  return value;
}

__device__ void angular_component(
    int component, const double* terms, const double* offsets,
    const double* unit, double& value, double* derivative) {
  value = 0.0;
  derivative[0] = derivative[1] = derivative[2] = 0.0;
  for (int t = int(offsets[component]); t < int(offsets[component + 1]); ++t) {
    const double* term = terms + t * 5;
    int exponents[3] = {int(term[2]), int(term[3]), int(term[4])};
    double monomial = term[1];
    for (int axis = 0; axis < 3; ++axis) monomial *= power(unit[axis], exponents[axis]);
    value += monomial;
    for (int axis = 0; axis < 3; ++axis) {
      if (!exponents[axis]) continue;
      double gradient = term[1] * exponents[axis];
      for (int other = 0; other < 3; ++other)
        gradient *= power(unit[other], exponents[other] - int(other == axis));
      derivative[axis] += gradient;
    }
  }
}

__device__ void radial_channel(
    double distance, double cutoff, const double* coefficients, int count,
    double& value, double& derivative) {
  value = derivative = 0.0;
  if (distance >= cutoff) return;
  const double shift = distance / cutoff - 1.0;
  const double x = 2.0 * shift * shift - 1.0;
  const double dx = 4.0 * shift / cutoff;
  const double fc = 0.5 * cos(pi * distance / cutoff) + 0.5;
  const double dfc = -0.5 * pi / cutoff * sin(pi * distance / cutoff);
  double previous = 1.0, current = x;
  double dprevious = 0.0, dcurrent = dx;
  for (int k = 0; k < count; ++k) {
    double polynomial, dpolynomial;
    if (k == 0) { polynomial = 1.0; dpolynomial = 0.0; }
    else if (k == 1) { polynomial = current; dpolynomial = dcurrent; }
    else {
      polynomial = 2.0 * x * current - previous;
      dpolynomial = 2.0 * dx * current + 2.0 * x * dcurrent - dprevious;
      previous = current; current = polynomial;
      dprevious = dcurrent; dcurrent = dpolynomial;
    }
    value += coefficients[k] * 0.5 * (polynomial + 1.0) * fc;
    derivative += coefficients[k] * 0.5 * (dpolynomial * fc + (polynomial + 1.0) * dfc);
  }
}

__global__ void basis_kernel(
    int edge_count, int components, const double* vectors,
    const double* terms, const double* offsets, double* basis) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edge_count * components) return;
  int edge = index / components, component = index % components;
  const double* vector = vectors + 3 * edge;
  double inverse = 1.0 / fmax(sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]), 1.0e-12);
  double unit[3] = {vector[0] * inverse, vector[1] * inverse, vector[2] * inverse};
  double value, derivative[3];
  angular_component(component, terms, offsets, unit, value, derivative);
  basis[index] = value;
}

__global__ void radial_kernel(
    int edge_count, int channels, int basis_count, int type_count, int qdim,
    double cutoff, const int* center, const int* neighbor, const int* types,
    const double* vectors, const double* coefficients, double* q,
    double* radial, double* radial_derivative) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edge_count * channels) return;
  int edge = index / channels, channel = index % channels;
  const double* vector = vectors + 3 * edge;
  double distance = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  int pair = types[center[edge]] * type_count + types[neighbor[edge]];
  double value, derivative;
  radial_channel(distance, cutoff, coefficients + (pair * channels + channel) * basis_count,
                 basis_count, value, derivative);
  radial_derivative[index] = derivative;
  if (radial) radial[index] = value;
  if (q) atomicAdd(q + center[edge] * qdim + channel, value);
}

__global__ void moments_kernel(
    int atoms, int channels, int components, const int* edge_offsets,
    const double* radial, const double* basis, double* moments) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= atoms * channels * components) return;
  int atom = index / (channels * components);
  int channel = (index / components) % channels, component = index % components;
  // Edges are contiguous by center. Accumulate each raw moment in a register,
  // rather than issuing a contended global atomic for every neighbor. Adjacent
  // component threads read adjacent basis values and write adjacent moments.
  double value = 0.0;
  for (int edge = edge_offsets[atom]; edge < edge_offsets[atom + 1]; ++edge)
    value += radial[edge * channels + channel] * basis[edge * components + component];
  moments[index] = value;
}

__global__ void invariant_kernel(
    int atoms, int channels, int components, int outputs, int radial_channels,
    int qdim, const double* terms, const double* offsets,
    const double* moments, double* q) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= atoms * channels * outputs) return;
  int atom = index / (channels * outputs), output = (index / channels) % outputs;
  int channel = index % channels;
  const double* row = moments + (atom * channels + channel) * components;
  double value = 0.0;
  for (int t = int(offsets[output]); t < int(offsets[output + 1]); ++t) {
    const double* term = terms + 7 * t;
    double monomial = term[1];
    for (int k = 0; k < int(term[2]); ++k) monomial *= row[int(term[3 + k])];
    value += monomial;
  }
  q[atom * qdim + radial_channels + output * channels + channel] = value;
}

__global__ void harmonics_kernel(
    int edges, int components, int eq_components, const double* basis,
    const double* scale, double* harmonics) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edges * eq_components) return;
  int edge = index / eq_components, component = index % eq_components;
  harmonics[index] = basis[edge * components + component] * scale[component];
}

__global__ void moment_gradient_kernel(
    int atoms, int channels, int components, int radial_channels, int qdim,
    const double* terms, const double* offsets, const double* moments,
    const double* dq, const double* direct, double* gradient) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= atoms * channels * components) return;
  int atom = index / (channels * components);
  int channel = (index / components) % channels, component = index % components;
  const double* row = moments + (atom * channels + channel) * components;
  // The loader differentiates the same exported polynomials once and groups
  // their terms by destination component. Keep this component's full sum in
  // one register, including the independent equivariant moment contribution.
  double value = direct ? direct[index] : 0.0;
  for (int t = int(offsets[component]); t < int(offsets[component + 1]); ++t) {
    const double* term = terms + 6 * t;
    int output = int(term[0]), degree = int(term[2]);
    double product = term[1] * dq[atom * qdim + radial_channels + output * channels + channel];
    for (int k = 0; k < degree; ++k) product *= row[int(term[3 + k])];
    value += product;
  }
  gradient[index] = value;
}

__global__ void geometry_gradient_kernel(
    int edges, int radial_channels, int angular_channels, int components,
    int eq_components, int qdim, const int* center, const double* vectors,
    const double* radial_derivative, const double* angular_radial,
    const double* angular_derivative, const double* moment_gradient,
    const double* dq, const double* dY, const double* scale,
    const double* terms, const double* offsets, double* edge_gradient) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  int atom = center[edge];
  const double* vector = vectors + 3 * edge;
  double inverse = 1.0 / fmax(sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]), 1.0e-12);
  double unit[3] = {vector[0] * inverse, vector[1] * inverse, vector[2] * inverse};
  double radial_gradient = 0.0, unit_gradient[3] = {0.0, 0.0, 0.0};
  for (int channel = 0; channel < radial_channels; ++channel)
    radial_gradient += dq[atom * qdim + channel] * radial_derivative[edge * radial_channels + channel];
  for (int component = 0; component < components; ++component) {
    double value, derivative[3];
    angular_component(component, terms, offsets, unit, value, derivative);
    double angular_weight = 0.0, radial_weight = 0.0;
    for (int channel = 0; channel < angular_channels; ++channel) {
      double weight = moment_gradient[(atom * angular_channels + channel) * components + component];
      angular_weight += weight * angular_radial[edge * angular_channels + channel];
      radial_weight += weight * angular_derivative[edge * angular_channels + channel];
    }
    radial_gradient += radial_weight * value;
    if (dY && component < eq_components)
      angular_weight += dY[edge * eq_components + component] * scale[component];
    for (int axis = 0; axis < 3; ++axis) unit_gradient[axis] += angular_weight * derivative[axis];
  }
  double projection = unit[0] * unit_gradient[0] + unit[1] * unit_gradient[1] + unit[2] * unit_gradient[2];
  for (int axis = 0; axis < 3; ++axis)
    edge_gradient[3 * edge + axis] += radial_gradient * unit[axis] + inverse * (unit_gradient[axis] - projection * unit[axis]);
}

int blocks(int count) { return (count + threads - 1) / threads; }
}  // namespace

void prepare_descriptor_model(Model& model) {
  // Differentiate the exported invariant polynomials once at model loading.
  // The original tables and all learned weights stay unchanged. Grouping by
  // destination component preserves each component's original term order.
  const Tensor& original = model.t("desc.invariant_terms");
  int order = model.i("l_max"), outputs = model.i("num_L");
  if (original.cols != 7 || order < 1 || order > 8 || outputs < 1)
    throw std::runtime_error("Invalid descriptor invariant table dimensions");
  int components = order * (order + 2);
  std::vector<double> host(original.size());
  if (!host.empty())
    cuda_check(cudaMemcpy(host.data(), original.data(), host.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
  auto index = [](double value, int lower, int upper) {
    if (!std::isfinite(value) || value < lower || value >= upper || value != std::floor(value))
      throw std::runtime_error("Invalid descriptor invariant polynomial index");
    return static_cast<int>(value);
  };
  std::vector<std::vector<double>> grouped(components);
  for (int t = 0; t < original.rows; ++t) {
    const double* term = host.data() + 7 * t;
    int output = index(term[0], 0, outputs), degree = index(term[2], 1, 5);
    if (!std::isfinite(term[1]))
      throw std::runtime_error("Invalid descriptor invariant polynomial coefficient");
    int factors[4] = {};
    for (int k = 0; k < degree; ++k) factors[k] = index(term[3 + k], 0, components);
    for (int k = 0; k < degree; ++k) {
      // Repeated factors intentionally remain separate derivative terms. This
      // retains the original arithmetic order instead of combining powers.
      double derivative[6] = {double(output), term[1], double(degree - 1), 0.0, 0.0, 0.0};
      int next = 3;
      for (int other = 0; other < degree; ++other)
        if (other != k) derivative[next++] = double(factors[other]);
      auto& destination = grouped[factors[k]];
      destination.insert(destination.end(), derivative, derivative + 6);
    }
  }
  std::vector<double> terms, offsets(components + 1, 0.0);
  for (int component = 0; component < components; ++component) {
    terms.insert(terms.end(), grouped[component].begin(), grouped[component].end());
    offsets[component + 1] = double(terms.size() / 6);
  }
  int term_count = checked_work_size(terms.size()) / 6;
  Tensor derivative_terms(term_count, 6, Init::uninitialized);
  Tensor derivative_offsets(1, components + 1, Init::uninitialized);
  derivative_terms.values.assign(terms);
  derivative_offsets.values.assign(offsets);
  model.tensors["desc.invariant_derivative_terms"] = std::move(derivative_terms);
  model.tensors["desc.invariant_derivative_offsets"] = std::move(derivative_offsets);
}

DescriptorState descriptor_forward(const Model& model, const Edges& edges) {
  int nr = model.i("n_desc_radial"), na = model.i("n_desc_angular");
  int order = model.i("l_max"), components = order * (order + 2);
  int eq_order = model.i("equivariant_l_max");
  int eq_components = model.i("message_mode") ? eq_order * (eq_order + 2) : 0;
  int outputs = model.i("num_L"), qdim = model.i("input_dim");
  checked_work_size(size_t(edges.atoms) * na * components);
  checked_work_size(size_t(edges.atoms) * na * outputs);
  DescriptorState state;
  // Radial descriptors are atomic sums and must start at zero. Raw moments
  // are written in full by the segmented kernel, including empty centers.
  state.q = Tensor(edges.atoms, qdim);
  state.moments = Tensor(edges.atoms, na * components, Init::uninitialized);
  state.harmonics = Tensor(edges.count, eq_components, Init::uninitialized);
  state.cache = std::make_shared<DescriptorCache>();
  auto& cache = *state.cache;
  cache.basis = Tensor(edges.count, components, Init::uninitialized);
  cache.angular_radial = Tensor(edges.count, na, Init::uninitialized);
  cache.radial_derivative = Tensor(edges.count, nr, Init::uninitialized);
  cache.angular_derivative = Tensor(edges.count, na, Init::uninitialized);
  if (edges.count) {
    basis_kernel<<<blocks(edges.count * components), threads>>>(
        edges.count, components, edges.vectors.data(), model.t("desc.basis_terms").data(),
        model.t("desc.basis_offsets").data(), cache.basis.data());
    radial_kernel<<<blocks(edges.count * nr), threads>>>(
        edges.count, nr, model.i("k_max_radial"), model.i("n_types"), qdim,
        model.r("rcut_radial"), edges.center.data(), edges.neighbor.data(), edges.types.data(),
        edges.vectors.data(), model.t("descriptor.radial.c_table").data(), state.q.data(),
        nullptr, cache.radial_derivative.data());
    radial_kernel<<<blocks(edges.count * na), threads>>>(
        edges.count, na, model.i("k_max_angular"), model.i("n_types"), qdim,
        model.r("rcut_angular"), edges.center.data(), edges.neighbor.data(), edges.types.data(),
        edges.vectors.data(), model.t("descriptor.angular.c_table").data(), nullptr,
        cache.angular_radial.data(), cache.angular_derivative.data());
    if (eq_components)
      harmonics_kernel<<<blocks(edges.count * eq_components), threads>>>(
          edges.count, components, eq_components, cache.basis.data(),
          model.t("desc.orthonormal_scale").data(), state.harmonics.data());
    launch_check();
  }
  if (edges.atoms) {
    moments_kernel<<<blocks(edges.atoms * na * components), threads>>>(
        edges.atoms, na, components, edges.offsets.data(), cache.angular_radial.data(),
        cache.basis.data(), state.moments.data());
    invariant_kernel<<<blocks(edges.atoms * na * outputs), threads>>>(
        edges.atoms, na, components, outputs, nr, qdim,
        model.t("desc.invariant_terms").data(), model.t("desc.invariant_offsets").data(),
        state.moments.data(), state.q.data());
    launch_check();
  }
  return state;
}

void descriptor_backward(
    const Model& model, const Edges& edges, const DescriptorState& state,
    const Tensor& dq, const Tensor& dmom, const Tensor& dY, Tensor& edge_gradient) {
  if (!edges.count) return;
  int nr = model.i("n_desc_radial"), na = model.i("n_desc_angular");
  int order = model.i("l_max"), components = order * (order + 2);
  int eq_order = model.i("equivariant_l_max");
  int eq_components = model.i("message_mode") ? eq_order * (eq_order + 2) : 0;
  int qdim = model.i("input_dim");
  const auto& cache = *state.cache;
  Tensor gradient(edges.atoms, na * components, Init::uninitialized);
  moment_gradient_kernel<<<blocks(edges.atoms * na * components), threads>>>(
      edges.atoms, na, components, nr, qdim,
      model.t("desc.invariant_derivative_terms").data(),
      model.t("desc.invariant_derivative_offsets").data(), state.moments.data(), dq.data(),
      dmom.size() ? dmom.data() : nullptr, gradient.data());
  geometry_gradient_kernel<<<blocks(edges.count), threads>>>(
      edges.count, nr, na, components, eq_components, qdim, edges.center.data(),
      edges.vectors.data(), cache.radial_derivative.data(), cache.angular_radial.data(),
      cache.angular_derivative.data(), gradient.data(), dq.data(), dY.size() ? dY.data() : nullptr,
      model.t("desc.orthonormal_scale").data(), model.t("desc.basis_terms").data(),
      model.t("desc.basis_offsets").data(), edge_gradient.data());
  launch_check();
}

}  // namespace mobius_native
