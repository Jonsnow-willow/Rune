#include "mobius.cuh"
#include "force/neighbor.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace {
using namespace mobius_native;

void require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error("Möbius: " + reason);
}

void expect(std::istream& in, const char* expected) {
  std::string actual;
  require(bool(in >> actual) && actual == expected,
          std::string("expected '") + expected + "' in model file");
}

Model read_model(const char* path) {
  std::ifstream in(path);
  require(bool(in), std::string("cannot open ") + path);
  expect(in, "mobius1");
  Model model;
  int count = 0, schema = 0;
  require(bool(in >> count) && count > 0 && count <= 118, "invalid element count");
  model.elements.resize(count);
  for (auto& element : model.elements) require(bool(in >> element), "missing element");
  require(std::set<std::string>(model.elements.begin(), model.elements.end()).size() ==
              model.elements.size(), "duplicate elements");
  expect(in, "schema");
  require(bool(in >> schema) && schema == 1, "unsupported native schema");
  expect(in, "ints");
  require(bool(in >> count) && count > 0 && count < 1000, "invalid integer metadata count");
  for (int n = 0; n < count; ++n) {
    std::string key; int value;
    require(bool(in >> key >> value), "invalid integer metadata");
    require(model.ints.emplace(key, value).second, "duplicate integer metadata");
  }
  expect(in, "reals");
  require(bool(in >> count) && count > 0 && count < 1000, "invalid real metadata count");
  for (int n = 0; n < count; ++n) {
    std::string key; double value;
    require(bool(in >> key >> value) && std::isfinite(value), "invalid real metadata");
    require(model.reals.emplace(key, value).second, "duplicate real metadata");
  }
  expect(in, "tensors");
  require(bool(in >> count) && count > 0 && count < 100000, "invalid tensor count");
  for (int n = 0; n < count; ++n) {
    std::string key; int rank;
    require(bool(in >> key >> rank) && rank >= 0 && rank <= 8, "invalid tensor rank");
    int rows = 1, cols = 1;
    size_t size = 1;
    for (int axis = 0; axis < rank; ++axis) {
      int dimension;
      require(bool(in >> dimension) && dimension >= 0, "invalid tensor dimension");
      require(dimension == 0 || size <= size_t(std::numeric_limits<int>::max()) / dimension,
              "tensor is too large");
      size *= dimension;
      if (axis == rank-1) cols = dimension; else rows *= dimension;
    }
    std::vector<double> values(size);
    for (double& value : values)
      require(bool(in >> value) && std::isfinite(value), "missing or nonfinite tensor values");
    Tensor tensor(rows, cols, Init::uninitialized);
    tensor.values.assign(values);
    require(model.tensors.emplace(key, tensor).second, "duplicate tensor name");
  }
  expect(in, "END");
  std::string trailing;
  require(!(in >> trailing), "unexpected data after END");
  require(model.i("n_types") == int(model.elements.size()), "element count mismatch");
  for (const char* key : {"n_desc_radial", "n_desc_angular", "k_max_radial", "k_max_angular",
                          "num_L", "input_dim", "hidden_dim", "ffn_dim", "message_dim",
                          "pair_dim", "message_groups", "message_basis_size"})
    require(model.i(key) > 0, std::string("invalid ") + key);
  require(model.i("message_layers") >= 0 && model.i("blocks") >= 0, "negative layer count");
  require(model.i("activation") >= 0 && model.i("activation") <= 2, "unsupported activation");
  require(model.i("message_mode") == 0 || model.i("message_mode") == 1, "unsupported message mode");
  require(model.i("l_max") >= 1 && model.i("l_max") <= 8, "unsupported angular order");
  require(!model.i("message_layers") || model.i("hidden_dim") % model.i("message_groups") == 0,
          "invalid message groups");
  require(model.r("rcut_radial") > 0 && model.r("rcut_angular") > 0 &&
              model.r("message_cutoff") > 0 && model.r("norm_eps") > 0,
          "cutoffs and normalization epsilon must be positive");
  require(model.r("message_cutoff") <= model.r("rcut_radial"), "message cutoff exceeds radial cutoff");
  require(model.i("max_neighbors_radial") > 0 && model.i("max_neighbors_angular") > 0 &&
              model.i("vector_channels") >= 0, "invalid neighbor/channel capacity");
  const int nt=model.i("n_types"), h=model.i("hidden_dim"), f=model.i("ffn_dim");
  const int nr=model.i("n_desc_radial"), na=model.i("n_desc_angular"), q=model.i("input_dim");
  const int l=model.i("l_max"), components=l*(l+2), ec=model.i("equivariant_channels");
  const int el=model.i("equivariant_l_max"), md=model.i("message_dim"), vc=model.i("vector_channels");
  const bool eq=model.i("message_mode") == 1;
  require(q == nr+na*model.i("num_L"), "descriptor dimension mismatch");
  require(!eq || (ec>0 && el>=1 && el<=l && model.i("message_layers")>0),
          "invalid equivariant dimensions");
  auto shape = [&](const std::string& name, int rows, int cols) {
    const auto found=model.tensors.find(name);
    require(found!=model.tensors.end() && found->second.rows==rows && found->second.cols==cols,
            "missing or incorrectly shaped tensor '"+name+"'");
  };
  auto affine = [&](const std::string& name, int out, int in, bool bias=true) {
    shape(name+".weight",out,in);
    if (bias) shape(name+".bias",1,out);
  };
  shape("descriptor.radial.c_table",nt*nt*nr,model.i("k_max_radial"));
  shape("descriptor.angular.c_table",nt*nt*na,model.i("k_max_angular"));
  shape("q_scaler",1,q); shape("element_embedding.weight",nt,h);
  affine("input_projection",h,q); shape("output_norm.weight",1,h);
  shape("energy_heads.weight",nt,h); shape("energy_heads.bias",1,nt);
  shape("desc.basis_offsets",1,components+1);
  shape("desc.invariant_offsets",1,model.i("num_L")+1);
  shape("desc.orthonormal_scale",1,components);
  require(model.t("desc.basis_terms").cols==5 && model.t("desc.invariant_terms").cols==7,
          "invalid descriptor polynomial tables");
  for (int b=0; b<model.i("blocks"); ++b) {
    const std::string p="residual_blocks."+std::to_string(b);
    shape(p+".norm.weight",1,h); affine(p+".ffn.0",f,h); affine(p+".ffn.2",h,f);
  }
  if (eq) for (int order=0; order<el; ++order)
    affine("direction_encoder.projections."+std::to_string(order),ec,na,false);
  for (int b=0; b<model.i("message_layers"); ++b) {
    const std::string p="message_blocks."+std::to_string(b);
    shape(p+".input_norm.weight",1,h); shape(p+".message_norm.weight",1,h);
    shape(p+".pair_embedding.weight",nt*nt,model.i("pair_dim"));
    affine(p+".edge_mlp.0",md,model.i("message_basis_size")+model.i("pair_dim")+(eq?2*md+2*ec*el:0));
    affine(p+".group_gate",model.i("message_groups"),md);
    affine(p+".value_projection",h,h,false);
    affine(p+".update.0",md,h+(eq?ec*el:vc)); affine(p+".update.2",h,md);
    if (eq) {
      affine(p+".center_context",md,h,false); affine(p+".neighbor_context",md,h,false);
      affine(p+".direction_gates",2*ec*el,md);
      for (int order=0; order<el; ++order)
        affine(p+".direction_mix."+std::to_string(order),ec,ec,false);
    } else if (vc) {
      affine(p+".vector_gate",vc,md); affine(p+".vector_source",vc,h,false);
      affine(p+".vector_mix",vc,vc,false);
    }
  }
  if (model.i("zbl_enabled")) {
    shape("zbl.atomic_numbers",1,nt); shape("zbl.universal_coefficients",4,2);
    shape("zbl.pair_parameters",model.i("zbl_flexible")?nt*(nt+1)/2:0,10);
  }
  return model;
}

__device__ void record_edge(int i, int j, double x, double y, double z, double rc2,
                           bool writing, const int* offsets, int& count,
                           int* centers, int* neighbors, double* vectors, int* invalid) {
  const double r2 = x*x + y*y + z*z;
  if (r2 >= rc2) return;
  if (r2 < 1.e-24) { atomicExch(invalid, 1); return; }
  if (writing) {
    const int edge = offsets[i] + count;
    if (centers) centers[edge] = i;
    neighbors[edge] = j;
    if (vectors) {
      vectors[3*edge] = x; vectors[3*edge+1] = y; vectors[3*edge+2] = z;
    }
  }
  ++count;
}

// Enumerate every periodic image in small boxes. The bounds follow
// |fractional_component| <= cutoff / perpendicular_box_thickness.
__global__ void small_edges(Box box, int n, const double* position, double cutoff,
                            int ax, int ay, int az, bool writing, int* counts,
                            const int* offsets, int* centers, int* neighbors,
                            double* vectors, int* invalid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int count = 0;
  for (int j = 0; j < n; ++j) {
    const double x = position[j] - position[i];
    const double y = position[n+j] - position[n+i];
    const double z = position[2*n+j] - position[2*n+i];
    double sx = box.cpu_h[9]*x + box.cpu_h[10]*y + box.cpu_h[11]*z;
    double sy = box.cpu_h[12]*x + box.cpu_h[13]*y + box.cpu_h[14]*z;
    double sz = box.cpu_h[15]*x + box.cpu_h[16]*y + box.cpu_h[17]*z;
    if (box.pbc_x) sx -= nearbyint(sx);
    if (box.pbc_y) sy -= nearbyint(sy);
    if (box.pbc_z) sz -= nearbyint(sz);
    for (int a = -ax; a <= ax; ++a)
      for (int b = -ay; b <= ay; ++b)
        for (int c = -az; c <= az; ++c) {
          if (i == j && a == 0 && b == 0 && c == 0) continue;
          const double u = sx+a, v = sy+b, w = sz+c;
          record_edge(i, j, box.cpu_h[0]*u+box.cpu_h[1]*v+box.cpu_h[2]*w,
                      box.cpu_h[3]*u+box.cpu_h[4]*v+box.cpu_h[5]*w,
                      box.cpu_h[6]*u+box.cpu_h[7]*v+box.cpu_h[8]*w,
                      cutoff*cutoff, writing, offsets, count, centers, neighbors, vectors, invalid);
        }
  }
  if (!writing) counts[i] = count;
}

// Use GPUMD's GPU cell list for production-size boxes. Cell width is cutoff/2;
// the five-cell stencil is unique when every periodic axis has >= 5 cells.
__global__ void large_edges(Box box, int n, const double* position, double cutoff,
                            int nx, int ny, int nz, const int* cell_counts,
                            const int* cell_sum, const int* contents, bool writing,
                            int* counts, const int* offsets, int* centers,
                            int* neighbors, double* vectors, int* invalid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int cx, cy, cz, cell;
  const double x = position[i], y = position[n+i], z = position[2*n+i];
  find_cell_id(box, x, y, z, 2.0/cutoff, nx, ny, nz, cx, cy, cz, cell);
  const int lx = box.pbc_x ? 2 : 0, ly = box.pbc_y ? 2 : 0, lz = box.pbc_z ? 2 : 0;
  int count = 0;
  for (int a = -lx; a <= lx; ++a)
    for (int b = -ly; b <= ly; ++b)
      for (int c = -lz; c <= lz; ++c) {
        const int bx = (cx+a+nx)%nx, by = (cy+b+ny)%ny, bz = (cz+c+nz)%nz;
        const int next = bx + nx*(by+ny*bz);
        for (int k = 0; k < cell_counts[next]; ++k) {
          const int j = contents[cell_sum[next]+k];
          if (i == j) continue;
          double dx = position[j]-x, dy = position[n+j]-y, dz = position[2*n+j]-z;
          apply_mic(box, dx, dy, dz);
          record_edge(i, j, dx, dy, dz, cutoff*cutoff, writing, offsets, count,
                      centers, neighbors, vectors, invalid);
        }
      }
  if (!writing) counts[i] = count;
}

__global__ void validate_types(int n, int n_types, const int* types, int* invalid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (types[i] < 0 || types[i] >= n_types)) atomicExch(invalid, 2);
}
// Reuse a candidate list only while every atom has moved less than skin/2.
// Wrapped endpoints plus the minimum-image displacement also handle inputs
// translated by arbitrarily many periodic box vectors between calls.
__global__ void check_displacement(Box box, int n, const double* position,
                                   const double* reference, double limit2, int* rebuild) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double x = position[i] - reference[i];
  double y = position[n+i] - reference[n+i];
  double z = position[2*n+i] - reference[2*n+i];
  apply_mic(box, x, y, z);
  if (x*x + y*y + z*z >= limit2) atomicExch(rebuild, 1);
}

// Candidate membership is cached; active cutoff membership and all vectors
// are evaluated from the current positions on every force evaluation.
__global__ void active_edges(Box box, int n, const double* position, double cutoff,
                             const int* candidate_offsets, const int* candidate_neighbors,
                             bool writing, int* counts, const int* offsets,
                             int* centers, int* neighbors, double* vectors, int* invalid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int count = 0;
  for (int e = candidate_offsets[i]; e < candidate_offsets[i+1]; ++e) {
    const int j = candidate_neighbors[e];
    double x = position[j] - position[i];
    double y = position[n+j] - position[n+i];
    double z = position[2*n+j] - position[2*n+i];
    apply_mic(box, x, y, z);
    record_edge(i, j, x, y, z, cutoff*cutoff, writing, offsets, count,
                centers, neighbors, vectors, invalid);
  }
  if (!writing) counts[i] = count;
}
__global__ void wrap_positions(Box box, int n, const double* positions, double* wrapped) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  const double x=positions[i],y=positions[n+i],z=positions[2*n+i];
  double sx=box.cpu_h[9]*x+box.cpu_h[10]*y+box.cpu_h[11]*z;
  double sy=box.cpu_h[12]*x+box.cpu_h[13]*y+box.cpu_h[14]*z;
  double sz=box.cpu_h[15]*x+box.cpu_h[16]*y+box.cpu_h[17]*z;
  if (box.pbc_x) sx-=floor(sx);
  if (box.pbc_y) sy-=floor(sy);
  if (box.pbc_z) sz-=floor(sz);
  wrapped[i]=box.cpu_h[0]*sx+box.cpu_h[1]*sy+box.cpu_h[2]*sz;
  wrapped[n+i]=box.cpu_h[3]*sx+box.cpu_h[4]*sy+box.cpu_h[5]*sz;
  wrapped[2*n+i]=box.cpu_h[6]*sx+box.cpu_h[7]*sy+box.cpu_h[8]*sz;
}
__global__ void validate_neighbor_counts(int n, const int* offsets, const double* vectors,
                                        double radial2, double angular2, int radial_max,
                                        int angular_max, int* invalid) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  int radial=0, angular=0;
  for (int e=offsets[i]; e<offsets[i+1]; ++e) {
    const double x=vectors[3*e],y=vectors[3*e+1],z=vectors[3*e+2];
    const double r2=x*x+y*y+z*z;
    radial += r2<radial2; angular += r2<angular2;
  }
  if (radial>radial_max) atomicExch(invalid,3);
  if (angular>angular_max) atomicExch(invalid,4);
}
} // namespace

Mobius::Mobius(const char* model_path, int number_of_atoms) : model_(read_model(model_path)) {
  require(number_of_atoms > 0, "empty atom collection");
  mobius_native::prepare_descriptor_model(model_);
  rc = std::max(model_.r("rcut_radial"), model_.r("rcut_angular"));
  N1 = 0; N2 = number_of_atoms;
  // Persistent neighbor bookkeeping is separate from the scratch workspace.
  counts_.resize(size_t(number_of_atoms)+1, mobius_native::Init::uninitialized);
  offsets_.resize(size_t(number_of_atoms)+1, mobius_native::Init::uninitialized);
  invalid_.resize(1, mobius_native::Init::uninitialized);
  rebuild_.resize(1, mobius_native::Init::uninitialized);
  printf("Möbius native CUDA: %zu elements, %d message layers, mode %s (double precision).\n",
         model_.elements.size(), model_.i("message_layers"),
         model_.i("message_mode") ? "equivariant" : "scalar/local");
}

mobius_native::Edges Mobius::make_edges(Box& box, const GPU_Vector<int>& types,
                                       const GPU_Vector<double>& position) {
  using namespace mobius_native;
  const int n = int(types.size());
  require(n > 0, "empty atom collection");
  require(position.size() == size_t(n)*3, "invalid GPUMD position shape");
  require(N1 == 0 && N2 == n, "partial-system potentials are not supported");
  Edges edges; edges.atoms = n; edges.types.resize(n, Init::uninitialized);
  cuda_check(cudaMemcpy(edges.types.data(), types.data(), sizeof(int)*n, cudaMemcpyDeviceToDevice));
  cuda_check(cudaMemset(invalid_.data(), 0, sizeof(int)));
  cuda_check(cudaMemset(counts_.data()+n, 0, sizeof(int)));
  const int grid = (n+127)/128;
  validate_types<<<grid,128>>>(n, model_.i("n_types"), types.data(), invalid_.data());
  box.get_inverse(); box.set_is_orthogonal();
  require(std::isfinite(box.get_volume()) && box.get_volume()>0, "invalid simulation box");
  if (wrapped_position_.size()!=size_t(n)*3) wrapped_position_.resize(size_t(n)*3);
  wrap_positions<<<grid,128>>>(box,n,position.data(),wrapped_position_.data());
  // An internal Verlet skin changes only the search, never the model cutoff.
  constexpr double skin = 1.0;
  const double candidate_cutoff = rc + skin;
  int bins[3]; box.get_num_bins(candidate_cutoff*.5, bins);
  const bool small = (box.pbc_x && bins[0] < 5) || (box.pbc_y && bins[1] < 5) ||
                     (box.pbc_z && bins[2] < 5) || (!box.pbc_x && !box.pbc_y && !box.pbc_z);
  int images[3] = {0,0,0};
  ScratchAllocator allocator(scratch_);
  if (small) {
    // Small cells may contain several images of one atom inside the cutoff.
    // Keep their exact enumeration instead of caching a minimum-image list.
    candidate_valid_ = false;
    ++neighbor_stats_.small_box_calls;
    const double thickness[] = {box.thickness_x, box.thickness_y, box.thickness_z};
    const int periodic[] = {box.pbc_x, box.pbc_y, box.pbc_z};
    for (int d=0; d<3; ++d) {
      require(std::isfinite(thickness[d]) && thickness[d] > 0, "invalid simulation box");
      images[d] = periodic[d] ? int(std::ceil(rc/thickness[d]+.5)) : 0;
    }
  } else {
    bool rebuild = !candidate_valid_;
    for (int d=0; d<9; ++d) rebuild = rebuild || reference_box_[d] != box.cpu_h[d];
    const int periodic[] = {box.pbc_x, box.pbc_y, box.pbc_z};
    for (int d=0; d<3; ++d) rebuild = rebuild || reference_pbc_[d] != periodic[d];
    if (!rebuild) {
      cuda_check(cudaMemset(rebuild_.data(), 0, sizeof(int)));
      check_displacement<<<grid,128>>>(box,n,wrapped_position_.data(),reference_position_.data(),
                                      skin*skin*.25,rebuild_.data());
      launch_check();
      int changed = 0;
      cuda_check(cudaMemcpy(&changed,rebuild_.data(),sizeof(int),cudaMemcpyDeviceToHost));
      rebuild = changed != 0;
    }
    if (rebuild) {
      candidate_valid_ = false;
      if (candidate_counts_.size() != size_t(n)+1) {
        candidate_counts_.resize(size_t(n)+1);
        candidate_offsets_.resize(size_t(n)+1);
      }
      if (reference_position_.size() != size_t(n)*3) reference_position_.resize(size_t(n)*3);
      if (cell_contents_.size() != size_t(n)) cell_contents_.resize(n);
      find_cell_list(candidate_cutoff*.5,bins,box,wrapped_position_,cell_count_,cell_sum_,cell_contents_);
      cuda_check(cudaMemset(candidate_counts_.data()+n,0,sizeof(int)));
      large_edges<<<grid,128>>>(box,n,wrapped_position_.data(),candidate_cutoff,bins[0],bins[1],bins[2],
          cell_count_.data(),cell_sum_.data(),cell_contents_.data(),false,candidate_counts_.data(),
          candidate_offsets_.data(),nullptr,nullptr,nullptr,invalid_.data());
      launch_check();
      thrust::exclusive_scan(thrust::cuda::par(allocator),candidate_counts_.data(),
                             candidate_counts_.data()+n+1,candidate_offsets_.data());
      int candidate_count = 0, invalid = 0;
      cuda_check(cudaMemcpy(&candidate_count,candidate_offsets_.data()+n,sizeof(int),cudaMemcpyDeviceToHost));
      cuda_check(cudaMemcpy(&invalid,invalid_.data(),sizeof(int),cudaMemcpyDeviceToHost));
      require(!invalid,invalid == 2 ? "atom type outside model element order" : "overlapping atoms");
      require(candidate_count >= 0,"candidate neighbor edge count overflow");
      if (candidate_neighbors_.size() != size_t(candidate_count)) candidate_neighbors_.resize(candidate_count);
      if (candidate_count) {
        large_edges<<<grid,128>>>(box,n,wrapped_position_.data(),candidate_cutoff,bins[0],bins[1],bins[2],
            cell_count_.data(),cell_sum_.data(),cell_contents_.data(),true,candidate_counts_.data(),
            candidate_offsets_.data(),nullptr,candidate_neighbors_.data(),nullptr,invalid_.data());
        launch_check();
      }
      cuda_check(cudaMemcpy(reference_position_.data(),wrapped_position_.data(),sizeof(double)*size_t(n)*3,
                            cudaMemcpyDeviceToDevice));
      std::copy(box.cpu_h,box.cpu_h+9,reference_box_);
      std::copy(periodic,periodic+3,reference_pbc_);
      candidate_valid_ = true;
      ++neighbor_stats_.candidate_builds;
    } else ++neighbor_stats_.candidate_reuses;
  }
  auto build = [&](bool writing) {
    if (small)
      small_edges<<<grid,128>>>(box,n,wrapped_position_.data(),rc,images[0],images[1],images[2],writing,
          counts_.data(),offsets_.data(),edges.center.data(),edges.neighbor.data(),edges.vectors.data(),invalid_.data());
    else
      active_edges<<<grid,128>>>(box,n,wrapped_position_.data(),rc,candidate_offsets_.data(),
          candidate_neighbors_.size() ? candidate_neighbors_.data() : nullptr,
          writing,counts_.data(),offsets_.data(),
          edges.center.data(),edges.neighbor.data(),edges.vectors.data(),invalid_.data());
    launch_check();
  };
  build(false);
  thrust::exclusive_scan(thrust::cuda::par(allocator), counts_.data(), counts_.data()+n+1, offsets_.data());
  cuda_check(cudaMemcpy(&edges.count, offsets_.data()+n, sizeof(int), cudaMemcpyDeviceToHost));
  int invalid = 0;
  cuda_check(cudaMemcpy(&invalid, invalid_.data(), sizeof(int), cudaMemcpyDeviceToHost));
  require(!invalid, invalid == 2 ? "atom type outside model element order" : "overlapping atoms");
  require(edges.count >= 0, "neighbor edge count overflow");
  edges.center.resize(edges.count, Init::uninitialized);
  edges.neighbor.resize(edges.count, Init::uninitialized);
  edges.vectors = Tensor(edges.count,3, Init::uninitialized);
  if (edges.count) build(true);
  validate_neighbor_counts<<<grid,128>>>(n,offsets_.data(),edges.vectors.data(),
      model_.r("rcut_radial")*model_.r("rcut_radial"),
      model_.r("rcut_angular")*model_.r("rcut_angular"),
      model_.i("max_neighbors_radial"),model_.i("max_neighbors_angular"),invalid_.data());
  launch_check();
  cuda_check(cudaMemcpy(&invalid,invalid_.data(),sizeof(int),cudaMemcpyDeviceToHost));
  require(!invalid, invalid==3 ? "radial neighbor capacity exceeded" : "angular neighbor capacity exceeded");
  edges.offsets = offsets_;
  return edges;
}

void Mobius::compute(Box& box, const GPU_Vector<int>& types,
                     const GPU_Vector<double>& position, GPU_Vector<double>& potential,
                     GPU_Vector<double>& force, GPU_Vector<double>& virial) {
  using namespace mobius_native;
  // Declare before tensors so all scratch aliases return before this scope ends.
  ScratchScope workspace(scratch_);
  const Edges edges = make_edges(box, types, position);
  const DescriptorState state = descriptor_forward(model_, edges);
  std::shared_ptr<LocalCache> local;
  Tensor hidden = local_forward(model_, edges, state.q, local), direction;
  const int layers = model_.i("message_layers");
  std::vector<std::shared_ptr<MessageCache>> scalar(layers);
  std::vector<std::shared_ptr<EquivariantCache>> angular(layers);
  const bool equivariant = model_.i("message_mode") == 1;
  if (equivariant) direction = direction_forward(model_, state.moments);
  for (int layer=0; layer<layers; ++layer) {
    const std::string prefix = "message_blocks."+std::to_string(layer);
    if (equivariant) {
      Tensor next_h, next_direction;
      equivariant_forward(model_,prefix,edges,hidden,direction,state.harmonics,
                          next_h,next_direction,angular[layer]);
      hidden = next_h; direction = next_direction;
    } else hidden = message_forward(model_,prefix,edges,hidden,scalar[layer]);
  }
  std::shared_ptr<ReadoutCache> readout;
  Tensor energy = readout_forward(model_,edges,hidden,readout);
  Tensor dh = readout_backward(model_,edges,*readout), edge_gradient(edges.count,3);
  Tensor d_direction, d_harmonics, d_moments;
  if (equivariant) {
    d_direction = Tensor(direction.rows,direction.cols);
    d_harmonics = Tensor(state.harmonics.rows,state.harmonics.cols);
  }
  for (int layer=layers-1; layer>=0; --layer) {
    const std::string prefix = "message_blocks."+std::to_string(layer);
    if (equivariant) {
      Tensor previous_h, previous_direction;
      equivariant_backward(model_,prefix,edges,*angular[layer],dh,d_direction,
                           previous_h,previous_direction,d_harmonics,edge_gradient);
      dh = previous_h; d_direction = previous_direction;
    } else dh = message_backward(model_,prefix,edges,*scalar[layer],dh,edge_gradient);
  }
  if (equivariant) d_moments = direction_backward(model_,d_direction);
  Tensor dq = local_backward(model_,*local,dh);
  descriptor_backward(model_,edges,state,dq,d_moments,d_harmonics,edge_gradient);
  add_zbl(model_,edges,energy,edge_gradient);
  accumulate_outputs(edges,energy,edge_gradient,potential.data(),force.data(),virial.data());
  launch_check();
}
