#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Runtime-sized, row-major native evaluator. No PyTorch runtime is required.
namespace mobius_native {
inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void launch_check() { cuda_check(cudaGetLastError()); }
inline int checked_work_size(size_t count) {
  // This correctness baseline uses 32-bit CUDA indexing. Leave room for
  // launch-grid rounding and reject oversized workloads before multiplication.
  if (count > size_t(std::numeric_limits<int>::max() - 255))
    throw std::runtime_error("Möbius workload exceeds the native 32-bit indexing limit");
  return static_cast<int>(count);
}

enum class Init { zero, uninitialized };

struct ScratchStats {
  size_t allocations = 0, reuses = 0, frees = 0;
  size_t live_bytes = 0, cached_bytes = 0, peak_live_bytes = 0;
  size_t zero_calls = 0, zero_bytes = 0;
};

// A model owns its scratch allocations; tensor aliases return an allocation
// only when their last shared owner dies. All users enqueue work on CUDA's
// default stream, so an idle allocation can be reused without synchronizing.
// This is deliberately not a process-global or multi-stream allocator.
class ScratchPool {
  struct Block {
    void* pointer;
    size_t bytes, epoch;
    bool leased;
  };
  std::vector<Block> blocks_;
  ScratchStats stats_;
  size_t epoch_ = 0;

  void free_idle(bool all) noexcept {
    for (size_t i = 0; i < blocks_.size();) {
      const Block& block = blocks_[i];
      if (!block.leased && (all || block.epoch != epoch_)) {
        cudaFree(block.pointer);
        stats_.cached_bytes -= block.bytes;
        ++stats_.frees;
        blocks_.erase(blocks_.begin() + i);
      } else ++i;
    }
  }

public:
  ScratchPool() = default;
  ScratchPool(const ScratchPool&) = delete;
  ScratchPool& operator=(const ScratchPool&) = delete;
  ~ScratchPool() {
    for (const Block& block : blocks_) cudaFree(block.pointer);
  }
  void begin() { ++epoch_; }
  // Retain only the working set used by the latest computation. Changed
  // neighbor counts therefore cannot leave an unbounded history of sizes.
  void finish() noexcept { free_idle(false); }
  const ScratchStats& stats() const { return stats_; }
  void record_zero(size_t bytes) { ++stats_.zero_calls; stats_.zero_bytes += bytes; }

  void* acquire(size_t bytes) {
    if (!bytes) return nullptr;
    size_t best = blocks_.size();
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const Block& block = blocks_[i];
      if (!block.leased && block.bytes >= bytes &&
          block.bytes - bytes <= bytes / 64 + 256 &&
          (best == blocks_.size() || block.bytes < blocks_[best].bytes)) best = i;
    }
    if (best != blocks_.size()) {
      Block& block = blocks_[best];
      block.leased = true; block.epoch = epoch_;
      stats_.cached_bytes -= block.bytes;
      stats_.live_bytes += block.bytes;
      if (stats_.live_bytes > stats_.peak_live_bytes) stats_.peak_live_bytes = stats_.live_bytes;
      ++stats_.reuses;
      return block.pointer;
    }
    void* pointer = nullptr;
    cudaError_t status = cudaMalloc(&pointer, bytes);
    if (status == cudaErrorMemoryAllocation) {
      // Cached buffers from other sizes must not cause an otherwise viable
      // workload to fail. Keep live aliases, release idle blocks, then retry.
      cudaGetLastError();
      free_idle(true);
      status = cudaMalloc(&pointer, bytes);
    }
    cuda_check(status);
    try { blocks_.push_back({pointer, bytes, epoch_, true}); }
    catch (...) { cudaFree(pointer); throw; }
    ++stats_.allocations;
    stats_.live_bytes += bytes;
    if (stats_.live_bytes > stats_.peak_live_bytes) stats_.peak_live_bytes = stats_.live_bytes;
    return pointer;
  }
  void release(void* pointer) noexcept {
    if (!pointer) return;
    for (Block& block : blocks_) if (block.pointer == pointer) {
      block.leased = false;
      stats_.live_bytes -= block.bytes;
      stats_.cached_bytes += block.bytes;
      return;
    }
  }
};

inline std::shared_ptr<ScratchPool>& active_scratch_pool() {
  static thread_local std::shared_ptr<ScratchPool> pool;
  return pool;
}

class ScratchScope {
  std::shared_ptr<ScratchPool> pool_, previous_;
public:
  explicit ScratchScope(std::shared_ptr<ScratchPool> pool)
      : pool_(std::move(pool)), previous_(active_scratch_pool()) {
    if (pool_) pool_->begin();
    active_scratch_pool() = pool_;
  }
  ScratchScope(const ScratchScope&) = delete;
  ScratchScope& operator=(const ScratchScope&) = delete;
  ~ScratchScope() {
    if (pool_) pool_->finish();
    active_scratch_pool() = previous_;
  }
};

// Thrust's scan also needs transient storage. Use the same lifetime rules as
// tensors instead of letting its default allocator call cudaMalloc every step.
class ScratchAllocator {
  std::shared_ptr<ScratchPool> pool_;
public:
  using value_type = char;
  explicit ScratchAllocator(std::shared_ptr<ScratchPool> pool) : pool_(std::move(pool)) {}
  char* allocate(std::ptrdiff_t bytes) {
    if (bytes < 0) throw std::runtime_error("Negative scratch allocation");
    return static_cast<char*>(pool_->acquire(static_cast<size_t>(bytes)));
  }
  void deallocate(char* pointer, size_t) { pool_->release(pointer); }
};

template <class T> class Buffer {
  std::shared_ptr<T> storage_;
  size_t count_ = 0;
public:
  Buffer() = default;
  explicit Buffer(size_t n, Init init = Init::zero) { resize(n, init); }
  void resize(size_t n, Init init = Init::zero) {
    if (n == count_) return;
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::runtime_error("Buffer allocation size overflow");
    storage_.reset(); count_ = 0;
    if (n) {
      const auto pool = active_scratch_pool();
      if (pool) {
        T* ptr = static_cast<T*>(pool->acquire(n * sizeof(T)));
        storage_ = std::shared_ptr<T>(ptr, [pool](T* p) { pool->release(p); });
      } else {
        T* ptr = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        storage_ = std::shared_ptr<T>(ptr, [](T* p) { cudaFree(p); });
      }
      if (init == Init::zero) {
        cuda_check(cudaMemset(storage_.get(), 0, n * sizeof(T)));
        if (pool) pool->record_zero(n * sizeof(T));
      }
    }
    count_ = n;
  }
  T* data() const { return storage_.get(); }
  size_t size() const { return count_; }
  void assign(const std::vector<T>& host) {
    resize(host.size(), Init::uninitialized);
    if (count_) cuda_check(cudaMemcpy(data(), host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
};
struct Tensor {
  int rows = 0, cols = 0;
  Buffer<double> values;
  Tensor() = default;
  Tensor(int r, int c, Init init = Init::zero) : rows(r), cols(c) {
    if (r < 0 || c < 0) throw std::runtime_error("Negative tensor dimension");
    values.resize(checked_work_size(static_cast<size_t>(r) * c), init);
  }
  double* data() const { return values.data(); }
  size_t size() const { return values.size(); }
};
struct Model {
  std::vector<std::string> elements;
  std::map<std::string, int> ints;
  std::map<std::string, double> reals;
  std::map<std::string, Tensor> tensors;
  int i(const std::string& key) const { return ints.at(key); }
  double r(const std::string& key) const { return reals.at(key); }
  const Tensor& t(const std::string& key) const { return tensors.at(key); }
};
struct Edges {
  int atoms = 0, count = 0;
  // Edges are contiguous by center; offsets has atoms+1 entries.
  Buffer<int> center, neighbor, types, offsets;
  Tensor vectors; // [directed edges, 3], includes the actual periodic image shift
};

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias = Tensor());
Tensor linear_backward(const Tensor& dy, const Tensor& weight);
Tensor activate(const Tensor& x, int kind);
Tensor activate_backward(const Tensor& x, const Tensor& dy, int kind);
Tensor norm(const Tensor& x, const Tensor& weight, double eps);
Tensor norm_backward(const Tensor& x, const Tensor& dy, const Tensor& weight, double eps);
Tensor add(const Tensor& a, const Tensor& b);
void add_inplace(Tensor& target, const Tensor& source);
Tensor concat(const std::vector<Tensor>& parts);
Tensor slice(const Tensor& x, int start, int width);
Tensor gather(const Tensor& x, const Buffer<int>& indices, int count);
Tensor scatter_sum(const Tensor& x, const Buffer<int>& indices, int rows);

struct LocalCache;
struct ReadoutCache;
Tensor local_forward(const Model&, const Edges&, const Tensor& q, std::shared_ptr<LocalCache>&);
Tensor local_backward(const Model&, const LocalCache&, const Tensor& dh);
Tensor readout_forward(const Model&, const Edges&, const Tensor& h, std::shared_ptr<ReadoutCache>&);
Tensor readout_backward(const Model&, const Edges&, const ReadoutCache&);

struct DescriptorCache;
struct DescriptorState {
  Tensor q, moments, harmonics;
  std::shared_ptr<DescriptorCache> cache;
};
// Derive coordinate-gradient polynomial tables once, without changing weights.
void prepare_descriptor_model(Model&);
DescriptorState descriptor_forward(const Model&, const Edges&);
void descriptor_backward(const Model&, const Edges&, const DescriptorState&,
                         const Tensor& dq, const Tensor& dmom, const Tensor& dY,
                         Tensor& edge_gradient);

struct EdgeGeometry {
  Tensor features, cutoff, unit, basis_derivative, cutoff_derivative;
};
EdgeGeometry edge_geometry(const Model&, const std::string& prefix, const Edges&);
void edge_geometry_backward(const Edges&, const EdgeGeometry&, const Tensor& d_features,
                            const Tensor& d_cutoff, const Tensor& d_unit, Tensor& d_vectors);
Tensor scalar_aggregate(const Edges&, const Tensor& gate_logits, const Tensor& cutoff,
                        const Tensor& atom_values);
void scalar_aggregate_backward(const Edges&, const Tensor& logits, const Tensor& cutoff,
                              const Tensor& values, const Tensor& d_sum, Tensor& d_logits,
                              Tensor& d_cutoff, Tensor& d_values);
struct MessageCache;
Tensor message_forward(const Model&, const std::string& prefix, const Edges&,
                       const Tensor& hidden, std::shared_ptr<MessageCache>&);
Tensor message_backward(const Model&, const std::string& prefix, const Edges&,
                        const MessageCache&, const Tensor& dh, Tensor& edge_gradient);
struct EquivariantCache;
Tensor direction_forward(const Model&, const Tensor& moments);
Tensor direction_backward(const Model&, const Tensor& d_direction);
void equivariant_forward(const Model&, const std::string& prefix, const Edges&,
                         const Tensor& h, const Tensor& direction, const Tensor& harmonics,
                         Tensor& new_h, Tensor& new_direction, std::shared_ptr<EquivariantCache>&);
void equivariant_backward(const Model&, const std::string& prefix, const Edges&,
                          const EquivariantCache&, const Tensor& dh, const Tensor& d_direction,
                          Tensor& previous_dh, Tensor& previous_d_direction,
                          Tensor& d_harmonics, Tensor& edge_gradient);

// Force and total virial are energy derivatives. The atomic partition returned
// here is not a validated heat-current/HNEMD partition for multi-hop models.
void add_zbl(const Model&, const Edges&, Tensor& atomic_energy, Tensor& edge_gradient);
void accumulate_outputs(const Edges&, const Tensor& energy, const Tensor& edge_gradient,
                        double* gpumd_energy, double* gpumd_force, double* gpumd_virial);
} // namespace mobius_native
