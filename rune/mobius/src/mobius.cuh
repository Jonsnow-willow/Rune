#pragma once

#include "force/potential.cuh"
#include "kernels.cuh"

// GPUMD integration for the same exported Local, scalar-message and
// persistent-equivariant weights used by the Python reference.
class Mobius : public Potential {
public:
  struct NeighborStats {
    size_t candidate_builds = 0;
    size_t candidate_reuses = 0;
    size_t small_box_calls = 0;
  };
  Mobius(const char* model_path, int number_of_atoms);
  void compute(Box&, const GPU_Vector<int>&, const GPU_Vector<double>&,
               GPU_Vector<double>&, GPU_Vector<double>&, GPU_Vector<double>&) override;
  bool supports_heat_current() const { return model_.i("message_layers") == 0; }
  const mobius_native::ScratchStats& scratch_stats() const { return scratch_->stats(); }
  const NeighborStats& neighbor_stats() const { return neighbor_stats_; }

private:
  mobius_native::Model model_;
  std::shared_ptr<mobius_native::ScratchPool> scratch_ =
      std::make_shared<mobius_native::ScratchPool>();
  GPU_Vector<int> cell_count_, cell_sum_, cell_contents_;
  GPU_Vector<double> wrapped_position_;
  // Candidate neighbors outlive each compute call and never enter the scratch pool.
  GPU_Vector<int> candidate_counts_, candidate_offsets_, candidate_neighbors_;
  GPU_Vector<double> reference_position_;
  mobius_native::Buffer<int> rebuild_;
  bool candidate_valid_ = false;
  double reference_box_[9] = {};
  int reference_pbc_[3] = {};
  NeighborStats neighbor_stats_;
  mobius_native::Buffer<int> counts_, offsets_, invalid_;
  mobius_native::Edges make_edges(Box&, const GPU_Vector<int>&,
                                  const GPU_Vector<double>&);
};
