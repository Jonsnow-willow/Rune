#include "kernels.cuh"
#include <cublas_v2.h>
#include <cmath>

namespace mobius_native {
namespace {
constexpr int threads = 128;
constexpr int warp_width = 32;
constexpr int rows_per_block = threads / warp_width;
static_assert(threads % warp_width == 0, "Möbius row reductions require complete warps");
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Möbius cuBLAS failure: " + std::to_string(int(status)));
}
cublasHandle_t handle() {
  struct Handle {
    cublasHandle_t value;
    Handle() { blas_check(cublasCreate(&value)); }
    ~Handle() { cublasDestroy(value); }
  };
  static Handle h;
  return h.value;
}
void same_shape(const Tensor& a, const Tensor& b) {
  if (a.rows != b.rows || a.cols != b.cols)
    throw std::runtime_error("Möbius tensor shape mismatch");
}
__global__ void bias_kernel(size_t size, int width, const double* b, double* y) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < size) y[k] += b[k % width];
}
__global__ void scale_kernel(size_t size, int width, const double* x, const double* w, double* y) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < size) y[k] = x[k] * w[k % width];
}
__global__ void add_kernel(size_t size, const double* a, const double* b, double* y) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < size) y[k] = a[k] + b[k];
}
__global__ void activation_kernel(size_t size, const double* x, const double* dy,
                                  int kind, double* y) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= size) return;
  const double v = x[k];
  double f, d;
  if (kind == 0) { f = tanh(v); d = 1.0 - f*f; }
  else if (kind == 1) {
    const double s = 1.0 / (1.0 + exp(-v));
    f = v*s; d = s + v*s*(1.0-s);
  } else {
    const double c = 0.5 * (1.0 + erf(v * 0.7071067811865475244));
    f = v*c; d = c + v*exp(-0.5*v*v)*0.39894228040143267794;
  }
  y[k] = dy ? dy[k]*d : f;
}
// Each complete warp owns one atom. Inactive channel lanes contribute zero,
// including widths below 32 and final partial channel groups.
__device__ __forceinline__ double warp_sum(double value) {
  for (int offset = warp_width / 2; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}
template <bool backward>
__global__ void norm_kernel(int rows, int width, const double* x, const double* dy,
                            const double* weight, double eps, double* y) {
  const int row = blockIdx.x * rows_per_block + threadIdx.x / warp_width;
  const int lane = threadIdx.x % warp_width;
  // The decision is uniform within each warp, so all shuffle participants live.
  if (row >= rows) return;
  double square = 0.0, dot = 0.0;
  for (int c=lane; c<width; c+=warp_width) {
    const double v = x[row*width+c];
    square += v*v;
    if (backward) dot += v*weight[c]*dy[row*width+c];
  }
  square = warp_sum(square);
  if (backward) dot = warp_sum(dot);
  double s = 0.0;
  if (!lane) s = 1.0/sqrt(square/width+eps);
  s = __shfl_sync(0xffffffffu, s, 0);
  if (backward) dot = __shfl_sync(0xffffffffu, dot, 0);
  for (int c=lane; c<width; c+=warp_width) {
    int k = row*width+c;
    y[k] = backward ? s*weight[c]*dy[k] - x[k]*s*s*s*dot/width : x[k]*s*weight[c];
  }
}
__global__ void copy_columns_kernel(int rows, int width, int src_width, int src_start,
                                    int dst_width, int dst_start,
                                    const double* src, double* dst) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < rows*width)
    dst[(k/width)*dst_width+dst_start+k%width] = src[(k/width)*src_width+src_start+k%width];
}
__global__ void embed_kernel(int n, int d, const int* types, const double* weight, double* h) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k<n*d) h[k] += weight[types[k/d]*d+k%d];
}
// The native evaluator always needs dE/dh for conservative forces. Compute it
// together with E, without materializing normalized h or gathered head weights.
__global__ void readout_kernel(int n, int d, const int* types, const double* h,
                               const double* norm_weight, double eps,
                               const double* w, const double* b, double* e,
                               double* dh) {
  const int i = blockIdx.x * rows_per_block + threadIdx.x / warp_width;
  const int lane = threadIdx.x % warp_width;
  if(i>=n) return;
  const int type = types[i];
  double square = 0.0, dot = 0.0;
  for (int c=lane; c<d; c+=warp_width) {
    const double value = h[i*d+c];
    square += value*value;
    dot += value*norm_weight[c]*w[type*d+c];
  }
  square = warp_sum(square);
  dot = warp_sum(dot);
  double s = 0.0;
  if (!lane) s = 1.0/sqrt(square/d+eps);
  s = __shfl_sync(0xffffffffu, s, 0);
  dot = __shfl_sync(0xffffffffu, dot, 0);
  double energy = 0.0;
  for (int c=lane; c<d; c+=warp_width) {
    const int k = i*d+c;
    const double value = h[k], gamma = norm_weight[c], head = w[type*d+c];
    // Preserve the original per-channel multiplication order. Only the
    // channel-sum reduction order changes from the serial reference.
    energy += (value*s*gamma)*head;
    dh[k] = s*gamma*head - value*s*s*s*dot/d;
  }
  energy = warp_sum(energy);
  if (!lane) e[i] = b[type]+energy;
}
Tensor scale(const Tensor& x, const Tensor& w) {
  if(w.size()!=size_t(x.cols)) throw std::runtime_error("Möbius scale dimension mismatch");
  Tensor y(x.rows,x.cols,Init::uninitialized);
  if(y.size()) scale_kernel<<<(y.size()+threads-1)/threads,threads>>>(y.size(),x.cols,x.data(),w.data(),y.data());
  launch_check(); return y;
}
} // namespace

Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) {
  if(x.cols!=w.cols || (b.size() && b.size()!=size_t(w.rows)))
    throw std::runtime_error("Möbius linear dimension mismatch");
  // beta=0 makes GEMM overwrite every output element.
  Tensor y(x.rows,w.rows,Init::uninitialized);
  if(!y.size()) return y;
  const double one=1.0, zero=0.0;
  blas_check(cublasDgemm(handle(),CUBLAS_OP_T,CUBLAS_OP_N,w.rows,x.rows,x.cols,
                         &one,w.data(),w.cols,x.data(),x.cols,&zero,y.data(),y.cols));
  if(b.size()) bias_kernel<<<(y.size()+threads-1)/threads,threads>>>(y.size(),y.cols,b.data(),y.data());
  launch_check(); return y;
}
Tensor linear_backward(const Tensor& dy, const Tensor& w) {
  if(dy.cols!=w.rows) throw std::runtime_error("Möbius linear backward dimension mismatch");
  Tensor dx(dy.rows,w.cols,Init::uninitialized);
  if(!dx.size()) return dx;
  const double one=1.0, zero=0.0;
  blas_check(cublasDgemm(handle(),CUBLAS_OP_N,CUBLAS_OP_N,w.cols,dy.rows,w.rows,
                         &one,w.data(),w.cols,dy.data(),dy.cols,&zero,dx.data(),dx.cols));
  return dx;
}
Tensor activate(const Tensor& x, int kind) {
  Tensor y(x.rows,x.cols,Init::uninitialized);
  if(y.size()) activation_kernel<<<(y.size()+threads-1)/threads,threads>>>(y.size(),x.data(),nullptr,kind,y.data());
  launch_check(); return y;
}
Tensor activate_backward(const Tensor& x,const Tensor& dy,int kind) {
  same_shape(x,dy); Tensor y(x.rows,x.cols,Init::uninitialized);
  if(y.size()) activation_kernel<<<(y.size()+threads-1)/threads,threads>>>(y.size(),x.data(),dy.data(),kind,y.data());
  launch_check(); return y;
}
Tensor norm(const Tensor& x,const Tensor& w,double eps) {
  if(w.size()!=size_t(x.cols) || !x.cols) throw std::runtime_error("Möbius norm dimension mismatch");
  Tensor y(x.rows,x.cols,Init::uninitialized);
  if(x.rows) norm_kernel<false><<<(x.rows+rows_per_block-1)/rows_per_block,threads>>>(x.rows,x.cols,x.data(),nullptr,w.data(),eps,y.data());
  launch_check(); return y;
}
Tensor norm_backward(const Tensor& x,const Tensor& dy,const Tensor& w,double eps) {
  same_shape(x,dy);
  if(w.size()!=size_t(x.cols) || !x.cols) throw std::runtime_error("Möbius norm backward dimension mismatch");
  Tensor y(x.rows,x.cols,Init::uninitialized);
  if(x.rows) norm_kernel<true><<<(x.rows+rows_per_block-1)/rows_per_block,threads>>>(x.rows,x.cols,x.data(),dy.data(),w.data(),eps,y.data());
  launch_check(); return y;
}
Tensor add(const Tensor& a,const Tensor& b) {
  same_shape(a,b); Tensor y(a.rows,a.cols,Init::uninitialized);
  if(y.size()) add_kernel<<<(y.size()+threads-1)/threads,threads>>>(y.size(),a.data(),b.data(),y.data());
  launch_check(); return y;
}
void add_inplace(Tensor& a,const Tensor& b) {
  same_shape(a,b);
  if(a.size()) add_kernel<<<(a.size()+threads-1)/threads,threads>>>(a.size(),a.data(),b.data(),a.data());
  launch_check();
}
Tensor concat(const std::vector<Tensor>& parts) {
  if(parts.empty()) throw std::runtime_error("Möbius empty concatenation");
  int rows=parts[0].rows, width=0;
  for(const auto& p:parts) {
    if(p.rows!=rows) throw std::runtime_error("Möbius concat row mismatch");
    width+=p.cols;
  }
  Tensor y(rows,width,Init::uninitialized); int start=0;
  for(const auto& p:parts) {
    if(p.size()) copy_columns_kernel<<<(p.size()+threads-1)/threads,threads>>>(rows,p.cols,p.cols,0,width,start,p.data(),y.data());
    start+=p.cols;
  }
  launch_check(); return y;
}
Tensor slice(const Tensor& x,int start,int width) {
  if(start<0 || width<0 || start+width>x.cols) throw std::runtime_error("Möbius invalid slice");
  Tensor y(x.rows,width,Init::uninitialized);
  if(y.size()) copy_columns_kernel<<<(y.size()+threads-1)/threads,threads>>>(x.rows,width,x.cols,start,width,0,x.data(),y.data());
  launch_check(); return y;
}

struct LocalCache {
  std::vector<Tensor> input, preactivation;
};
Tensor local_forward(const Model& m,const Edges& e,const Tensor& q,std::shared_ptr<LocalCache>& cache) {
  cache=std::make_shared<LocalCache>();
  Tensor h=linear(scale(q,m.t("q_scaler")),m.t("input_projection.weight"),m.t("input_projection.bias"));
  if(h.size()) embed_kernel<<<(h.size()+threads-1)/threads,threads>>>(h.rows,h.cols,e.types.data(),m.t("element_embedding.weight").data(),h.data());
  launch_check();
  for(int b=0;b<m.i("blocks");++b) {
    std::string p="residual_blocks."+std::to_string(b);
    cache->input.push_back(h);
    Tensor v=norm(h,m.t(p+".norm.weight"),m.r("norm_eps"));
    Tensor pre=linear(v,m.t(p+".ffn.0.weight"),m.t(p+".ffn.0.bias"));
    cache->preactivation.push_back(pre);
    h=add(h,linear(activate(pre,m.i("activation")),m.t(p+".ffn.2.weight"),m.t(p+".ffn.2.bias")));
  }
  return h;
}
Tensor local_backward(const Model& m,const LocalCache& cache,const Tensor& upstream) {
  Tensor dh=upstream;
  for(int b=m.i("blocks")-1;b>=0;--b) {
    std::string p="residual_blocks."+std::to_string(b);
    Tensor d=linear_backward(dh,m.t(p+".ffn.2.weight"));
    d=activate_backward(cache.preactivation[b],d,m.i("activation"));
    d=linear_backward(d,m.t(p+".ffn.0.weight"));
    dh=add(dh,norm_backward(cache.input[b],d,m.t(p+".norm.weight"),m.r("norm_eps")));
  }
  return scale(linear_backward(dh,m.t("input_projection.weight")),m.t("q_scaler"));
}
struct ReadoutCache { Tensor gradient; };
Tensor readout_forward(const Model& m,const Edges& e,const Tensor& h,std::shared_ptr<ReadoutCache>& cache) {
  if(m.t("output_norm.weight").size()!=size_t(h.cols) || !h.cols)
    throw std::runtime_error("Möbius norm dimension mismatch");
  cache=std::make_shared<ReadoutCache>();
  cache->gradient=Tensor(h.rows,h.cols,Init::uninitialized);
  Tensor energy(h.rows,1,Init::uninitialized);
  if(h.rows) readout_kernel<<<(h.rows+rows_per_block-1)/rows_per_block,threads>>>(h.rows,h.cols,e.types.data(),h.data(),m.t("output_norm.weight").data(),m.r("norm_eps"),m.t("energy_heads.weight").data(),m.t("energy_heads.bias").data(),energy.data(),cache->gradient.data());
  launch_check(); return energy;
}
Tensor readout_backward(const Model&,const Edges&,const ReadoutCache& cache) {
  return cache.gradient;
}
} // namespace mobius_native
