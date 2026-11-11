// Scalar messages and their geometric adjoints. Parameter gradients are not
// needed in MD; all node and edge derivatives are retained.
#include "kernels.cuh"
#include <cmath>

namespace mobius_native {
namespace {
constexpr int threads = 128;
constexpr double pi = 3.141592653589793238462643383279502884;
int blocks(size_t n) { return int((n + threads - 1) / threads); }

__device__ double sigmoid(double x) {
  return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

__global__ void gather_kernel(int count, int width, const int* index,
                              const double* x, double* y) {
  size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < size_t(count) * width) y[k] = x[size_t(index[k / width]) * width + k % width];
}

__global__ void scatter_kernel(int count, int width, const int* index,
                               const double* x, double* y) {
  size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < size_t(count) * width)
    atomicAdd(y + size_t(index[k / width]) * width + k % width, x[k]);
}

__global__ void geometry_kernel(int count, int basis, int pair, int elements,
    double rc, const int* center, const int* neighbor, const int* types,
    const double* vectors, const double* embedding, double* features,
    double* cutoff, double* unit, double* derivatives, double* cutoff_derivative) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= count) return;
  const double* v = vectors + 3 * e;
  double r = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  double safe = fmax(r, 1.0e-12);
  for (int a=0; a<3; ++a) unit[3*e+a] = v[a] / safe;
  double fc = r < rc ? 0.5 * cos(pi*r/rc) + 0.5 : 0.0;
  double dfc = r < rc ? -0.5*pi/rc*sin(pi*r/rc) : 0.0;
  cutoff[e] = fc;
  cutoff_derivative[e] = dfc;
  double x = 2.0*(r/rc-1.0)*(r/rc-1.0)-1.0;
  double dx = 4.0*(r/rc-1.0)/rc;
  double prev = 1.0, curr = x, dprev = 0.0, dcurr = dx;
  for (int b=0; b<basis; ++b) {
    double t, dt;
    if (b == 0) { t = 1.0; dt = 0.0; }
    else if (b == 1) { t = curr; dt = dcurr; }
    else {
      t = 2.0*x*curr-prev;
      dt = 2.0*dx*curr+2.0*x*dcurr-dprev;
      prev=curr; curr=t; dprev=dcurr; dcurr=dt;
    }
    features[size_t(e)*(basis+pair)+b] = 0.5*(t+1.0)*fc;
    derivatives[size_t(e)*basis+b] = 0.5*(dt*fc+(t+1.0)*dfc);
  }
  int pair_index = types[center[e]]*elements+types[neighbor[e]];
  for (int p=0; p<pair; ++p)
    features[size_t(e)*(basis+pair)+basis+p] = embedding[size_t(pair_index)*pair+p];
}

__global__ void geometry_backward_kernel(int count, int basis, int feature_width,
    const double* vectors, const double* unit, const double* basis_derivative,
    const double* cutoff_derivative, const double* dfeatures,
    const double* dcutoff, const double* dunit, double* dvectors) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= count) return;
  double radial = dcutoff[e] * cutoff_derivative[e];
  for (int b=0; b<basis; ++b)
    radial += dfeatures[size_t(e)*feature_width+b]*basis_derivative[size_t(e)*basis+b];
  double r2=0.0, dot=0.0;
  for (int a=0; a<3; ++a) {
    r2 += vectors[3*e+a]*vectors[3*e+a];
    dot += dunit[3*e+a]*unit[3*e+a];
  }
  double r=sqrt(r2), safe=fmax(r,1.0e-12);
  for (int a=0; a<3; ++a) {
    // clamp_min only affects the denominator of r_hat, not |r| in the basis.
    double angular = dunit[3*e+a]/safe;
    if (r >= 1.0e-12) angular -= unit[3*e+a]*dot/r;
    double dr = r > 0.0 ? vectors[3*e+a]/r : 0.0;
    dvectors[3*e+a] += radial*dr+angular;
  }
}

__global__ void scalar_forward_kernel(int count, int width, int groups,
    const int* center, const int* neighbor, const double* logits,
    const double* cutoff, const double* values, double* sum) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (k >= size_t(count)*width) return;
  int e=k/width, c=k%width, g=c/(width/groups);
  atomicAdd(sum+size_t(center[e])*width+c,
    cutoff[e]*sigmoid(logits[size_t(e)*groups+g])*values[size_t(neighbor[e])*width+c]);
}

__global__ void scalar_backward_kernel(int count, int width, int groups,
    const int* center, const int* neighbor, const double* logits,
    const double* cutoff, const double* values, const double* dsum,
    double* dlogits, double* dcutoff, double* dvalues) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (k >= size_t(count)*groups) return;
  int e=k/groups, g=k%groups, gw=width/groups;
  double s=sigmoid(logits[k]), dot=0.0;
  for (int c=g*gw; c<(g+1)*gw; ++c) {
    double grad=dsum[size_t(center[e])*width+c];
    dot += grad*values[size_t(neighbor[e])*width+c];
    atomicAdd(dvalues+size_t(neighbor[e])*width+c, grad*cutoff[e]*s);
  }
  dlogits[k]=dot*cutoff[e]*s*(1.0-s);
  atomicAdd(dcutoff+e,dot*s);
}

__global__ void vector_forward_kernel(int count, int channels, const int* center,
    const int* neighbor, const double* logits, const double* cutoff,
    const double* source, const double* unit, double* sum) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (k >= size_t(count)*channels*3) return;
  int a=k%3, c=(k/3)%channels, e=k/(3*channels);
  double factor=cutoff[e]*sigmoid(logits[size_t(e)*channels+c])*
    source[size_t(neighbor[e])*channels+c];
  atomicAdd(sum+(size_t(center[e])*channels+c)*3+a, factor*unit[3*e+a]);
}

__global__ void vector_mix_kernel(int atoms, int channels, bool transpose,
    const double* x, const double* weight, double* y) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (k >= size_t(atoms)*channels*3) return;
  int a=k%3, c=(k/3)%channels, i=k/(3*channels);
  double sum=0;
  for(int b=0; b<channels; ++b)
    sum += x[(size_t(i)*channels+b)*3+a]*
      weight[transpose ? size_t(b)*channels+c : size_t(c)*channels+b];
  y[k]=sum;
}

__global__ void vector_norm_kernel(int atoms, int channels, const double* mixed,
                                  double* invariant) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k >= size_t(atoms)*channels) return;
  double sum=0;
  for(int a=0;a<3;++a) sum+=mixed[3*k+a]*mixed[3*k+a];
  invariant[k]=sum;
}

__global__ void vector_norm_backward_kernel(size_t count, const double* mixed,
    const double* dinvariant, double* dmixed) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k<count) dmixed[k]=2.0*mixed[k]*dinvariant[k/3];
}

__global__ void vector_backward_kernel(int count, int channels, const int* center,
    const int* neighbor, const double* logits, const double* cutoff,
    const double* source, const double* unit, const double* dsum,
    double* dlogits, double* dcutoff, double* dsource, double* dunit) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k >= size_t(count)*channels) return;
  int e=k/channels, c=k%channels;
  double s=sigmoid(logits[k]), value=source[size_t(neighbor[e])*channels+c], dot=0;
  for(int a=0;a<3;++a) {
    double d=dsum[(size_t(center[e])*channels+c)*3+a];
    dot+=d*unit[3*e+a];
    atomicAdd(dunit+3*e+a, d*cutoff[e]*s*value);
  }
  dlogits[k]=cutoff[e]*value*dot*s*(1.0-s);
  atomicAdd(dcutoff+e,s*value*dot);
  atomicAdd(dsource+size_t(neighbor[e])*channels+c,cutoff[e]*s*dot);
}
}  // namespace

Tensor gather(const Tensor& x, const Buffer<int>& index, int count) {
  Tensor y(count,x.cols,Init::uninitialized);
  if(y.size()) { gather_kernel<<<blocks(y.size()),threads>>>(count,x.cols,index.data(),x.data(),y.data()); launch_check(); }
  return y;
}

Tensor scatter_sum(const Tensor& x, const Buffer<int>& index, int rows) {
  // Scatter may leave rows untouched and accumulates every contribution.
  Tensor y(rows,x.cols);
  if(x.size()) { scatter_kernel<<<blocks(x.size()),threads>>>(x.rows,x.cols,index.data(),x.data(),y.data()); launch_check(); }
  return y;
}

EdgeGeometry edge_geometry(const Model& m, const std::string& prefix, const Edges& edges) {
  int basis=m.i("message_basis_size"), pair=m.i("pair_dim");
  EdgeGeometry g;
  g.features=Tensor(edges.count,basis+pair,Init::uninitialized);
  g.cutoff=Tensor(edges.count,1,Init::uninitialized);
  g.unit=Tensor(edges.count,3,Init::uninitialized);
  g.basis_derivative=Tensor(edges.count,basis,Init::uninitialized);
  g.cutoff_derivative=Tensor(edges.count,1,Init::uninitialized);
  if(edges.count) {
    geometry_kernel<<<blocks(edges.count),threads>>>(edges.count,basis,pair,m.i("n_types"),
      m.r("message_cutoff"),edges.center.data(),edges.neighbor.data(),edges.types.data(),
      edges.vectors.data(),m.t(prefix+".pair_embedding.weight").data(),g.features.data(),
      g.cutoff.data(),g.unit.data(),g.basis_derivative.data(),g.cutoff_derivative.data());
    launch_check();
  }
  return g;
}

void edge_geometry_backward(const Edges& e, const EdgeGeometry& g,
    const Tensor& dfeatures, const Tensor& dcutoff, const Tensor& dunit, Tensor& dvectors) {
  if(e.count) {
    geometry_backward_kernel<<<blocks(e.count),threads>>>(e.count,g.basis_derivative.cols,
      dfeatures.cols,e.vectors.data(),g.unit.data(),g.basis_derivative.data(),
      g.cutoff_derivative.data(),dfeatures.data(),dcutoff.data(),dunit.data(),dvectors.data());
    launch_check();
  }
}

Tensor scalar_aggregate(const Edges& e,const Tensor& logits,const Tensor& cutoff,
                        const Tensor& values) {
  Tensor sum(e.atoms,values.cols);
  if(e.count) {
    scalar_forward_kernel<<<blocks(size_t(e.count)*values.cols),threads>>>(e.count,
      values.cols,logits.cols,e.center.data(),e.neighbor.data(),logits.data(),
      cutoff.data(),values.data(),sum.data()); launch_check();
  }
  return sum;
}

void scalar_aggregate_backward(const Edges& e,const Tensor& logits,const Tensor& cutoff,
    const Tensor& values,const Tensor& dsum,Tensor& dlogits,Tensor& dcutoff,Tensor& dvalues) {
  if(e.count) {
    scalar_backward_kernel<<<blocks(logits.size()),threads>>>(e.count,values.cols,
      logits.cols,e.center.data(),e.neighbor.data(),logits.data(),cutoff.data(),
      values.data(),dsum.data(),dlogits.data(),dcutoff.data(),dvalues.data()); launch_check();
  }
}

struct MessageCache {
  Tensor hidden,normalized,edge_pre,edge_hidden,group_logits,values,scalar_sum;
  Tensor vector_logits,source,vector_sum,mixed;
  Tensor update_input,update_pre,update_active;
  EdgeGeometry geometry;
};

Tensor message_forward(const Model& m,const std::string& p,const Edges& e,
    const Tensor& hidden,std::shared_ptr<MessageCache>& state) {
  state=std::make_shared<MessageCache>(); auto& c=*state;
  int channels=m.i("vector_channels"), activation=m.i("activation");
  double eps=m.r("norm_eps");
  c.hidden=hidden;
  // With no edges the vector invariants are exactly zero; otherwise the norm
  // kernel writes every channel, including zeros for isolated atoms.
  Tensor invariants(e.atoms,channels,e.count ? Init::uninitialized : Init::zero);
  if(e.count) {
    c.normalized=norm(hidden,m.t(p+".input_norm.weight"),eps);
    c.geometry=edge_geometry(m,p,e);
    c.edge_pre=linear(c.geometry.features,m.t(p+".edge_mlp.0.weight"),m.t(p+".edge_mlp.0.bias"));
    c.edge_hidden=activate(c.edge_pre,activation);
    c.group_logits=linear(c.edge_hidden,m.t(p+".group_gate.weight"),m.t(p+".group_gate.bias"));
    c.values=linear(c.normalized,m.t(p+".value_projection.weight"));
    c.scalar_sum=scalar_aggregate(e,c.group_logits,c.geometry.cutoff,c.values);
    if(channels) {
      c.vector_logits=linear(c.edge_hidden,m.t(p+".vector_gate.weight"),m.t(p+".vector_gate.bias"));
      c.source=linear(c.normalized,m.t(p+".vector_source.weight"));
      c.vector_sum=Tensor(e.atoms,channels*3);
      c.mixed=Tensor(e.atoms,channels*3,Init::uninitialized);
      vector_forward_kernel<<<blocks(size_t(e.count)*channels*3),threads>>>(e.count,channels,
        e.center.data(),e.neighbor.data(),c.vector_logits.data(),c.geometry.cutoff.data(),
        c.source.data(),c.geometry.unit.data(),c.vector_sum.data()); launch_check();
      vector_mix_kernel<<<blocks(c.mixed.size()),threads>>>(e.atoms,channels,false,
        c.vector_sum.data(),m.t(p+".vector_mix.weight").data(),c.mixed.data()); launch_check();
      vector_norm_kernel<<<blocks(invariants.size()),threads>>>(e.atoms,channels,c.mixed.data(),invariants.data()); launch_check();
    }
  } else c.scalar_sum=Tensor(e.atoms,hidden.cols);
  c.update_input=concat({norm(c.scalar_sum,m.t(p+".message_norm.weight"),eps),invariants});
  c.update_pre=linear(c.update_input,m.t(p+".update.0.weight"),m.t(p+".update.0.bias"));
  c.update_active=activate(c.update_pre,activation);
  return add(hidden,linear(c.update_active,m.t(p+".update.2.weight"),m.t(p+".update.2.bias")));
}

Tensor message_backward(const Model& m,const std::string& p,const Edges& e,
    const MessageCache& c,const Tensor& dhidden,Tensor& dvectors) {
  // No edge means the update is position-independent and the hidden derivative
  // is exactly the residual identity (all parameter gradients are irrelevant).
  if(!e.count) return dhidden;
  int channels=m.i("vector_channels"), activation=m.i("activation");
  double eps=m.r("norm_eps");
  Tensor dpre=activate_backward(c.update_pre,linear_backward(dhidden,m.t(p+".update.2.weight")),activation);
  Tensor dinput=linear_backward(dpre,m.t(p+".update.0.weight"));
  Tensor dsum=norm_backward(c.scalar_sum,slice(dinput,0,dhidden.cols),m.t(p+".message_norm.weight"),eps);
  Tensor dlogits(e.count,c.group_logits.cols,Init::uninitialized);
  // These derivatives accumulate over groups/neighbors. dunit must also be
  // zero when vector channels are disabled, as geometry backward reads it.
  Tensor dcutoff(e.count,1),dvalues(e.atoms,c.values.cols);
  Tensor dunit(e.count,3);
  scalar_aggregate_backward(e,c.group_logits,c.geometry.cutoff,c.values,dsum,dlogits,dcutoff,dvalues);
  Tensor dnormalized=linear_backward(dvalues,m.t(p+".value_projection.weight"));
  Tensor dedge_hidden=linear_backward(dlogits,m.t(p+".group_gate.weight"));
  if(channels) {
    Tensor dinvariants=slice(dinput,dhidden.cols,channels);
    Tensor dmixed(e.atoms,channels*3,Init::uninitialized);
    Tensor dvsum(e.atoms,channels*3,Init::uninitialized);
    vector_norm_backward_kernel<<<blocks(dmixed.size()),threads>>>(dmixed.size(),c.mixed.data(),dinvariants.data(),dmixed.data()); launch_check();
    vector_mix_kernel<<<blocks(dvsum.size()),threads>>>(e.atoms,channels,true,dmixed.data(),m.t(p+".vector_mix.weight").data(),dvsum.data()); launch_check();
    Tensor dvector_logits(e.count,channels,Init::uninitialized),dsource(e.atoms,channels);
    vector_backward_kernel<<<blocks(dvector_logits.size()),threads>>>(e.count,channels,
      e.center.data(),e.neighbor.data(),c.vector_logits.data(),c.geometry.cutoff.data(),
      c.source.data(),c.geometry.unit.data(),dvsum.data(),dvector_logits.data(),dcutoff.data(),
      dsource.data(),dunit.data()); launch_check();
    add_inplace(dnormalized,linear_backward(dsource,m.t(p+".vector_source.weight")));
    add_inplace(dedge_hidden,linear_backward(dvector_logits,m.t(p+".vector_gate.weight")));
  }
  Tensor dedge_pre=activate_backward(c.edge_pre,dedge_hidden,activation);
  Tensor dfeatures=linear_backward(dedge_pre,m.t(p+".edge_mlp.0.weight"));
  edge_geometry_backward(e,c.geometry,dfeatures,dcutoff,dunit,dvectors);
  return add(dhidden,norm_backward(c.hidden,dnormalized,m.t(p+".input_norm.weight"),eps));
}
}  // namespace mobius_native
