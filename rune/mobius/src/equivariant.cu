// Persistent angular channels: all m components of each angular order share
// channel maps and scalar gates, exactly as in equivariant.py.
#include "kernels.cuh"
#include <cmath>

namespace mobius_native {
namespace {
constexpr int threads=128;
int blocks(size_t n) { return int((n+threads-1)/threads); }

__global__ void encoder_kernel(int atoms,int radial,int channels,int source_m,
    int target_m,int order,const double* source,const double* scale,
    const double* weight,double* target) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int width=2*order+1, start=order*order-1;
  if(k>=size_t(atoms)*channels*width) return;
  int angular=k%width+start,c=(k/width)%channels,i=k/(width*channels);
  double sum=0;
  for(int n=0;n<radial;++n)
    sum+=source[(size_t(i)*radial+n)*source_m+angular]*weight[size_t(c)*radial+n];
  target[(size_t(i)*channels+c)*target_m+angular]=sum*scale[angular];
}

__global__ void encoder_backward_kernel(int atoms,int radial,int channels,int source_m,
    int target_m,int order,const double* dtarget,const double* scale,
    const double* weight,double* dsource) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int width=2*order+1,start=order*order-1;
  if(k>=size_t(atoms)*radial*width) return;
  int angular=k%width+start,n=(k/width)%radial,i=k/(width*radial);
  double sum=0;
  for(int c=0;c<channels;++c)
    sum+=dtarget[(size_t(i)*channels+c)*target_m+angular]*weight[size_t(c)*radial+n];
  dsource[(size_t(i)*radial+n)*source_m+angular]=sum*scale[angular];
}

__global__ void direction_mix_kernel(int atoms,int channels,int components,int order,
    bool transpose,const double* source,const double* weight,double* target) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int width=2*order+1,start=order*order-1;
  if(k>=size_t(atoms)*channels*width) return;
  int angular=k%width+start,c=(k/width)%channels,i=k/(width*channels);
  double sum=0;
  for(int b=0;b<channels;++b)
    sum+=source[(size_t(i)*channels+b)*components+angular]*
      weight[transpose?size_t(b)*channels+c:size_t(c)*channels+b];
  target[(size_t(i)*channels+c)*components+angular]=sum;
}

__global__ void probes_kernel(int count,int channels,int maximum,int components,
    const int* center,const int* neighbor,const double* direction,const double* harmonics,
    double* pcenter,double* pneighbor) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(count)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,e=k/(maximum*channels);
  double pc=0,pn=0;
  for(int a=l*l-1;a<(l+1)*(l+1)-1;++a) {
    double y=harmonics[size_t(e)*components+a];
    pc+=direction[(size_t(center[e])*channels+c)*components+a]*y;
    pn+=direction[(size_t(neighbor[e])*channels+c)*components+a]*y;
  }
  pcenter[k]=pc; pneighbor[k]=(l%2?-1.0:1.0)*pn;
}

__global__ void direction_update_kernel(int count,int channels,int maximum,int components,
    const int* center,const int* neighbor,const double* mixed,const double* harmonics,
    const double* logits,const double* cutoff,double* target) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(count)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,e=k/(maximum*channels);
  int angular_width=maximum*channels,offset=(l-1)*channels+c;
  double a=tanh(logits[size_t(e)*2*angular_width+offset]);
  double b=tanh(logits[size_t(e)*2*angular_width+angular_width+offset]);
  for(int m=l*l-1;m<(l+1)*(l+1)-1;++m) {
    double message=cutoff[e]*(a*mixed[(size_t(neighbor[e])*channels+c)*components+m]
      +b*harmonics[size_t(e)*components+m]);
    atomicAdd(target+(size_t(center[e])*channels+c)*components+m,message);
  }
}

__global__ void invariants_kernel(int atoms,int channels,int maximum,int components,
    const double* direction,double* invariants) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(atoms)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,i=k/(maximum*channels);
  double sum=0;
  for(int m=l*l-1;m<(l+1)*(l+1)-1;++m) {
    double v=direction[(size_t(i)*channels+c)*components+m]; sum+=v*v;
  }
  invariants[k]=sum;
}

__global__ void invariants_backward_kernel(int atoms,int channels,int maximum,int components,
    const double* direction,const double* dinvariants,const double* downstream,double* result) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(atoms)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,i=k/(maximum*channels);
  for(int m=l*l-1;m<(l+1)*(l+1)-1;++m) {
    size_t index=(size_t(i)*channels+c)*components+m;
    result[index]=downstream[index]+2.0*direction[index]*dinvariants[k];
  }
}

__global__ void direction_update_backward_kernel(int count,int channels,int maximum,
    int components,const int* center,const int* neighbor,const double* mixed,
    const double* harmonics,const double* logits,const double* cutoff,const double* dtarget,
    double* dmixed,double* dharmonics,double* dlogits,double* dcutoff) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(count)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,e=k/(maximum*channels);
  int aw=maximum*channels,offset=(l-1)*channels+c;
  size_t ia=size_t(e)*2*aw+offset,ib=ia+aw;
  double a=tanh(logits[ia]),b=tanh(logits[ib]),da=0,db=0;
  for(int m=l*l-1;m<(l+1)*(l+1)-1;++m) {
    size_t node=(size_t(neighbor[e])*channels+c)*components+m;
    size_t edge=size_t(e)*components+m;
    double grad=dtarget[(size_t(center[e])*channels+c)*components+m];
    da+=grad*mixed[node]; db+=grad*harmonics[edge];
    atomicAdd(dmixed+node,cutoff[e]*a*grad);
    atomicAdd(dharmonics+edge,cutoff[e]*b*grad);
  }
  dlogits[ia]=cutoff[e]*da*(1.0-a*a);
  dlogits[ib]=cutoff[e]*db*(1.0-b*b);
  atomicAdd(dcutoff+e,a*da+b*db);
}

__global__ void probes_backward_kernel(int count,int channels,int maximum,int components,
    const int* center,const int* neighbor,const double* direction,const double* harmonics,
    const double* dpcenter,const double* dpneighbor,double* ddirection,double* dharmonics) {
  size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=size_t(count)*maximum*channels) return;
  int c=k%channels,l=(k/channels)%maximum+1,e=k/(maximum*channels);
  double dc=dpcenter[k],dn=dpneighbor[k]*(l%2?-1.0:1.0);
  for(int m=l*l-1;m<(l+1)*(l+1)-1;++m) {
    size_t ic=(size_t(center[e])*channels+c)*components+m;
    size_t in=(size_t(neighbor[e])*channels+c)*components+m;
    size_t iy=size_t(e)*components+m;
    atomicAdd(ddirection+ic,dc*harmonics[iy]);
    atomicAdd(ddirection+in,dn*harmonics[iy]);
    atomicAdd(dharmonics+iy,dc*direction[ic]+dn*direction[in]);
  }
}
} // namespace

Tensor direction_forward(const Model& m,const Tensor& moments) {
  int radial=m.i("n_desc_angular"),channels=m.i("equivariant_channels");
  int maximum=m.i("equivariant_l_max"),source_m=moments.cols/radial;
  int target_m=maximum*(maximum+2);
  Tensor result(moments.rows,channels*target_m,Init::uninitialized);
  for(int l=1;l<=maximum;++l) if(moments.rows) {
    encoder_kernel<<<blocks(size_t(moments.rows)*channels*(2*l+1)),threads>>>(moments.rows,
      radial,channels,source_m,target_m,l,moments.data(),m.t("desc.orthonormal_scale").data(),
      m.t("direction_encoder.projections."+std::to_string(l-1)+".weight").data(),result.data());
    launch_check();
  }
  return result;
}

Tensor direction_backward(const Model& m,const Tensor& ddirection) {
  int radial=m.i("n_desc_angular"),channels=m.i("equivariant_channels");
  int maximum=m.i("equivariant_l_max"),descriptor_l=m.i("l_max");
  int source_m=descriptor_l*(descriptor_l+2),target_m=maximum*(maximum+2);
  // Orders above equivariant_l_max are absent from the direction encoder and
  // therefore have zero adjoints in the full descriptor moment tensor.
  Tensor result(ddirection.rows,radial*source_m,
    maximum==descriptor_l ? Init::uninitialized : Init::zero);
  for(int l=1;l<=maximum;++l) if(ddirection.rows) {
    encoder_backward_kernel<<<blocks(size_t(ddirection.rows)*radial*(2*l+1)),threads>>>(ddirection.rows,
      radial,channels,source_m,target_m,l,ddirection.data(),m.t("desc.orthonormal_scale").data(),
      m.t("direction_encoder.projections."+std::to_string(l-1)+".weight").data(),result.data());
    launch_check();
  }
  return result;
}

struct EquivariantCache {
  Tensor hidden,direction,harmonics,normalized,mixed,new_direction;
  Tensor edge_pre,edge_hidden,group_logits,values,direction_logits,scalar_sum;
  Tensor update_input,update_pre,update_active;
  EdgeGeometry geometry;
};

void equivariant_forward(const Model& m,const std::string& p,const Edges& e,
    const Tensor& hidden,const Tensor& direction,const Tensor& harmonics,
    Tensor& new_hidden,Tensor& new_direction,std::shared_ptr<EquivariantCache>& state) {
  state=std::make_shared<EquivariantCache>(); auto& c=*state;
  int channels=m.i("equivariant_channels"),maximum=m.i("equivariant_l_max");
  int components=maximum*(maximum+2),aw=maximum*channels,activation=m.i("activation");
  double eps=m.r("norm_eps");
  c.hidden=hidden; c.direction=direction; c.harmonics=harmonics;
  c.new_direction=add(direction,Tensor(direction.rows,direction.cols));
  if(e.count) {
    c.normalized=norm(hidden,m.t(p+".input_norm.weight"),eps);
    c.mixed=Tensor(e.atoms,channels*components,Init::uninitialized);
    for(int l=1;l<=maximum;++l) {
      direction_mix_kernel<<<blocks(size_t(e.atoms)*channels*(2*l+1)),threads>>>(e.atoms,channels,
        components,l,false,direction.data(),m.t(p+".direction_mix."+std::to_string(l-1)+".weight").data(),c.mixed.data());
      launch_check();
    }
    Tensor pcenter(e.count,aw,Init::uninitialized),pneighbor(e.count,aw,Init::uninitialized);
    probes_kernel<<<blocks(pcenter.size()),threads>>>(e.count,channels,maximum,components,
      e.center.data(),e.neighbor.data(),direction.data(),harmonics.data(),pcenter.data(),pneighbor.data()); launch_check();
    c.geometry=edge_geometry(m,p,e);
    Tensor center_context=gather(linear(c.normalized,m.t(p+".center_context.weight")),e.center,e.count);
    Tensor neighbor_context=gather(linear(c.normalized,m.t(p+".neighbor_context.weight")),e.neighbor,e.count);
    Tensor edge_input=concat({c.geometry.features,center_context,neighbor_context,pcenter,pneighbor});
    c.edge_pre=linear(edge_input,m.t(p+".edge_mlp.0.weight"),m.t(p+".edge_mlp.0.bias"));
    c.edge_hidden=activate(c.edge_pre,activation);
    c.group_logits=linear(c.edge_hidden,m.t(p+".group_gate.weight"),m.t(p+".group_gate.bias"));
    c.values=linear(c.normalized,m.t(p+".value_projection.weight"));
    c.scalar_sum=scalar_aggregate(e,c.group_logits,c.geometry.cutoff,c.values);
    c.direction_logits=linear(c.edge_hidden,m.t(p+".direction_gates.weight"),m.t(p+".direction_gates.bias"));
    direction_update_kernel<<<blocks(size_t(e.count)*aw),threads>>>(e.count,channels,maximum,components,
      e.center.data(),e.neighbor.data(),c.mixed.data(),harmonics.data(),c.direction_logits.data(),
      c.geometry.cutoff.data(),c.new_direction.data()); launch_check();
  } else c.scalar_sum=Tensor(e.atoms,hidden.cols);
  Tensor invariants(e.atoms,aw,Init::uninitialized);
  if(invariants.size()) {
    invariants_kernel<<<blocks(invariants.size()),threads>>>(e.atoms,channels,maximum,components,
      c.new_direction.data(),invariants.data()); launch_check();
  }
  c.update_input=concat({norm(c.scalar_sum,m.t(p+".message_norm.weight"),eps),invariants});
  c.update_pre=linear(c.update_input,m.t(p+".update.0.weight"),m.t(p+".update.0.bias"));
  c.update_active=activate(c.update_pre,activation);
  new_hidden=add(hidden,linear(c.update_active,m.t(p+".update.2.weight"),m.t(p+".update.2.bias")));
  new_direction=c.new_direction;
}

void equivariant_backward(const Model& m,const std::string& p,const Edges& e,
    const EquivariantCache& c,const Tensor& dhidden,const Tensor& ddirection,
    Tensor& previous_dh,Tensor& previous_direction,Tensor& dharmonics,Tensor& dvectors) {
  int channels=m.i("equivariant_channels"),maximum=m.i("equivariant_l_max");
  int components=maximum*(maximum+2),aw=maximum*channels,activation=m.i("activation");
  int md=m.i("message_dim"),base=m.i("message_basis_size")+m.i("pair_dim");
  double eps=m.r("norm_eps");
  Tensor dpre=activate_backward(c.update_pre,linear_backward(dhidden,m.t(p+".update.2.weight")),activation);
  Tensor dinput=linear_backward(dpre,m.t(p+".update.0.weight"));
  Tensor dinvariants=slice(dinput,dhidden.cols,aw);
  Tensor dnew_direction(ddirection.rows,ddirection.cols,Init::uninitialized);
  if(dinvariants.size()) {
    invariants_backward_kernel<<<blocks(dinvariants.size()),threads>>>(e.atoms,channels,maximum,components,
      c.new_direction.data(),dinvariants.data(),ddirection.data(),dnew_direction.data()); launch_check();
  }
  previous_direction=add(dnew_direction,Tensor(dnew_direction.rows,dnew_direction.cols));
  previous_dh=dhidden;
  if(!e.count) return;
  Tensor dsum=norm_backward(c.scalar_sum,slice(dinput,0,dhidden.cols),m.t(p+".message_norm.weight"),eps);
  Tensor dlogits(e.count,c.group_logits.cols,Init::uninitialized);
  Tensor dcutoff(e.count,1),dvalues(e.atoms,c.values.cols);
  scalar_aggregate_backward(e,c.group_logits,c.geometry.cutoff,c.values,dsum,dlogits,dcutoff,dvalues);
  Tensor dnormalized=linear_backward(dvalues,m.t(p+".value_projection.weight"));
  Tensor dedge_hidden=linear_backward(dlogits,m.t(p+".group_gate.weight"));
  Tensor ddirection_logits(e.count,2*aw,Init::uninitialized);
  Tensor dmixed(e.atoms,channels*components);
  direction_update_backward_kernel<<<blocks(size_t(e.count)*aw),threads>>>(e.count,channels,maximum,
    components,e.center.data(),e.neighbor.data(),c.mixed.data(),c.harmonics.data(),c.direction_logits.data(),
    c.geometry.cutoff.data(),dnew_direction.data(),dmixed.data(),dharmonics.data(),ddirection_logits.data(),dcutoff.data());
  launch_check();
  Tensor dtransport(e.atoms,channels*components,Init::uninitialized);
  for(int l=1;l<=maximum;++l) {
    direction_mix_kernel<<<blocks(size_t(e.atoms)*channels*(2*l+1)),threads>>>(e.atoms,channels,
      components,l,true,dmixed.data(),m.t(p+".direction_mix."+std::to_string(l-1)+".weight").data(),dtransport.data());
    launch_check();
  }
  add_inplace(previous_direction,dtransport);
  add_inplace(dedge_hidden,linear_backward(ddirection_logits,m.t(p+".direction_gates.weight")));
  Tensor dedge_pre=activate_backward(c.edge_pre,dedge_hidden,activation);
  Tensor dedge_input=linear_backward(dedge_pre,m.t(p+".edge_mlp.0.weight"));
  Tensor dc=scatter_sum(slice(dedge_input,base,md),e.center,e.atoms);
  Tensor dn=scatter_sum(slice(dedge_input,base+md,md),e.neighbor,e.atoms);
  add_inplace(dnormalized,linear_backward(dc,m.t(p+".center_context.weight")));
  add_inplace(dnormalized,linear_backward(dn,m.t(p+".neighbor_context.weight")));
  Tensor dpc=slice(dedge_input,base+2*md,aw),dpn=slice(dedge_input,base+2*md+aw,aw);
  probes_backward_kernel<<<blocks(size_t(e.count)*aw),threads>>>(e.count,channels,maximum,components,
    e.center.data(),e.neighbor.data(),c.direction.data(),c.harmonics.data(),dpc.data(),dpn.data(),
    previous_direction.data(),dharmonics.data()); launch_check();
  // Harmonic gradients are accumulated separately; this geometry path has no
  // unit-vector contribution and still requires an explicit zero tensor.
  Tensor dunit(e.count,3);
  edge_geometry_backward(e,c.geometry,slice(dedge_input,0,base),dcutoff,dunit,dvectors);
  previous_dh=add(dhidden,norm_backward(c.hidden,dnormalized,m.t(p+".input_norm.weight"),eps));
}
} // namespace mobius_native
