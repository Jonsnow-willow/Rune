#include "kernels.cuh"
#include <cmath>

namespace mobius_native {
namespace {
constexpr int threads=128;
__global__ void zbl_kernel(int edges, int ntypes, bool flexible, double inner, double outer,
                           const int* center, const int* neighbor, const int* types,
                           const double* vectors, const double* numbers,
                           const double* universal, const double* pairs,
                           double* energy, double* gradient) {
  int edge=blockIdx.x*blockDim.x+threadIdx.x;
  if(edge>=edges) return;
  const int i=center[edge], j=neighbor[edge], ti=types[i], tj=types[j];
  const double x=vectors[3*edge],y=vectors[3*edge+1],z=vectors[3*edge+2];
  const double raw_r=sqrt(x*x+y*y+z*z), r=fmax(raw_r,1.e-12);
  const double* coeff=universal;
  if(flexible) {
    int low=min(ti,tj), high=max(ti,tj);
    int pair=low*ntypes-low*(low-1)/2+high-low;
    const double* p=pairs+10*pair;
    inner=p[0]; outer=p[1]; coeff=p+2;
  }
  if(r>=outer) return;
  const double ai=(pow(numbers[ti],0.23)+pow(numbers[tj],0.23))*2.134563;
  const double prefactor=14.399645*numbers[ti]*numbers[tj];
  double s=0.,ds=0.;
  for(int k=0;k<4;++k) {
    const double rate=coeff[2*k+1]*ai;
    const double term=coeff[2*k]*exp(-rate*r);
    s+=term; ds-=rate*term;
  }
  double fc=1.,dfc=0.;
  if(r>=inner) {
    const double width=fmax(outer-inner,1.e-12);
    const double angle=3.14159265358979323846*(r-inner)/width;
    fc=0.5*cos(angle)+0.5;
    dfc=-0.5*sin(angle)*3.14159265358979323846/width;
  }
  const double screened=prefactor*s/r;
  atomicAdd(energy+i,0.5*screened*fc);
  const double dr=raw_r>=1.e-12 ? 0.5*(prefactor*(ds/r-s/(r*r))*fc+screened*dfc)/r : 0.;
  gradient[3*edge]+=dr*x;
  gradient[3*edge+1]+=dr*y;
  gradient[3*edge+2]+=dr*z;
}
__global__ void energy_kernel(int n,const double* energy,double* potential) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) potential[i]+=energy[i];
}
__global__ void force_kernel(int edges,int n,const int* center,const int* neighbor,
                             const double* vectors,const double* gradient,
                             double* forces,double* virial) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e>=edges) return;
  int i=center[e],j=neighbor[e];
  for(int a=0;a<3;++a) {
    double g=gradient[3*e+a];
    atomicAdd(forces+a*n+i,g);
    atomicAdd(forces+a*n+j,-g);
  }
  // GPUMD uses -r (x) g, the transpose of RUNE's atomic convention,
  // stored as xx,yy,zz,xy,xz,yz,yx,zx,zy in nine atom arrays.
  const int a[9]={0,1,2,0,0,1,1,2,2};
  const int b[9]={0,1,2,1,2,2,0,0,1};
  for(int k=0;k<9;++k)
    atomicAdd(virial+k*n+j,-vectors[3*e+a[k]]*gradient[3*e+b[k]]);
}
} // namespace
void add_zbl(const Model& m,const Edges& e,Tensor& energy,Tensor& gradient) {
  if(!m.i("zbl_enabled") || !e.count) return;
  zbl_kernel<<<(e.count+threads-1)/threads,threads>>>(e.count,m.i("n_types"),m.i("zbl_flexible"),
      m.r("zbl_rc_inner"),m.r("zbl_rc_outer"),e.center.data(),e.neighbor.data(),e.types.data(),
      e.vectors.data(),m.t("zbl.atomic_numbers").data(),m.t("zbl.universal_coefficients").data(),
      m.t("zbl.pair_parameters").data(),energy.data(),gradient.data());
  launch_check();
}
void accumulate_outputs(const Edges& e,const Tensor& energy,const Tensor& gradient,
                        double* potential,double* forces,double* virial) {
  if(e.atoms) energy_kernel<<<(e.atoms+threads-1)/threads,threads>>>(e.atoms,energy.data(),potential);
  if(e.count) force_kernel<<<(e.count+threads-1)/threads,threads>>>(e.count,e.atoms,e.center.data(),
      e.neighbor.data(),e.vectors.data(),gradient.data(),forces,virial);
  launch_check();
}
} // namespace mobius_native
