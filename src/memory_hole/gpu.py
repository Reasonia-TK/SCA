"""CuPy RawKernel transport, using the same P1 mesh and face intersections as CPU."""

from __future__ import annotations

import numpy as np

CUDA_SOURCE = r"""
#define INFINITY __int_as_float(0x7f800000)
__device__ double hit_time(double b, double v, double a, double lim) {
    b=fmax(b,0.0); const double tiny=1e-22;
    if(fabs(a)*lim<1e-12*fmax(fabs(v),1.0)) {
        if(v<0) { double t=-b/v; if(t>tiny && t<=lim) return t;
                   if(t<=tiny && b<1e-8) return tiny; }
        return INFINITY;
    }
    double disc=v*v-4*a*b;
    if(disc<0) return INFINITY;
    double q=-0.5*(v+copysign(sqrt(disc),v));
    double roots[2]={q/a, q!=0?b/q:INFINITY}; double best=INFINITY;
    for(int k=0;k<2;k++) { double t=roots[k];
        if(t>=0 && t<=lim && v+2*a*t<0) best=fmin(best,fmax(t,tiny)); }
    return best;
}
extern "C" __global__ void transport(
 const double* p0,const double* v0,const long long* starts,const double* charges,const double* masses,
 const double* field,const double* inv,const double* grad,const long long* neighbors,
 const signed char* kind,const long long* patch,int n,int maxsteps,double maxdt,double cell,double fraction,int np,
 double* pend,double* vend,signed char* status,long long* patches,long long* finaltet,double* times,
 double* normals,double* angles,double* paths,long long* lengths,double* path_times) {
 int j=blockDim.x*blockIdx.x+threadIdx.x; if(j>=n) return;
 double p[3],v[3]; for(int k=0;k<3;k++){p[k]=p0[3*j+k];v[k]=v0[3*j+k];}
 long long tet=starts[j]; int stored=0; double elapsed=0;
 if(j<np){for(int k=0;k<3;k++)paths[(j*128)*3+k]=p[k];path_times[j*128]=0;stored=1;}
 if(tet<0) return;
 for(int step=0;step<maxsteps;step++){
   double a[3],speed2=0,acc2=0;
   for(int k=0;k<3;k++){a[k]=field[3*tet+k]*charges[j]/masses[j];speed2+=v[k]*v[k];acc2+=a[k]*a[k];}
   double dt=fmin(maxdt,fraction*cell/fmax(sqrt(speed2),1.0));
   if(acc2>0)dt=fmin(dt,sqrt(2*fraction*cell/sqrt(acc2)));
   int side=-1;
   for(int face=0;face<4;face++){
     double b=inv[16*tet+face],bv=0,ba=0;
     for(int k=0;k<3;k++){b+=p[k]/1e-9*inv[16*tet+(k+1)*4+face];
       bv+=grad[12*tet+face*3+k]*v[k];ba+=0.5*grad[12*tet+face*3+k]*a[k];}
     double t=hit_time(b,bv,ba,dt);if(t<=dt){dt=t;side=face;}
   }
   for(int k=0;k<3;k++){p[k]+=v[k]*dt+0.5*a[k]*dt*dt;v[k]+=a[k]*dt;}elapsed+=dt;
   if(j<np && step%(maxsteps/120>1?maxsteps/120:1)==0 && stored<126){
     for(int k=0;k<3;k++)paths[(j*128+stored)*3+k]=p[k];path_times[j*128+stored]=elapsed;stored++;}
   if(side>=0){int outcome=kind[tet*4+side];
     if(outcome<0){tet=neighbors[tet*4+side];continue;}
     status[j]=outcome;patches[j]=patch[tet*4+side];double norm2=0,dot=0,vsq=0;
     for(int k=0;k<3;k++){double g=grad[12*tet+side*3+k];norm2+=g*g;}
     for(int k=0;k<3;k++){double normal=-grad[12*tet+side*3+k]/sqrt(norm2);
       normals[3*j+k]=normal;dot+=v[k]*normal;vsq+=v[k]*v[k];}
     angles[j]=acos(fmin(1.0,fmax(0.0,dot/fmax(sqrt(vsq),1e-30))))*180/3.141592653589793;
     break;
   }
 }
 for(int k=0;k<3;k++){pend[j*3+k]=p[k];vend[j*3+k]=v[k];}
 times[j]=elapsed;finaltet[j]=tet;
 if(j<np){for(int k=0;k<3;k++)paths[(j*128+stored)*3+k]=p[k];path_times[j*128+stored]=elapsed;lengths[j]=stored+1;}
}
"""


class GPUTracer:
    def __init__(self, mesh, face_kind, face_patch):
        from .cuda_runtime import prepare_cuda_runtime

        prepare_cuda_runtime()
        import cupy as cp

        self.cp = cp
        memory = sum(
            array.nbytes for array in (mesh.inverse, mesh.gradients, mesh.neighbors, face_kind, face_patch)
        )
        free, _ = cp.cuda.runtime.memGetInfo()
        if memory > min(8 * 1024**3, free * 0.8):
            raise MemoryError("GPUメッシュがVRAM設計目標または空き容量を超えます。CPUを選択してください。")
        self.fixed = tuple(
            cp.asarray(np.ascontiguousarray(a))
            for a in (mesh.inverse, mesh.gradients, mesh.neighbors, face_kind, face_patch)
        )
        self.kernel = cp.RawKernel(CUDA_SOURCE, "transport", options=("--std=c++17",))
        self.field_key = None
        self.device_field = None
        self.phase_fields = {}

    def begin_step(self, fields):
        self.phase_fields.clear()
        self.field_key = None
        self.device_field = None
        total = sum(field.nbytes for field in fields)
        used = sum(a.nbytes for a in self.fixed)
        free, _ = self.cp.cuda.runtime.memGetInfo()
        if total + used > 8 * 1024**3 or total > free * 0.8:
            raise MemoryError("RF位相の電場配列がGPUメモリ設計目標を超えます。位相ビン数を減らしてください。")
        self.phase_fields = {id(field): self.cp.asarray(field) for field in fields}

    def __call__(
        self,
        positions,
        velocities,
        starts,
        charges,
        masses,
        field,
        inverse,
        gradients,
        neighbors,
        face_kind,
        face_patch,
        max_steps,
        max_dt,
        cell_size,
        cell_fraction,
        path_count,
    ):
        cp = self.cp
        n = len(positions)
        npth = min(path_count, n)
        inputs = tuple(
            cp.asarray(np.ascontiguousarray(a)) for a in (positions, velocities, starts, charges, masses)
        )
        # Keep a field on device while all batches of that RF phase use it.
        if id(field) in self.phase_fields:
            self.device_field = self.phase_fields[id(field)]
        elif self.field_key is not field:
            self.device_field = cp.asarray(field)
            self.field_key = field
        out = (
            inputs[0].copy(),
            inputs[1].copy(),
            cp.full(n, 3, dtype=cp.int8),
            cp.full(n, -1, dtype=cp.int64),
            inputs[2].copy(),
            cp.zeros(n),
            cp.zeros((n, 3)),
            cp.zeros(n),
            cp.full((npth, 128, 3), cp.nan),
            cp.zeros(npth, dtype=cp.int64),
            cp.full((npth, 128), cp.nan),
        )
        self.kernel(
            ((n + 127) // 128,),
            (128,),
            (
                *inputs,
                self.device_field,
                *self.fixed,
                np.int32(n),
                np.int32(max_steps),
                np.float64(max_dt),
                np.float64(cell_size),
                np.float64(cell_fraction),
                np.int32(npth),
                *out,
            ),
        )
        return tuple(cp.asnumpy(a) for a in out)
