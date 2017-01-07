/*********
Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********/
#include "Defines.h"
#include "cucommon.cuh"
#include <iostream>

void CUDA_CHECK_ERR(unsigned lineNumber, const char* fileName) {

   cudaError_t err = cudaGetLastError();
   if (err) std::cout << "Error " << err << " on line " << lineNumber << " of " << fileName << ": " << cudaGetErrorString(err) << std::endl;
}

float getElapsed(cudaEvent_t start, cudaEvent_t stop) {
   float elapsed;
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   return elapsed;
}
__device__ int2 convert(int asize, int Qpx, float pin) {

   float frac; float round;
   //TODO add the 1 afterward?
   frac = modf((pin+1)*asize, &round);
   return make_int2(int(round), int(frac*Qpx));
}

__device__ void atomicAddWrap(float* address, float val)
{
#ifdef __NOATOMIC
    *address+=val;
#else
   #if 1 || defined(__CASATOMIC) || __CUDA_ARCH__ < 800
  float old_v, new_v;

  do {
    old_v = *address;
    new_v = old_v + val;
  } while (atomicCAS((unsigned *) address, __float_as_int(old_v), __float_as_int(new_v)) != __float_as_int(old_v));
   #else
   atomicAdd(address, val);
   #endif
#endif
}
__device__ void atomicAddWrap(double* address, double val)
{
#ifdef __NOATOMIC
    *address+=val;
#else
   #if defined(__CASATOMIC) || __CUDA_ARCH__ < 800
   //#if 1
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
   #else
    atomicAdd(address, val);
   #endif
#endif
}

__device__ double make_zero(double2* in) { return (double)0.0;}
__device__ float make_zero(float2* in) { return (float)0.0;}

template <int gcf_dim, class CmplxOutType, class CmplxType>
__global__ void 
__launch_bounds__(256, 8)
grid_kernel(CmplxOutType* out, CmplxType* in, CmplxType* in_vals, size_t npts,
                              size_t img_dim, CmplxType* gcf) {
   
   //TODO remove hard-coded 32
   CmplxType __shared__ inbuff[32];
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   for (int n = 32*blockIdx.x; n<npts; n+= 32*gridDim.x) {
      __syncthreads(); 
      int raw_idx = threadIdx.x+blockDim.x*threadIdx.y;
      if (raw_idx < 32) inbuff[raw_idx]= in[n+raw_idx];
      //if (threadIdx.x<32 && threadIdx.y==blockDim.y-1) invalbuff[threadIdx.x]=in_vals[n+threadIdx.x];
      __syncthreads(); 
      
   for (int q=threadIdx.y;q<32&&n+q<npts;q+=blockDim.y) {
      CmplxType inn = inbuff[q];
      int sub_x = floor(GCF_GRID*(inn.x-floor(inn.x)));
      int sub_y = floor(GCF_GRID*(inn.y-floor(inn.y)));
      int main_x = floor(inn.x); 
      int main_y = floor(inn.y); 
      auto sum_r = make_zero(out);
      auto sum_i = make_zero(out);
      for(int a = -(int)threadIdx.x+gcf_dim/2;a>-gcf_dim/2;a-=blockDim.x)
      for(int b = gcf_dim/2;b>-gcf_dim/2;b--)
      {
         //auto this_img = img[main_x+a+img_dim*(main_y+b)]; 
         //auto r1 = this_img.x;
         //auto i1 = this_img.y;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
         } else {
            //auto this_gcf = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
            //               gcf_dim*b+a]);
            //auto r2 = this_gcf.x;
            //auto i2 = this_gcf.y;
#ifdef __COMPUTE_GCF
            //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
            //double r2 = sin(phase);
            //double i2 = cos(phase);
            float xsquare = (main_x-inn.x+sub_x*1.0/8.0);
            float ysquare = (main_y-inn.y+sub_y*1.0/8.0);
            xsquare *= xsquare;
            ysquare *= ysquare;
            float phase = p1 - p2*sqrt(xsquare + ysquare);
            float r2,i2;
            sincos(phase, &r2, &i2);
#else
            auto r2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].x;
            auto i2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].y;
#endif
            //#pragma unroll
            for (int p=0;p<POLARIZATIONS;p++) {
               auto r1 = in_vals[(n+q)*POLARIZATIONS+p].x;
               auto i1 = in_vals[(n+q)*POLARIZATIONS+p].y;
#ifdef DEBUG1
               atomicAddWrap(&out[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE].x, 1.0);
               atomicAddWrap(&out[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE].y, n+q);
#else
               atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)+p*img_dim*img_dim].x, r1*r2 - i1*i2); 
               atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)+p*img_dim*img_dim].y, r1*i2 + r2*i1); 
               //out[main_x+a+img_dim*(main_y+b)].x += r1*r2 - i1*i2; 
               //out[main_x+a+img_dim*(main_y+b)].y += r1*i2 + r2*i1; 
#endif
            } //p
         }
      } //b

   } //q
   } //n
}
template <int gcf_dim, class CmplxType>
__global__ void 
//__launch_bounds__(256, 6)
grid_kernel_basic(CmplxType* out, CmplxType* in, CmplxType* in_vals, size_t npts,
                              size_t img_dim, CmplxType* gcf) {
   
   //TODO remove hard-coded 32
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   for (int n = 32*blockIdx.x; n<npts; n+= 32*gridDim.x) {
   for (int q=threadIdx.y;q<32;q+=blockDim.y) {
      CmplxType inn = in[n+q];
      int sub_x = floor(GCF_GRID*(inn.x-floor(inn.x)));
      int sub_y = floor(GCF_GRID*(inn.y-floor(inn.y)));
      int main_x = floor(inn.x); 
      int main_y = floor(inn.y); 
      auto sum_r = make_zero(out);
      auto sum_i = make_zero(out);
      for(int a = -(int)threadIdx.x+gcf_dim/2;a>-gcf_dim/2;a-=blockDim.x)
      for(int b = gcf_dim/2;b>-gcf_dim/2;b--)
      {
         //auto this_img = img[main_x+a+img_dim*(main_y+b)]; 
         //auto r1 = this_img.x;
         //auto i1 = this_img.y;
         auto r1 = in_vals[n+q].x;
         auto i1 = in_vals[n+q].y;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
            r1=i1=0.0;
         } else {
            //auto this_gcf = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
            //               gcf_dim*b+a]);
            //auto r2 = this_gcf.x;
            //auto i2 = this_gcf.y;
#ifdef __COMPUTE_GCF
            //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
            //double r2 = sin(phase);
            //double i2 = cos(phase);
            float xsquare = (main_x-inn.x+sub_x*1.0/8.0);
            float ysquare = (main_y-inn.y+sub_y*1.0/8.0);
            xsquare *= xsquare;
            ysquare *= ysquare;
            float phase = p1 - p2*sqrt(xsquare + ysquare);
            float r2,i2;
            sincos(phase, &r2, &i2);
#else
            auto r2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].x;
            auto i2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].y;
#endif
#ifdef DEBUG1
            atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)].x, n+q);
            atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)].y, gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x)+gcf_dim*b+a].y);
#else
            atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)].x, r1*r2 - i1*i2); 
            atomicAddWrap(&out[main_x+a+img_dim*(main_y+b)].y, r1*i2 + r2*i1); 
#endif
         }
      }

#if 0
      for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
         sum_r += __shfl_down(sum_r,s);
         sum_i += __shfl_down(sum_i,s);
      }
      CmplxType tmp;
      tmp.x = sum_r;
      tmp.y = sum_i;
      if (threadIdx.x == 0) {
         out[n+q] = tmp;
      }
#endif
   }
   }
}
template <int gcf_dim, class CmplxType, class CmplxOutType>
__global__ void 
//__launch_bounds__(256, 6)
grid_kernel_small_gcf(CmplxOutType* out, CmplxType* in, CmplxType* in_vals, size_t npts,
                              size_t img_dim, CmplxType* gcf) {
   
   //TODO remove hard-coded 32
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   for (int n = 32*blockIdx.x; n<npts; n+= 32*gridDim.x) {
   for (int q=threadIdx.y;q<32;q+=blockDim.y) {
      CmplxType inn = in[n+q];
      int sub_x = floor(GCF_GRID*(inn.x-floor(inn.x)));
      int sub_y = floor(GCF_GRID*(inn.y-floor(inn.y)));
      int main_x = floor(inn.x); 
      int main_y = floor(inn.y); 
      auto sum_r = make_zero(out);
      auto sum_i = make_zero(out);
      int a = -gcf_dim/2 + (int)threadIdx.x%gcf_dim;
      for(int b = -gcf_dim/2+(int)threadIdx.x/gcf_dim;b<gcf_dim/2;b+=blockDim.x/gcf_dim)
      {
         //auto this_img = img[main_x+a+img_dim*(main_y+b)]; 
         //auto r1 = this_img.x;
         //auto i1 = this_img.y;
         auto r1 = in_vals[n+q].x;
         auto i1 = in_vals[n+q].y;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
            r1=i1=0.0;
         }
         //auto this_gcf = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
         //               gcf_dim*b+a]);
         //auto r2 = this_gcf.x;
         //auto i2 = this_gcf.y;
#ifdef __COMPUTE_GCF
         //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
         //double r2 = sin(phase);
         //double i2 = cos(phase);
         float xsquare = (main_x-inn.x+sub_x*1.0/GCF_GRID);
         float ysquare = (main_y-inn.y+sub_y*1.0/GCF_GRID);
         xsquare *= xsquare;
         ysquare *= ysquare;
         float phase = p1 - p2*sqrt(xsquare + ysquare);
         float r2,i2;
         sincos(phase, &r2, &i2);
#else
         auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                        gcf_dim*b+a].x);
         auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                        gcf_dim*b+a].y);
#endif
         out[main_x+a+img_dim*(main_y+b)].x += r1*r2 - i1*i2; 
         out[main_x+a+img_dim*(main_y+b)].y += r1*i2 + r2*i1;
      }

#if 0
      for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
         sum_r += __shfl_down(sum_r,s);
         sum_i += __shfl_down(sum_i,s);
      }
      CmplxType tmp;
      tmp.x = sum_r;
      tmp.y = sum_i;
      if (threadIdx.x == 0) {
         out[n+q] = tmp;
      }
#endif
   }
   }
}
__device__ void warp_reduce(double &in, int sz = 16) {
   if (16<sz) sz=16;
   for(int s = sz; s>0;s/=2) {
      in += __shfl_down(in,s);
   }
}
__device__ void warp_reduce(float &in, int sz = 16) {
   if (16<sz) sz=16;
   for(int s = sz; s>0;s/=2) {
      in += __shfl_down(in,s);
   }
}
__device__ void warp_reduce2(float &in, int sz = 32) {
   if (32<sz) sz=32;
   for(int s=1; s<sz; s*=2) {
      in += __shfl_down(in,s);
   } 
}
__device__ void warp_reduce2(double &in, int sz = 32) {
   if (32<sz) sz=32;
   for(int s=1; s<sz; s*=2) {
      in += __shfl_down(in,s);
   } 
}
template <class CmplxType>
__global__ void vis2ints(CmplxType *vis_in, int2* vis_out, int npts) {
   for (int q=threadIdx.x+blockIdx.x*blockDim.x; 
        q<npts; 
        q+=gridDim.x*blockDim.x) {
      CmplxType inn = vis_in[q];
      int main_y = floor(inn.y); 
      int sub_y = floor(GCF_GRID*(inn.y-main_y));
      int main_x = floor(inn.x); 
      int sub_x = floor(GCF_GRID*(inn.x-main_x));
      vis_out[q].x = main_x*GCF_GRID+sub_x;
      vis_out[q].y = main_y*GCF_GRID+sub_y;
   }
}
//Make sure visibilities are sorted by  main_x/blocksize then main_y/blocksize
// blockgrid should be (img_dim+blocksize-1)/blocksize
__global__ void set_bookmarks(int2* vis_in, int npts, int blocksize, int blockgrid, int* bookmarks) {
   for (int q=threadIdx.x+blockIdx.x*blockDim.x;q<=npts;q+=gridDim.x*blockDim.x) {
      int main_x, main_y, main_x_last, main_y_last;
      int2 this_vis = vis_in[q];
      if (0==q) {
         main_y_last=0;
         main_x_last=-1;
      } else {
         int2 last_vis = vis_in[q-1];
         main_x_last = last_vis.x/GCF_GRID/blocksize;
         main_y_last = last_vis.y/GCF_GRID/blocksize;
      }
      main_x = this_vis.x/GCF_GRID/blocksize;
      main_y = this_vis.y/GCF_GRID/blocksize;
      if (npts==q) main_x = main_y = blockgrid;
      if (main_x != main_x_last || main_y != main_y_last)  { 
         for (int z=main_y_last*blockgrid+main_x_last+1;
                  z<=main_y*blockgrid+main_x; z++) {
            bookmarks[z] = q;
         }
      }
   }
}
template <int gcf_dim, class CmplxType>
__global__ void 
#if POLARIZATIONS == 1
__launch_bounds__(1024, 2)
#else
__launch_bounds__(GCF_DIM*GCF_DIM/4/4/GCF_STRIPES/PTS, 12)
#endif
grid_kernel_gather(CmplxType* out, int2* in, CmplxType* in_vals, size_t npts, 
                              int img_dim, CmplxType* gcf, int* bookmarks, int yoff) {
   
   int2 __shared__ inbuff[32];
   CmplxType __shared__ invalbuff[POLARIZATIONS][32+32/POLARIZATIONS];
   const int bm_dim = (img_dim+gcf_dim-1)/gcf_dim*2;
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   int left = blockIdx.x*blockDim.x;
   int top = blockIdx.y*blockDim.y*PTS*GCF_STRIPES;
   int this_x = left+threadIdx.x;
   int this_y = top+threadIdx.y+yoff;
   //if (this_x >= img_dim) return;
   //if (this_y >= img_dim) return;
   CmplxType sum[POLARIZATIONS][PTS]; 
   for (int p=0;p<PTS;p++) {
      //#pragma unroll
      for (int pz=0;pz<POLARIZATIONS;pz++) {
         sum[pz][p] = out[this_x + this_y*img_dim+p*blockDim.y*img_dim+pz*img_dim*img_dim];
      }
   }
   int half_gcf = gcf_dim/2;
   
   int bm_x = left/half_gcf-1;
   int bm_y = top/half_gcf-1;
   for (int y=bm_y<0?0:bm_y;(y<bm_y+2+(blockDim.y+half_gcf-1)/half_gcf)&&(y<(img_dim+half_gcf-1)/half_gcf);y++) {
   for (int x=bm_x<0?0:bm_x;(x<bm_x+2+(blockDim.x+half_gcf-1)/half_gcf)&&(x<(img_dim+half_gcf-1)/half_gcf);x++) {
   int bm_start = bookmarks[y*bm_dim+x];
   int bm_end = bookmarks[y*bm_dim+x+1];
   for (int n=bm_start; n<= bm_end; n+=32) {
      __syncthreads(); 
      int raw_idx = threadIdx.x+blockDim.x*threadIdx.y;
      if (raw_idx < 32) inbuff[raw_idx]= in[n+raw_idx];
      else {
         raw_idx -= 32;
         if (raw_idx < 32*POLARIZATIONS) invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS]= in_vals[n*POLARIZATIONS+raw_idx];
      }
      //if (threadIdx.x<32 && threadIdx.y==blockDim.y-1) invalbuff[threadIdx.x]=in_vals[n+threadIdx.x];
      __syncthreads(); 
      
   for (int q = 0; q<32 && n+q < bm_end; q++) {
      int2 inn = inbuff[q];
      for (int p = 0; p < PTS; p++) {
      int main_y = inn.y/GCF_GRID;
      if (this_y + blockDim.y*p >= img_dim) continue;
      int b = this_y + blockDim.y*p - main_y;
      if (b > half_gcf || b <= - half_gcf) continue;
      int main_x = inn.x/GCF_GRID;
      int a = this_x - main_x;
      if (a > half_gcf || a <= - half_gcf) continue;
#ifdef __COMPUTE_GCF
      //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
      //double r2 = sin(phase);
      //double i2 = cos(phase);
      int sub_x = inn.x%GCF_GRID;
      int sub_y = inn.y%GCF_GRID;
      float xsquare = (main_x-inn.x+sub_x*1.0/GCF_GRID);
      float ysquare = (main_y-inn.y+sub_y*1.0/GCF_GRID);
      xsquare *= xsquare;
      ysquare *= ysquare;
      float phase = p1 - p2*sqrt(xsquare + ysquare);
      float r2,i2;
      sincos(phase, &r2, &i2);
#else
      int sub_x = inn.x%GCF_GRID;
      int sub_y = inn.y%GCF_GRID;
      CmplxType ctmp = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                     gcf_dim*b+a]);
      auto r2 = ctmp.x;
      auto i2 = ctmp.y;
      //auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
      //               gcf_dim*b+a].x);
      //auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
       //              gcf_dim*b+a].y);
#endif
      //#pragma unroll
      for (int pz=0;pz<POLARIZATIONS;pz++) {
         CmplxType r1 = invalbuff[pz][q];
         //CmplxType r1 = in_vals[p+POLARIZATIONS*(n+q)];
#ifdef DEBUG1
         sum[pz][p].x += 1.0;
         sum[pz][p].y += n+q;
#else
         sum[pz][p].x += r1.x*r2 - r1.y*i2; 
         sum[pz][p].y += r1.x*i2 + r2*r1.y;
#endif
      } //pz

   } //p
   } //q
   } //n
   } //x
   } //y
   for (int p=0;p<PTS;p++) {
      if (this_y + blockDim.y*p >= img_dim) continue;
      if (this_x >= img_dim) continue;
      //#pragma unroll
      for (int pz=0;pz<POLARIZATIONS;pz++) {
         out[this_x + img_dim * (this_y+blockDim.y*p) + pz*img_dim*img_dim] = sum[pz][p];
      }
   }
}
template <int gcf_dim, class CmplxOutType, class CmplxType>
__global__ void 
__launch_bounds__(GCF_DIM*BLOCK_Y, 4)
grid_kernel_window(CmplxOutType* out, int2* in, CmplxType* in_vals, int* in_gcfinx, size_t npts, 
                              int img_dim, CmplxType* gcf) {
   
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   int2 __shared__ inbuff[32];
   CmplxType __shared__ invalbuff[POLARIZATIONS][32+32/POLARIZATIONS];
   int __shared__ ingcfinxbuff[POLARIZATIONS][32+32/POLARIZATIONS];
   CmplxOutType sum[POLARIZATIONS];
   CmplxType r1;
   int half_gcf = gcf_dim/2;
   int local_npt = (npts+gridDim.x-1)/gridDim.x; //number of points assigned to this block
   //TODO find a way to switch this on CmplxOutType
   //TODO What about odd values of img_dim?
   double* out1 = (double*)out;
   double* out2 = (double*)(out+POLARIZATIONS*img_dim*img_dim/2);
   in += local_npt*blockIdx.x;
   in_vals += local_npt*blockIdx.x*POLARIZATIONS;
   //TODO What about odd values of npts?
   //double* in_vals1 = (double*)in_vals;
   //double* in_vals2 = (double*)(in_vals+npts/2);
   //double* gcf1 = (double*)gcf;
   //double* gcf2 = (double*)(gcf+gcf_dim*gcf_dim*GCF_GRID*GCF_GRID/2);
   int last_idx = -INT_MAX;
   size_t gcf_y = threadIdx.y + blockIdx.y*blockDim.y;
   if (blockIdx.x==gridDim.x-1) local_npt = npts-local_npt*blockIdx.x;
   
   int raw_idx = threadIdx.x+blockDim.x*threadIdx.y;
   int tidx = threadIdx.x;
   for (int n=0; n<local_npt; n+=32) {

      __syncthreads(); 
      if (raw_idx < 32) inbuff[raw_idx]= in[n+raw_idx];
      else {
	 //TODO What if gridDim < 32+64*POLARIZATIONS
         if (raw_idx - 32 < 32*POLARIZATIONS) 
	 {
            raw_idx -= 32;
	    #if 0
	    //Coalescing the input reads has no discernable performance impact
	    invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS].x= in_vals1[n*POLARIZATIONS+raw_idx];
	    invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS].y= in_vals2[n*POLARIZATIONS+raw_idx];
	    #else
	    invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS].x= in_vals[n*POLARIZATIONS+raw_idx].x;
	    invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS].y= in_vals[n*POLARIZATIONS+raw_idx].y;
	    //invalbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS]= in_vals[n*POLARIZATIONS+raw_idx];
	    #endif
	 } else if (raw_idx < 2*32*POLARIZATIONS)
         {
            raw_idx -= 32*POLARIZATIONS;
	    ingcfinxbuff[raw_idx%POLARIZATIONS][raw_idx/POLARIZATIONS] = in_gcfinx[n*POLARIZATIONS+raw_idx];
         }
      }
      
      //shm[threadIdx.x][threadIdx.y].x = 0.00;
      //shm[threadIdx.x][threadIdx.y].y = 0.00;
      __syncthreads(); 
      if (gcf_y >= GCF_DIM) continue;
   for (int q = 0; q<32 && n+q < local_npt; q++) {
      int2 inn = inbuff[q];
      int main_y = inn.y/GCF_GRID;
      int main_x = inn.x/GCF_GRID;
      int this_x = gcf_dim*((main_x+half_gcf-tidx)/gcf_dim)+tidx;
      int this_y;
      this_y = gcf_dim*((main_y+half_gcf-gcf_y)/gcf_dim)+gcf_y;
      if (main_x+half_gcf < tidx || this_x >= img_dim ||
          main_y+half_gcf < gcf_y || this_y >= img_dim) {
          //TODO pad instead?
      } else {
          int this_idx = this_x + img_dim * this_y;
          __prof_trigger(0);
          if (last_idx != this_idx) {
             __prof_trigger(1);
             if (last_idx != -INT_MAX) {
                //#pragma unrol
                for (int pz=0;pz<POLARIZATIONS;pz++) {
		   __prof_trigger(2);
		   #if 1
		   //Coalescing atomic writes is a 25% performance boost
                   atomicAddWrap(&out1[last_idx+pz*img_dim*img_dim], sum[pz].x);
                   atomicAddWrap(&out2[last_idx+pz*img_dim*img_dim], sum[pz].y);
		   #else
                   atomicAddWrap(&out[last_idx+pz*img_dim*img_dim].x, sum[pz].x);
                   atomicAddWrap(&out[last_idx+pz*img_dim*img_dim].y, sum[pz].y);
		   #endif
                }
             }
             for (int pz=0;pz<POLARIZATIONS;pz++) sum[pz].x = sum[pz].y = 0.0;
             last_idx = this_idx;
          }
#ifdef __COMPUTE_GCF
          //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
          //double r2 = sin(phase);
          //double i2 = cos(phase);
          int sub_x = inn.x%GCF_GRID;
          int sub_y = inn.y%GCF_GRID;
          float xsquare = (main_x-inn.x+sub_x*1.0/GCF_GRID);
          float ysquare = (main_y-inn.y+sub_y*1.0/GCF_GRID);
          xsquare *= xsquare;
          ysquare *= ysquare;
          float phase = p1 - p2*sqrt(xsquare + ysquare);
          float r2,i2;
          sincos(phase, &r2, &i2);
#else
          int sub_x = inn.x%GCF_GRID;
          int sub_y = inn.y%GCF_GRID;
          int b = this_y - main_y;
          int a = this_x - main_x;
	  #if 0
	  //This is coalesced read is actually a perf hit
          auto r2 = __ldg(&gcf1[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a]);
          auto i2 = __ldg(&gcf2[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a]);
          //auto r2 = gcf1[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
          //               gcf_dim*b+a];
          //auto i2 = gcf2[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
          //               gcf_dim*b+a];
	  #else
          #if 1 
          //TODO what if polarizations have different GCFs?
	  auto z2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y + sub_x + GCF_GRID*GCF_GRID*ingcfinxbuff[0][q]) + 
                         gcf_dim*b+a]);
          auto r2 = z2.x;
          auto i2 = z2.y;
	  #else
          auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a].x);
          auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a].y);
	  #endif
          //auto r2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
          //               gcf_dim*b+a].x;
          //auto i2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
          //               gcf_dim*b+a].y;
	  #endif
#endif
          //#pragma unroll
          for (int pz=0;pz<POLARIZATIONS;pz++) {
             r1 = invalbuff[pz][q];
             //r1 = in_vals[POLARIZATIONS*(n+q)+pz];
#ifdef DEBUG1
             sum[pz].x += 1.0;
             sum[pz].y += n+q + blockIdx.x*((npts+gridDim.x-1)/gridDim.x); 
#else
             //sum[pz].x += r1.x*r2 - r1.y*i2; 
             sum[pz].x += r1.x*r2; 
             sum[pz].x -= r1.y*i2; 
             //sum[pz].y += r1.x*i2 + r2*r1.y;
             sum[pz].y += r1.x*i2; 
             sum[pz].y += r2*r1.y;
#endif
          }
      }

     //reduce in two directions
      //WARNING: Adjustments must be made if blockDim.y and blockDim.x are no
      //         powers of 2 
      //Reduce using shuffle first
   } //q
   } //n
   if (last_idx != -INT_MAX) {
      //#pragma unroll
      for(int pz=0;pz<POLARIZATIONS;pz++) {
         atomicAddWrap(&out1[last_idx+pz*img_dim*img_dim], sum[pz].x);
         atomicAddWrap(&out2[last_idx+pz*img_dim*img_dim], sum[pz].y);
      }
   }
}

template <class CmplxOutType, class CmplxType>
void gridGPU(CmplxOutType* out, CmplxType* in, CmplxType* in_vals, int* in_gcfinx, size_t npts, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim) {
//grid on the GPU
//  out (out) - the output image
//  in  (in)  - the input locations
//  in_vals (in) - input values
//  in_gcfinx (in) - index of the GCF for each visibility
//  npts (in) - number of locations
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   CmplxOutType *d_out;
   CmplxType *d_in, *d_in_vals, *d_gcf;
   int* d_in_gcfinx;

   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);

   CUDA_CHECK_ERR(__LINE__,__FILE__);
#ifdef __MANAGED
   d_gcf = gcf;
   std::cout << "d_out = out" << std::endl;
   d_out = out;
   d_in = in;
   d_in_vals = in_vals;
   d_in_gcfinx = in_gcfinx;
                                                                 sizeof(CmplxType) << std::endl;
#else
   //img is padded to avoid overruns. Subtract to find the real head

   //Pin CPU memory
   cudaHostRegister(out, sizeof(CmplxOutType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS, cudaHostRegisterMapped);
   cudaHostRegister(gcf, sizeof(CmplxType)*GCF_GRID*GCF_GRID*gcf_dim*gcf_dim, cudaHostRegisterMapped);
   cudaHostRegister(in, sizeof(CmplxType)*npts, cudaHostRegisterMapped);
   cudaHostRegister(in_vals, sizeof(CmplxType)*npts*POLARIZATIONS, cudaHostRegisterMapped);

   //Allocate GPU memory
   cudaMalloc(&d_out, sizeof(CmplxOutType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS);
   cudaMalloc(&d_gcf, sizeof(CmplxType)*GCF_GRID*GCF_GRID*gcf_dim*gcf_dim*NGCF);
   cudaMalloc(&d_in, sizeof(CmplxType)*npts);
   cudaMalloc(&d_in_vals, sizeof(CmplxType)*npts*POLARIZATIONS);
   cudaMalloc(&d_in_gcfinx, sizeof(int)*npts*POLARIZATIONS);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Copy in img, gcf and out
   cudaEventRecord(start);
   cudaMemcpy(d_gcf, gcf, sizeof(CmplxType)*64*gcf_dim*gcf_dim, 
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in, in, sizeof(CmplxType)*npts,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in_vals, in_vals, sizeof(CmplxType)*npts*POLARIZATIONS,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in_gcfinx, in_gcfinx, sizeof(int)*npts*POLARIZATIONS,
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   std::cout << "memcpy time: " << getElapsed(start, stop) << " ms." << std::endl;

   //move d_img and d_gcf to remove padding
#endif
   //offset gcf to point to the middle of the first GCF for cleaner code later
   d_gcf += gcf_dim*(gcf_dim-1)/2-1;
   CmplxOutType* d_out_unpad = d_out + img_dim*gcf_dim+gcf_dim;

#ifdef __GATHER
   int2* in_ints;
   int* bookmarks;
   cudaMalloc(&in_ints, sizeof(int2)*npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   printf("cudaMallocated in_ints : %p\n", in_ints);
   cudaMalloc(&bookmarks, sizeof(int)*((img_dim/gcf_dim)*(img_dim/gcf_dim)*4+1));
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   vis2ints<<<4,256>>>(d_in, in_ints, npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   set_bookmarks<<<4,256>>>(in_ints, npts, gcf_dim/2, (img_dim+gcf_dim/2-1)/(gcf_dim/2), 
                               bookmarks);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   int2* h_ints = (int2*)malloc(sizeof(int2)*npts);
   printf("allocated h_ints : %p\n", h_ints);
   printf("copy %d bytes from %p to %p\n", sizeof(int2)*npts, in_ints,
   h_ints);
   cudaMemcpy(h_ints, in_ints, sizeof(int2)*npts, cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   
   
   cudaMemset(d_out, 0, sizeof(CmplxOutType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS);
   cudaEventRecord(start);
   for (int stripe=0;stripe<GCF_STRIPES;stripe++)
      //TODO add gcfinx to gather
      grid_kernel_gather<GCF_DIM>
            <<<dim3((img_dim+gcf_dim/4-1)/(gcf_dim/4), (img_dim+gcf_dim/4-1)/(gcf_dim/4)),
               dim3(gcf_dim/4, gcf_dim/4/PTS/GCF_STRIPES)>>> // <-- Must not truncate here
   //         <<<dim3((img_dim+gcf_dim-1)/(gcf_dim), (img_dim+gcf_dim-1)/(gcf_dim)),
   //            dim3(gcf_dim, gcf_dim)>>>
                             (d_out_unpad,in_ints,d_in_vals,npts,img_dim,d_gcf,bookmarks,stripe*gcf_dim/4/GCF_STRIPES); 
   //std::cout<< "grid_kernel_gather<<<(" << (img_dim+gcf_dim-1)/gcf_dim << ", " << (img_dim+gcf_dim/4-1)/(gcf_dim/4) << "), (" << gcf_dim << ", " << gcf_dim/4 << ")>>>()" << std::endl; 
   CUDA_CHECK_ERR(__LINE__,__FILE__);
#else
#ifdef __MOVING_WINDOW
   int2* in_ints;
   cudaMalloc(&in_ints, sizeof(int2)*npts);
   vis2ints<<<4,256>>>(d_in, in_ints, npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemset(d_out, 0, sizeof(CmplxOutType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS);
   cudaEventRecord(start);
   grid_kernel_window<GCF_DIM>
               <<<dim3((npts+31)/32,(GCF_DIM+BLOCK_Y-1)/BLOCK_Y),dim3(GCF_DIM,BLOCK_Y)>>>(d_out_unpad,in_ints,d_in_vals,d_in_gcfinx,npts,img_dim,d_gcf); 
   //vis2ints<<<dim3(npts/64,8),dim3(GCF_DIM,GCF_DIM/8)>>>(d_in, in_ints, npts);
#else
   cudaEventRecord(start);
   //TODO add gcfinx to scatter
   cudaMemset(d_out, 0, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS);
   if (GCF_DIM < 32) {
      grid_kernel_small_gcf<GCF_DIM>
               <<<npts/32,dim3(32,32)>>>(d_out_unpad,d_in,d_in_vals,npts,img_dim,d_gcf); 
   } else {
      grid_kernel<GCF_DIM>
               <<<npts/32,dim3(32,8)>>>(d_out_unpad,d_in,d_in_vals,npts,img_dim,d_gcf); 
   }
#endif
#endif
   float kernel_time = getElapsed(start,stop);
   std::cout << "Processed " << npts << " complex points in " << kernel_time << " ms." << std::endl;
   std::cout << npts / 1000000.0 / kernel_time * gcf_dim * gcf_dim * 8 * POLARIZATIONS << " Gflops" << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

#ifdef __MANAGED
   cudaDeviceSynchronize();
#else
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(out, d_out, 
              sizeof(CmplxOutType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*POLARIZATIONS, 
              cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Unpin CPU memory
   cudaHostUnregister(gcf);
   cudaHostUnregister(out);
   cudaHostUnregister(in);
   cudaHostUnregister(in_vals);

   //Restore d_img and d_gcf for deallocation
   d_gcf -= gcf_dim*(gcf_dim-1)/2-1;
   cudaFree(d_out);
#ifdef __GATHER
   cudaFree(in_ints);
   cudaFree(bookmarks);
#endif
#endif
   cudaEventDestroy(start); cudaEventDestroy(stop);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
}
template void gridGPU<double2, double2>(double2* out, double2* in, double2* in_vals, int* in_gcfinx, 
                                 size_t npts, size_t img_dim, double2 *gcf, size_t gcf_dim); 
template void gridGPU<double2, float2>(double2* out, float2* in, float2* in_vals, int* in_gcfinx, 
                                size_t npts, size_t img_dim, float2 *gcf, size_t gcf_dim); 
template void gridGPU<float2, float2>(float2* out, float2* in, float2* in_vals, int* in_gcfinx, 
                                size_t npts, size_t img_dim, float2 *gcf, size_t gcf_dim); 
