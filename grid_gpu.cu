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

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double make_zero(double2* in) { return (double)0.0;}
__device__ float make_zero(float2* in) { return (float)0.0;}

template <int gcf_dim, class CmplxType>
__global__ void 
//__launch_bounds__(256, 6)
grid_kernel(CmplxType* out, CmplxType* in, CmplxType* in_vals, size_t npts,
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
      int sub_x = floorf(GCF_GRID*(inn.x-floorf(inn.x)));
      int sub_y = floorf(GCF_GRID*(inn.y-floorf(inn.y)));
      int main_x = floorf(inn.x); 
      int main_y = floorf(inn.y); 
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
            float xsquare = (main_x-inn.x);
            float ysquare = (main_x-inn.x);
            xsquare *= xsquare;
            ysquare *= ysquare;
            float phase = p1 - p2*sqrt(xsquare + ysquare);
            float r2,i2;
            sincosf(phase, &r2, &i2);
#else
            auto r2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].x;
            auto i2 = gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                           gcf_dim*b+a].y;
#endif
#ifdef DEBUG1
            atomicAdd(&out[main_x+a+img_dim*(main_y+b)].x, n+q);
            atomicAdd(&out[main_x+a+img_dim*(main_y+b)].y, gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x)+gcf_dim*b+a].y);
#else
            atomicAdd(&out[main_x+a+img_dim*(main_y+b)].x, r1*r2 - i1*i2); 
            atomicAdd(&out[main_x+a+img_dim*(main_y+b)].y, r1*i2 + r2*i1); 
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
template <int gcf_dim, class CmplxType>
__global__ void 
//__launch_bounds__(256, 6)
grid_kernel_small_gcf(CmplxType* out, CmplxType* in, CmplxType* in_vals, size_t npts,
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
      int sub_x = floorf(GCF_GRID*(inn.x-floorf(inn.x)));
      int sub_y = floorf(GCF_GRID*(inn.y-floorf(inn.y)));
      int main_x = floorf(inn.x); 
      int main_y = floorf(inn.y); 
      auto sum_r = make_zero(out);
      auto sum_i = make_zero(out);
      int a = -gcf_dim/2 + threadIdx.x%gcf_dim;
      for(int b = -gcf_dim/2+threadIdx.x/gcf_dim;b<gcf_dim/2;b+=blockDim.x/gcf_dim)
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
         float xsquare = (main_x-inn.x);
         float ysquare = (main_x-inn.x);
         xsquare *= xsquare;
         ysquare *= ysquare;
         float phase = p1 - p2*sqrt(xsquare + ysquare);
         float r2,i2;
         sincosf(phase, &r2, &i2);
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
      int main_y = floorf(inn.y); 
      int sub_y = floorf(GCF_GRID*(inn.y-main_y));
      int main_x = floorf(inn.x); 
      int sub_x = floorf(GCF_GRID*(inn.x-main_x));
      vis_out[q].x = main_x*GCF_GRID+sub_x;
      vis_out[q].y = main_y*GCF_GRID+sub_y;
   }
}
//Make sure visibilities are sorted by  main_x/blocksize then main_y/blocksize
// blockgrid should be (img_dim+blocksize-1)/blocksize
__global__ void set_bookmarks(int2* vis_in, int npts, int blocksize, int blockgrid, int* bookmarks) {
   for (int q=threadIdx.x+blockIdx.x*blockDim.x;q<=npts;q+=gridDim.x*blockDim.x) {
      int2 this_vis = vis_in[q];
      int2 last_vis = vis_in[q-1];
      int main_x = this_vis.x/GCF_GRID/blocksize;
      int main_x_last = last_vis.x/GCF_GRID/blocksize;
      int main_y = this_vis.y/GCF_GRID/blocksize;
      int main_y_last = last_vis.y/GCF_GRID/blocksize;
      if (0==q) {
         main_y_last=0;
         main_x_last=-1;
      }
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
__launch_bounds__(1024, 2)
grid_kernel_gather(CmplxType* out, int2* in, CmplxType* in_vals, size_t npts, 
                              int img_dim, CmplxType* gcf, int* bookmarks) {
   
   int2 __shared__ inbuff[32];
   CmplxType __shared__ invalbuff[32];
   const int bm_dim = (img_dim+gcf_dim-1)/gcf_dim*2;
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   int left = blockIdx.x*blockDim.x;
   int top = blockIdx.y*blockDim.y*PTS;
   int this_x = left+threadIdx.x;
   int this_y = top+threadIdx.y;
   //if (this_x >= img_dim) return;
   //if (this_y >= img_dim) return;
   //auto r1 = img[this_x + img_dim * this_y].x;
   //auto i1 = img[this_x + img_dim * this_y].y;
   //TODO use sum like the complex number it is (no more make_zero nonsense)
   CmplxType sum[PTS]; 
   for (int p=0;p<PTS;p++) {sum[p].x = sum[p].y = 0.0;}
   int half_gcf = gcf_dim/2;
   
   int bm_x = left/half_gcf-1;
   int bm_y = top/half_gcf-1;
   for (int y=bm_y<0?0:bm_y;(y<bm_y+2+(blockDim.y+half_gcf-1)/half_gcf)&&(y<(img_dim+half_gcf-1)/half_gcf);y++) {
   for (int x=bm_x<0?0:bm_x;(x<bm_x+2+(blockDim.x+half_gcf-1)/half_gcf)&&(x<(img_dim+half_gcf-1)/half_gcf);x++) {
   int bm_start = bookmarks[y*bm_dim+x];
   int bm_end = bookmarks[y*bm_dim+x+1];
   for (int n=bm_start; n<= bm_end; n+=32) {
      //TODO make warp-synchronous?
      __syncthreads(); 
      int raw_idx = threadIdx.x+blockDim.x*threadIdx.y;
      if (raw_idx < 32) inbuff[raw_idx]= in[n+raw_idx];
      else if (raw_idx < 64) invalbuff[raw_idx-32]= in_vals[n+raw_idx-32];
      //if (threadIdx.x<32 && threadIdx.y==blockDim.y-1) invalbuff[threadIdx.x]=in_vals[n+threadIdx.x];
      __syncthreads(); 
      
   for (int q = 0; q<32 && n+q < bm_end; q++) {
      int2 inn = inbuff[q];
      CmplxType in_valn = invalbuff[q];
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
      float xsquare = (main_x-inn.x);
      float ysquare = (main_x-inn.x);
      xsquare *= xsquare;
      ysquare *= ysquare;
      float phase = p1 - p2*sqrt(xsquare + ysquare);
      float r2,i2;
      sincosf(phase, &r2, &i2);
#else
      int sub_x = inn.x%GCF_GRID;
      int sub_y = inn.y%GCF_GRID;
      auto r1 = in_valn.x;
      auto i1 = in_valn.y;
      auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                     gcf_dim*b+a].x);
      auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                     gcf_dim*b+a].y);
#endif
#ifdef DEBUG1
      sum[p].x += 1.0;
      sum[p].y += n+q;
#else
      sum[p].x += r1*r2 - i1*i2; 
      sum[p].y += r1*i2 + r2*i1;
#endif
      //}

   } //p
   } //q
   } //n
   } //x
   } //y
   for (int p=0;p<PTS;p++) {
      if (this_y + blockDim.y*p >= img_dim) continue;
      if (this_x >= img_dim) continue;
      out[this_x + img_dim * (this_y+blockDim.y*p)] = sum[p];
   }
}
#if 0
template <int gcf_dim, class CmplxType>
__global__ void 
__launch_bounds__(1024, 1)
grid_kernel_window(CmplxType* out, int2* in, CmplxType* in_vals, size_t npts, 
                              int img_dim, CmplxType* gcf) {
   
#ifdef __COMPUTE_GCF
   double T = gcf[0].x;
   double w = gcf[0].y;
   float p1 = 2*3.1415926*w;
   float p2 = p1*T;
#endif
   CmplxType __shared__ shm[BLOCK_Y][gcf_dim];
   int2 __shared__ inbuff[32];
   auto sum_r = make_zero(img);
   auto sum_i = make_zero(img);
   auto r1 = sum_r;
   auto i1 = sum_r;
   int half_gcf = gcf_dim/2;
   in += npts/gridDim.x*blockIdx.x;
   out += npts/gridDim.x*blockIdx.x;
   int last_idx = -INT_MAX;
   size_t gcf_y = threadIdx.y + blockIdx.y*blockDim.y;
   int end_pt = npts/gridDim.x;
   if (blockIdx.x==gridDim.x-1) end_pt = npts-npts/gridDim.x*blockIdx.x;
   
   for (int n=0; n<end_pt; n+=32) {

      if (threadIdx.x<32 && threadIdx.y==0) inbuff[threadIdx.x]=in[n+threadIdx.x];
      
      //shm[threadIdx.x][threadIdx.y].x = 0.00;
      //shm[threadIdx.x][threadIdx.y].y = 0.00;
      __syncthreads(); 
   for (int q = 0; q<32 && n+q < end_pt; q++) {
      int2 inn = inbuff[q];
      int main_y = inn.y/GCF_GRID;
      int main_x = inn.x/GCF_GRID;
      int this_x = gcf_dim*((main_x+half_gcf-threadIdx.x-1)/gcf_dim)+threadIdx.x;
      int this_y;
      this_y = gcf_dim*((main_y+half_gcf-gcf_y-1)/gcf_dim)+gcf_y;
      if (this_x < 0 || this_x >= img_dim ||
          this_y < 0 || this_y >= img_dim) {
          //TODO pad instead?
          sum_r = 0.0;
          sum_i = 0.0;
      } else {
      //TODO is this the same as last time?
          int this_idx = this_x + img_dim * this_y;
          prof_trigger(0);
          if (last_idx != this_idx) {
             prof_trigger(1);
             r1 = img[this_idx].x;
             i1 = img[this_idx].y;
             last_idx = this_idx;
          }
#ifdef __COMPUTE_GCF
          //double phase = 2*3.1415926*w*(1-T*sqrt((main_x-inn.x)*(main_x-inn.x)+(main_y-inn.y)*(main_y-inn.y)));
          //double r2 = sin(phase);
          //double i2 = cos(phase);
          float xsquare = (main_x-inn.x);
          float ysquare = (main_x-inn.x);
          xsquare *= xsquare;
          ysquare *= ysquare;
          float phase = p1 - p2*sqrt(xsquare + ysquare);
          float r2,i2;
          sincosf(phase, &r2, &i2);
#else
          int sub_x = inn.x%GCF_GRID;
          int sub_y = inn.y%GCF_GRID;
          int b = this_y - main_y;
          int a = this_x - main_x;
          auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a].x);
          auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                         gcf_dim*b+a].y);
#endif
          sum_r = r1*r2 - i1*i2; 
          sum_i = r1*i2 + r2*i1;
      }

     //reduce in two directions
      //WARNING: Adjustments must be made if blockDim.y and blockDim.x are no
      //         powers of 2 
      //Reduce using shuffle first
#if 1
      warp_reduce2(sum_r);
      warp_reduce2(sum_i);
#if 0
      //Write immediately
      if (0 == threadIdx.x%32) {
         atomicAdd(&(out[n+q].x),sum_r);
         atomicAdd(&(out[n+q].y),sum_i);
      }
#else
      //Reduce again in shared mem
      if (0 == threadIdx.x%32) {
         //Save results as if shared memory were blockDim.y*32 by blockDim.x/32
         //Each q writes a unique set of blockDim.y rows
         shm[0][(threadIdx.y+q*blockDim.y)*blockDim.x/32+threadIdx.x/32].x = sum_r;
         shm[0][(threadIdx.y+q*blockDim.y)*blockDim.x/32+threadIdx.x/32].y = sum_i;
      }
      if (q==31 || n+q == npts/gridDim.x-1) {
         //Once we have filled all of shared memory, reduce further
         //and write using atomicAdd
         __syncthreads();
         sum_r=shm[threadIdx.y][threadIdx.x].x;
         sum_i=shm[threadIdx.y][threadIdx.x].y;
         if (blockDim.x*blockDim.y>1024) {
            warp_reduce2(sum_r);
            warp_reduce2(sum_i);
            if (0==(threadIdx.x + threadIdx.y*blockDim.x)%32) {
               atomicAdd(&(out[n+(threadIdx.x+threadIdx.y*blockDim.x)/(blockDim.x*blockDim.y/32)].x), sum_r);
               atomicAdd(&(out[n+(threadIdx.x+threadIdx.y*blockDim.x)/(blockDim.x*blockDim.y/32)].y), sum_i);
            }
         } else {
            warp_reduce2(sum_r,blockDim.x*blockDim.y/32); 
            warp_reduce2(sum_i,blockDim.x*blockDim.y/32); 
            if (0==(threadIdx.x + threadIdx.y*blockDim.x)%(blockDim.x*blockDim.y/32)) {
               atomicAdd(&(out[n+(threadIdx.x+threadIdx.y*blockDim.x)/(blockDim.x*blockDim.y/32)].x), sum_r);
               atomicAdd(&(out[n+(threadIdx.x+threadIdx.y*blockDim.x)/(blockDim.x*blockDim.y/32)].y), sum_i);
            }
         }
      }
#endif
#else

      //Simple reduction
      
      shm[threadIdx.y][threadIdx.x].x = sum_r;
      shm[threadIdx.y][threadIdx.x].y = sum_i;
      __syncthreads();
      //Reduce in y
      for(int s = blockDim.y/2;s>0;s/=2) {
         if (threadIdx.y < s) {
           shm[threadIdx.y][threadIdx.x].x += shm[threadIdx.y+s][threadIdx.x].x;
           shm[threadIdx.y][threadIdx.x].y += shm[threadIdx.y+s][threadIdx.x].y;
         }
         __syncthreads();
      }

      //Reduce the top row
      for(int s = blockDim.x/2;s>16;s/=2) {
         if (0 == threadIdx.y && threadIdx.x < s) 
                    shm[0][threadIdx.x].x += shm[0][threadIdx.x+s].x;
         if (0 == threadIdx.y && threadIdx.x < s) 
                    shm[0][threadIdx.x].y += shm[0][threadIdx.x+s].y;
         __syncthreads();
      }
      if (threadIdx.y == 0) {
         //Reduce the final warp using shuffle
         CmplxType tmp = shm[0][threadIdx.x];
         for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
            tmp.x += __shfl_down(tmp.x,s);
            tmp.y += __shfl_down(tmp.y,s);
         }
         if (threadIdx.x == 0) {
            atomicAdd(&(out[n+q].x),tmp.x);
            atomicAdd(&(out[n+q].y),tmp.y);
            //out[n+q].x=tmp.x;
            //out[n+q].y=tmp.y;
         }
      }
      __syncthreads();
#endif
   } //q
   __syncthreads();
   } //n
}
#endif

template <class CmplxType>
void gridGPU(CmplxType* out, CmplxType* in, CmplxType* in_vals, size_t npts, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim) {
//grid on the GPU
//  out (out) - the output image
//  in  (in)  - the input locations
//  in_vals (in) - input values
//  npts (in) - number of locations
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   CmplxType *d_out, *d_in, *d_in_vals, *d_gcf;

   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);

   CUDA_CHECK_ERR(__LINE__,__FILE__);
#ifdef __MANAGED
   d_gcf = gcf;
   std::cout << "d_out = out" << std::endl;
   d_out = out;
   d_in = in;
   d_in_vals = in_vals;
                                                                 sizeof(CmplxType) << std::endl;
#else
   //img is padded to avoid overruns. Subtract to find the real head

   //Pin CPU memory
   cudaHostRegister(out, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim), cudaHostRegisterMapped);
   cudaHostRegister(gcf, sizeof(CmplxType)*GCF_GRID*GCF_GRID*gcf_dim*gcf_dim, cudaHostRegisterMapped);
   cudaHostRegister(in, sizeof(CmplxType)*npts, cudaHostRegisterMapped);
   cudaHostRegister(in_vals, sizeof(CmplxType)*npts, cudaHostRegisterMapped);

   //Allocate GPU memory
   cudaMalloc(&d_out, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim));
   cudaMalloc(&d_gcf, sizeof(CmplxType)*GCF_GRID*GCF_GRID*gcf_dim*gcf_dim);
   cudaMalloc(&d_in, sizeof(CmplxType)*npts);
   cudaMalloc(&d_in_vals, sizeof(CmplxType)*npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Copy in img, gcf and out
   cudaEventRecord(start);
   cudaMemcpy(d_gcf, gcf, sizeof(CmplxType)*64*gcf_dim*gcf_dim, 
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in, in, sizeof(CmplxType)*npts,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in_vals, in_vals, sizeof(CmplxType)*npts,
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   std::cout << "memcpy time: " << getElapsed(start, stop) << " ms." << std::endl;

   //move d_img and d_gcf to remove padding
#endif
   //offset gcf to point to the middle of the first GCF for cleaner code later
   d_gcf += gcf_dim*(gcf_dim-1)/2-1;
   CmplxType* d_out_unpad = d_out + img_dim*gcf_dim+gcf_dim;

#ifdef __GATHER
   int2* in_ints;
   int* bookmarks;
   cudaMalloc(&in_ints, sizeof(int2)*npts);
   cudaMalloc(&bookmarks, sizeof(int)*((img_dim/gcf_dim)*(img_dim/gcf_dim)*4+1));
   vis2ints<<<4,256>>>(d_in, in_ints, npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   set_bookmarks<<<4,256>>>(in_ints, npts, gcf_dim/2, (img_dim+gcf_dim/2-1)/(gcf_dim/2), 
                               bookmarks);
   int2* h_ints = (int2*)malloc(sizeof(int2)*npts);
   cudaMemcpy(h_ints, in_ints, sizeof(int2)*npts, cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   
   
   cudaMemset(d_out, 0, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim));
   cudaEventRecord(start);
   grid_kernel_gather<GCF_DIM>
            <<<dim3((img_dim+gcf_dim/4-1)/(gcf_dim/4), (img_dim+gcf_dim/4-1)/(gcf_dim/4)),
               dim3(gcf_dim/4, gcf_dim/4/PTS)>>> // <-- Must not truncate here
   //         <<<dim3((img_dim+gcf_dim-1)/(gcf_dim), (img_dim+gcf_dim-1)/(gcf_dim)),
   //            dim3(gcf_dim, gcf_dim)>>>
                             (d_out_unpad,in_ints,d_in_vals,npts,img_dim,d_gcf,bookmarks); 
   //std::cout<< "grid_kernel_gather<<<(" << (img_dim+gcf_dim-1)/gcf_dim << ", " << (img_dim+gcf_dim/4-1)/(gcf_dim/4) << "), (" << gcf_dim << ", " << gcf_dim/4 << ")>>>()" << std::endl; 
   CUDA_CHECK_ERR(__LINE__,__FILE__);
#else
#ifdef __MOVING_WINDOW
   int2* in_ints;
   cudaMalloc(&in_ints, sizeof(int2)*npts);
   vis2ints<<<4,256>>>(d_in, in_ints, npts);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemset(d_out, 0, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim));
   cudaEventRecord(start);
   grid_kernel_window<GCF_DIM>
               <<<dim3(npts/32,GCF_DIM/BLOCK_Y),dim3(GCF_DIM,BLOCK_Y)>>>(d_out_unpad,in_ints,d_in_vals,npts,img_dim,d_gcf); 
   //vis2ints<<<dim3(npts/64,8),dim3(GCF_DIM,GCF_DIM/8)>>>(d_in, in_ints, npts);
#else
   cudaEventRecord(start);
   cudaMemset(d_out, 0, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim));
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
   std::cout << npts / 1000000.0 / kernel_time * gcf_dim * gcf_dim * 8 << "Gflops" << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

#ifdef __MANAGED
   cudaDeviceSynchronize();
#else
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(out, d_out, 
              sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim), 
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
template void gridGPU<double2>(double2* out, double2* in, double2* in_vals, size_t npts,
                                 size_t img_dim, double2 *gcf, size_t gcf_dim); 
template void gridGPU<float2>(float2* out, float2* in, float2* in_vals, size_t npts,
                                size_t img_dim, float2 *gcf, size_t gcf_dim); 
