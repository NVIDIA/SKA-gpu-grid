#include <iostream>
#include "math.h"
#include "stdlib.h"

#include "grid_gpu.cuh"
#include "Defines.h"
#include "cuda.h"

//With managed memory, grid.cpp must be compiled as CUDA
//in which case float2 and double2 are predefined.
//typedef struct {float x,y;} float2;
//typedef struct {double x,y;} double2;

#define single 77
#if PRECISION==single
#define PRECISION float
#endif

#ifndef PRECISION
#define PRECISION double
#endif
#define PASTER(x) x ## 2
#define EVALUATOR(x) PASTER(x)
#define PRECISION2 EVALUATOR(PRECISION)


void init_gcf(PRECISION2 *gcf, size_t size) {

  for (size_t sub_x=0; sub_x<GCF_GRID; sub_x++ )
   for (size_t sub_y=0; sub_y<GCF_GRID; sub_y++ )
    for(size_t x=0; x<size; x++)
     for(size_t y=0; y<size; y++) {
       //Some nonsense GCF
       PRECISION tmp = sin(6.28*x/size/GCF_GRID)*exp(-(1.0*x*x+1.0*y*y*sub_y)/size/size/2);
       gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].x = tmp*sin(1.0*x*sub_x/(y+1));
       gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].y = tmp*cos(1.0*x*sub_x/(y+1));
       //std::cout << tmp << gcf[x+y*size].x << gcf[x+y*size].y << std::endl;
     }

}

void gridCPU(PRECISION2* out, PRECISION2 *in, size_t npts, PRECISION2 *img, size_t img_dim, PRECISION2 *gcf, size_t gcf_dim) {
//grid on the CPU
//  out (out) - the output values for each location
//  in  (in)  - the locations to be interpolated 
//  npts (in) - number of locations
//  img (in) - the image
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   //offset gcf to point to the middle for cleaner code later
   gcf += GCF_DIM*(GCF_DIM+1)/2;
#pragma acc parallel loop copyout(out[0:NPOINTS]) copyin(in[0:NPOINTS],gcf[0:GCF_GRID*GCF_GRID*GCF_DIM*GCF_DIM],img[IMG_SIZE*IMG_SIZE]) gang
#pragma omp parallel for
   for(size_t n=0; n<NPOINTS; n++) {
      //std::cout << "in = " << in[n].x << ", " << in[n].y << std::endl;
      int sub_x = floorf(GCF_GRID*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(GCF_GRID*(in[n].y-floorf(in[n].y)));
      //std::cout << "sub = "  << sub_x << ", " << sub_y << std::endl;
      int main_x = floor(in[n].x); 
      int main_y = floor(in[n].y); 
      //std::cout << "main = " << main_x << ", " << main_y << std::endl;
      PRECISION sum_r = 0.0;
      PRECISION sum_i = 0.0;
      #pragma acc parallel loop collapse(2) reduction(+:sum_r,sum_i) vector
#pragma omp parallel for collapse(2) reduction(+:sum_r, sum_i)
      for (int a=-GCF_DIM/2; a<GCF_DIM/2 ;a++)
      for (int b=-GCF_DIM/2; b<GCF_DIM/2 ;b++) {
         PRECISION r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x; 
         PRECISION i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y; 
         PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + 
                        GCF_DIM*b+a].x;
         PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + 
                        GCF_DIM*b+a].y;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
         } else {
            sum_r += r1*r2 - i1*i2; 
            sum_i += r1*i2 + r2*i1;
         }
      }
      out[n].x = sum_r;
      out[n].y = sum_i;
      //std::cout << "val = " << out[n].r << "+ i" << out[n].i << std::endl;
   } 
   gcf -= GCF_DIM*(GCF_DIM+1)/2;
}
template <class T,class Thalf>
int w_comp_main(const void* A, const void* B) {
   Thalf quota, rema, quotb, remb;
   rema = modf((*((T*)A)).x, &quota);
   remb = modf((*((T*)B)).x, &quotb);
   if (quota > quotb) return 1;
   if (quota < quotb) return -1;
   else {
     rema = modf((*((T*)A)).y, &quota);
     remb = modf((*((T*)B)).y, &quotb);
     if (quota > quotb) return 1;
     if (quota < quotb) return -1;
     else return 0;
   }
   return 0;
}
template <class T,class Thalf>
int w_comp_sub(const void* A, const void* B) {
   Thalf quota, rema, quotb, remb;
   rema = modf((*((T*)A)).x, &quota);
   remb = modf((*((T*)B)).x, &quotb);
   int sub_xa = (int) (GCF_GRID*rema);
   int sub_xb = (int) (GCF_GRID*remb);
   rema = modf((*((T*)A)).y, &quota);
   remb = modf((*((T*)B)).y, &quotb);
   int suba = (int) (GCF_GRID*rema) + GCF_GRID*sub_xa;
   int subb = (int) (GCF_GRID*remb) + GCF_GRID*sub_xb;
   if (suba > subb) return 1;
   if (suba < subb) return -1;
   return 0;
}
template <class T,class Thalf>
int w_comp_full(const void* A, const void* B) {
   int result = w_comp_sub<T,Thalf>(A,B);
   if (0==result) return w_comp_main<T,Thalf>(A,B);
   else return result;
}
#if 0
struct comp_grid {
   int blockgrid, blocksize;
   public:
   comp_grid(int img_dim, int gcf_dim) {
      blocksize = gcf_dim/2;
      blockgrid = img_dim/blocksize;
   }
   int __cdecl operator () (const void* A, const void* B) const {
      int gridxa = (*(int2*)A).x/GCF_GRID;
      int gridxb = (*(int2*)B).x/GCF_GRID;
      int gridya = (*(int2*)A).y/GCF_GRID;
      int gridyb = (*(int2*)B).y/GCF_GRID;
      if (gridya > gridyb) return 1;
      if (gridya < gridyb) return -1;
      if (gridxa > gridxb) return 1;
      if (gridxa < gridxb) return  -1;
      int suba = GCF_GRID*((*(int2*)A).x%GCF_GRID) + (*(int2*)A).y%GCF_GRID;
      int subb = GCF_GRID*((*(int2*)B).x%GCF_GRID) + (*(int2*)B).y%GCF_GRID;
      if (suba > subb) return 1;
      if (suba < subb) return -1;
      return  0;
   }
};
#else
template <class T, class Thalf>
int comp_grid (const void* A, const void* B) {
      int blocksize = GCF_DIM/2;
      int mainxa = floorf((*(T*)A).x);
      int mainxb = floorf((*(T*)B).x);
      int mainya = floorf((*(T*)A).y);
      int mainyb = floorf((*(T*)B).y);
      int gridxa = mainxa/blocksize;
      int gridxb = mainxb/blocksize;
      int gridya = mainya/blocksize;
      int gridyb = mainyb/blocksize;
      if (gridya*IMG_SIZE/blocksize+gridxa > 
          gridyb*IMG_SIZE/blocksize+gridxb) return 1;
      if (gridya*IMG_SIZE/blocksize+gridxa < 
          gridyb*IMG_SIZE/blocksize+gridxb) return -1;
      Thalf suba = GCF_GRID*((*(T*)A).x-mainxa) + (*(T*)A).y-mainya;
      Thalf subb = GCF_GRID*((*(T*)B).x-mainxb) + (*(T*)B).y-mainyb;
      if (suba > subb) return 1;
      if (suba < subb) return -1;
      return  0;
}
#endif


int main(void) {

#ifdef __MANAGED
   PRECISION2* out, *in, *img, *gcf;
   cudaMallocManaged(&out, sizeof(PRECISION2)*NPOINTS);
   cudaMallocManaged(&in, sizeof(PRECISION2)*NPOINTS);
   cudaMallocManaged(&img, sizeof(PRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM));
   cudaMallocManaged(&gcf, sizeof(PRECISION2)*64*GCF_DIM*GCF_DIM);
#else
   PRECISION2* out = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS);
   PRECISION2* in = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS);
   PRECISION2 *img = (PRECISION2*) malloc((IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*sizeof(PRECISION2));

   PRECISION2 *gcf = (PRECISION2*) malloc(64*GCF_DIM*GCF_DIM*sizeof(PRECISION2));
#endif

   //img is padded (above and below) to avoid overruns
   img += IMG_SIZE*GCF_DIM+GCF_DIM;
    
   init_gcf(gcf, GCF_DIM);
   srand(2541617);
   for(size_t n=0; n<NPOINTS; n++) {
      in[n].x = ((float)rand())/RAND_MAX*8000;
      in[n].y = ((float)rand())/RAND_MAX*8000;
   }
   for(size_t x=0; x<IMG_SIZE;x++)
   for(size_t y=0; y<IMG_SIZE;y++) {
      img[x+IMG_SIZE*y].x = exp(-((x-1400.0)*(x-1400.0)+(y-3800.0)*(y-3800.0))/8000000.0)+1.0;
      img[x+IMG_SIZE*y].y = 0.4;
   }
   //Zero the data in the offset areas
   for (int x=-IMG_SIZE*GCF_DIM-GCF_DIM;x<0;x++) {
      img[x].x = 0.0; img[x].y = 0.0;
   }
   for (int x=0;x<IMG_SIZE*GCF_DIM+GCF_DIM;x++) {
      img[x+IMG_SIZE*IMG_SIZE].x = 0.0; img[x+IMG_SIZE*IMG_SIZE].y = 0.0;
   }

#ifdef __SCATTER
   std::qsort(in, NPOINTS, sizeof(PRECISION2), comp_grid<PRECISION2,PRECISION>);
#else
#ifdef __MOVING_WINDOW
   std::qsort(in, NPOINTS, sizeof(PRECISION2), w_comp_main<PRECISION2,PRECISION>);
#else
   std::qsort(in, NPOINTS, sizeof(PRECISION2), w_comp_sub<PRECISION2,PRECISION>);
#endif
#endif
   std::cout << "sorted" << std::endl;
   
   gridGPU(out,in,NPOINTS,img,IMG_SIZE,gcf,GCF_DIM);
#ifdef __CPU_CHECK
   std::cout << "Computing on CPU..." << std::endl;
   PRECISION2 *out_cpu=(PRECISION2*)malloc(sizeof(PRECISION2)*NPOINTS);
   gridCPU(out_cpu,in,NPOINTS,img,IMG_SIZE,gcf,GCF_DIM);
#endif


#ifdef __CPU_CHECK
   std::cout << "Checking results against CPU:" << std::endl;
   for (size_t n = 0; n < NPOINTS; n++) {
     if (fabs(out[n].x-out_cpu[n].x) > 0.0000001 ||
         fabs(out[n].y-out_cpu[n].y) > 0.0000001 )
        std::cout << n << ": F(" << in[n].x << ", " << in[n].y << ") = " 
                  << out[n].x << ", " << out[n].y 
                  << " vs. " << out_cpu[n].x << ", " << out_cpu[n].y 
                  << std::endl;
   }
#endif
   img -= GCF_DIM + IMG_SIZE*GCF_DIM;
#ifdef __MANAGED
   cudaFree(out);
   cudaFree(in);
   cudaFree(img);
   cudaFree(gcf);
#else
   free(out);
   free(in);
   free(img);
   free(gcf);
#endif
}
