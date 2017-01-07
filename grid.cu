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

#include <iostream>
#include "math.h"
#include "stdlib.h"

#include "grid_gpu.cuh"
#include "Defines.h"
#include "cuda.h"
#ifdef __HDF5_INPUT
#include <vector>
#include "H5Cpp.h"
#include "vis.h"

using namespace H5;

std::vector<struct vis> HDF5_to_struct(H5File* file);
#endif

//With managed memory, grid.cpp must be compiled as CUDA
//in which case float2 and double2 are predefined.
//typedef struct {float x,y;} float2;
//typedef struct {double x,y;} double2;

#define single 77
#if PRECISION==single
#define PRECISION float
#endif
#if OUTPRECISION==single
#define OUTPRECISION float
#endif

#ifndef PRECISION
#define PRECISION double
#endif
#define PASTER(x) x ## 2
#define EVALUATOR(x) PASTER(x)
#define PRECISION2 EVALUATOR(PRECISION)

#ifndef OUTPRECISION
#define OUTPRECISION PRECISION
#endif
#define OUTPRECISION2 EVALUATOR(OUTPRECISION)


void init_gcf(PRECISION2 *gcf, size_t size) {

  for (size_t sub_x=0; sub_x<GCF_GRID; sub_x++ )
   for (size_t sub_y=0; sub_y<GCF_GRID; sub_y++ )
    for(size_t x=0; x<size; x++)
     for(size_t y=0; y<size; y++) {
       //Some nonsense GCF
       PRECISION tmp = sin(6.28*x/size/GCF_GRID)*exp(-(1.0*x*x+1.0*y*y*sub_y)/size/size/2);
       gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].x = tmp*sin(1.0*x*sub_x/(y+1))+0.4;
       gcf[size*size*(sub_x+sub_y*GCF_GRID)+x+y*size].y = tmp*cos(1.0*x*sub_x/(y+1))-0.2;
       //std::cout << tmp << gcf[x+y*size].x << gcf[x+y*size].y << std::endl;
     }

}

void gridCPU(PRECISION2* out, PRECISION2 *in, PRECISION2 *in_vals, size_t npts, size_t img_dim, PRECISION2 *gcf, size_t gcf_dim) {
//degrid on the CPU
//  out (out) - the output image
//  in  (in)  - the input locations
//  in_vals (in) - input values
//  npts (in) - number of locations
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   //Zero the output
   for (size_t n=0;n<IMG_SIZE*IMG_SIZE; n++) out[n].x = out[n].y = 0.0;
   //offset gcf to point to the middle for cleaner code later
   gcf += GCF_DIM*(GCF_DIM-1)/2-1;
   double* out1 = (double*)out;
   double* out2 = (double*)(out+POLARIZATIONS*img_dim*img_dim/2);
//#pragma acc parallel loop copyout(out[0:NPOINTS]) copyin(in[0:NPOINTS],gcf[0:GCF_GRID*GCF_GRID*GCF_DIM*GCF_DIM],img[IMG_SIZE*IMG_SIZE]) gang
//#pragma omp parallel for
   for(size_t n=0; n<npts; n++) {
      //std::cout << "in = " << in[n].x << ", " << in[n].y << std::endl;
      int sub_x = floorf(GCF_GRID*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(GCF_GRID*(in[n].y-floorf(in[n].y)));
      //std::cout << "sub = "  << sub_x << ", " << sub_y << std::endl;
      int main_x = floor(in[n].x); 
      int main_y = floor(in[n].y); 
      //std::cout << "main = " << main_x << ", " << main_y << std::endl;
//      #pragma acc parallel loop collapse(2) reduction(+:sum_r,sum_i) vector
//#pragma omp parallel for collapse(2) reduction(+:sum_r, sum_i)
      for (int a=GCF_DIM/2; a>-GCF_DIM/2 ;a--)
      for (int b=GCF_DIM/2; b>-GCF_DIM/2 ;b--) {
         PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + 
                        GCF_DIM*b+a].x;
         PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x) + 
                        GCF_DIM*b+a].y;
         PRECISION r1, i1;
         r1 = in_vals[n].x;
         i1 = in_vals[n].y;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
         } else {
#ifdef DEBUG1
               out1[main_x+a+IMG_SIZE*(main_y+b)] += 1;
               out2[main_x+a+IMG_SIZE*(main_y+b)] += n;
#else
               out1[main_x+a+IMG_SIZE*(main_y+b)] += r1*r2-i1*i2; 
               out2[main_x+a+IMG_SIZE*(main_y+b)] += r1*i2+r2*i1;
#endif
         }
      }
      //std::cout << "val = " << out[n].r << "+ i" << out[n].i << std::endl;
   } 
   gcf -= GCF_DIM*(GCF_DIM-1)/2-1;
}
void gridCPU_pz(PRECISION2* out, PRECISION2 *in, PRECISION2 *in_vals, int* in_gcfinx, 
                size_t npts, size_t img_dim, PRECISION2 *gcf, size_t gcf_dim) {
//degrid on the CPU
//  out (out) - the output image
//  in  (in)  - the input locations
//  in_vals (in) - input values
//  npts (in) - number of locations
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   //Zero the output
   //offset gcf to point to the middle for cleaner code later
   gcf += GCF_DIM*(GCF_DIM-1)/2-1;
   double* out1 = (double*)out;
   double* out2 = (double*)(out+POLARIZATIONS*img_dim*img_dim/2);
//#pragma acc parallel loop copyout(out[0:NPOINTS]) copyin(in[0:NPOINTS],gcf[0:GCF_GRID*GCF_GRID*GCF_DIM*GCF_DIM],img[IMG_SIZE*IMG_SIZE]) gang
//#pragma omp parallel for
   for(size_t n=0; n<npts; n++) {
      //std::cout << "in = " << in[n].x << ", " << in[n].y << std::endl;
      int sub_x = floorf(GCF_GRID*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(GCF_GRID*(in[n].y-floorf(in[n].y)));
      //std::cout << "sub = "  << sub_x << ", " << sub_y << std::endl;
      int main_x = floor(in[n].x); 
      int main_y = floor(in[n].y); 
      //std::cout << "main = " << main_x << ", " << main_y << std::endl;
//      #pragma acc parallel loop collapse(2) reduction(+:sum_r,sum_i) vector
//#pragma omp parallel for collapse(2) reduction(+:sum_r, sum_i)
      for (int a=GCF_DIM/2; a>-GCF_DIM/2 ;a--)
      for (int b=GCF_DIM/2; b>-GCF_DIM/2 ;b--) {
         PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x+GCF_GRID*GCF_GRID*in_gcfinx[n]) + 
                        GCF_DIM*b+a].x;
         PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(GCF_GRID*sub_y+sub_x+GCF_GRID*GCF_GRID*in_gcfinx[n]) + 
                        GCF_DIM*b+a].y;
         PRECISION r1, i1;
         if (main_x+a < 0 || main_y+b < 0 || 
             main_x+a >= IMG_SIZE  || main_y+b >= IMG_SIZE) {
         } else {
            for (int p=0;p< POLARIZATIONS;p++) {
               r1 = in_vals[POLARIZATIONS*n+p].x;
               i1 = in_vals[POLARIZATIONS*n+p].y;
#ifdef DEBUG1
               out1[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE] += 1;
               out2[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE] += n;
#else
               out1[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE] += r1*r2-i1*i2; 
               out2[main_x+a+IMG_SIZE*(main_y+b)+p*IMG_SIZE*IMG_SIZE] += r1*i2+r2*i1;
#endif
            }
         }
      }
      //std::cout << "val = " << out[n].r << "+ i" << out[n].i << std::endl;
   } 
   gcf -= GCF_DIM*(GCF_DIM-1)/2-1;
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
      if (gridya*(IMG_SIZE+blocksize-1)/blocksize+gridxa > 
          gridyb*(IMG_SIZE+blocksize-1)/blocksize+gridxb) return 1;
      if (gridya*(IMG_SIZE+blocksize-1)/blocksize+gridxa < 
          gridyb*(IMG_SIZE+blocksize-1)/blocksize+gridxb) return -1;
      Thalf suba = GCF_GRID*((*(T*)A).x-mainxa) + (*(T*)A).y-mainya;
      Thalf subb = GCF_GRID*((*(T*)B).x-mainxb) + (*(T*)B).y-mainyb;
      if (suba > subb) return 1;
      if (suba < subb) return -1;
      return  0;
}
#endif


int main(int argc, char** argv) {

#ifdef __MANAGED
   OUTPRECISION2 *out;
   PRECISION2 *in, *in_vals, *gcf;
   cudaMallocManaged(&out, sizeof(OUTPRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*POLARIZATIONS);
   cudaMallocManaged(&in, sizeof(PRECISION2)*NPOINTS);
   cudaMallocManaged(&in_vals, sizeof(PRECISION2)*NPOINTS*POLARIZATIONS);
   cudaMallocManaged(&in_gcfinx, sizeof(PRECISION2)*NPOINTS*POLARIZATIONS);
   cudaMallocManaged(&gcf, sizeof(PRECISION2)*GCF_GRID*GCF_GRID*GCF_DIM*GCF_DIM*NGCF);
#else
   OUTPRECISION2* out = (OUTPRECISION2*) malloc(sizeof(OUTPRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*POLARIZATIONS);
   PRECISION2* in = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS);
   PRECISION2* in_vals = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS*POLARIZATIONS);
   int* in_gcfinx = (int*) malloc(sizeof(int)*NPOINTS*POLARIZATIONS);

   PRECISION2 *gcf = (PRECISION2*) malloc(GCF_GRID*GCF_GRID*GCF_DIM*GCF_DIM*NGCF*sizeof(PRECISION2));
#endif
   int npts=NPOINTS;

  // ***  Report run parameters ***
   printf("*** GPU Gridding ***\n");
#ifdef DEBUG1
   printf("\n   Debug\n\n");
#endif
#ifdef __GATHER
   printf("   Gather strategy\n");
#else
   #ifdef __MOVING_WINDOW
      printf("   Moving Window strategy\n");
   #else
      printf("   Simple scatter strategy\n");
   #endif
   #ifdef __NOATOMIC
      printf("   No atomics\n");
   #endif
#endif
#if PRECISION==double
   printf("   Double precision\n");
#else
   printf("   Single precision\n");
#endif
   printf("   Image size %dx%d\n", IMG_SIZE, IMG_SIZE);
   printf("   GCF size %dx%d\n", GCF_DIM, GCF_DIM);
   printf("   %d polarizations\n", POLARIZATIONS);
   printf("   %d visibilities\n", npts);
   printf("   Subgrid: 1/%d\n", GCF_GRID);
#ifdef __FILE_INPUT
   printf("   File input\n");
#endif
#ifdef __HDF5_INPUT
   printf("   HDF5 input\n");
#endif
#ifdef __COMPUTE_GCF
   printf("   Computed GCF\n"); 
#endif
   printf("\n\n\n");


   for (int q=0; q<NGCF;q++)
   {
      init_gcf(gcf + q*GCF_DIM*GCF_DIM*GCF_GRID*GCF_GRID, GCF_DIM);
   }
#ifdef __FILE_INPUT
   char filename[400];
   if (argc>1)
   {
      filename = argv[1];
   } else {
      sprintf(filename, "%s", "UVW_in.dat");
   }
   FILE *uvw_f = fopen("UVW_in.dat", "r");
   int junka,junkb,junkc;
   float fjunka, fjunkb, fjunkc;
   float max_x, min_x, max_y, min_y;
   max_x = max_y = INT_MIN;
   min_x = min_y = INT_MAX;
   for(size_t n=0; n<npts; n++) {
      fscanf(uvw_f, "%d,%d,%d: %f, %f, %f\n", &junka, &junkb, &junkc, &fjunka, &fjunkb, &fjunkc);
      in[n].x = fjunka*IMG_SIZE/2048.;
      in[n].y = fjunkb*IMG_SIZE/2048.;
      min_x = in[n].x < min_x ? in[n].x : min_x;
      max_x = in[n].x > max_x ? in[n].x : max_x;
      min_y = in[n].y < min_y ? in[n].y : min_y;
      max_y = in[n].y > max_y ? in[n].y : max_y;
      for (int p=0;p<POLARIZATIONS;p++) {
         in_vals[POLARIZATIONS*n+p].x = ((float)rand())/RAND_MAX;
         in_vals[POLARIZATIONS*n+p].y = ((float)rand())/RAND_MAX;
         in_gcfinx[POLARIZATIONS*n+p] = rand()%NCGF;
      }
   }
   printf("%f -- %f, %f -- %f\n", min_x, max_x, min_y, max_y);
   fclose(uvw_f);
#else
#ifdef __HDF5_INPUT
   //char* filename[]="vis.hdf5";
#if 0
   if (argc>1)
   {
      sprintf(filename, "%s", argv[1]);
   } else {
      sprintf(filename, "%s", "vis.h5");
   }
#endif

   H5File* file;
   if (argc>1) file = new H5File(H5std_string(argv[1]), H5F_ACC_RDONLY);
   else file = new H5File(H5std_string("vis.h5"), H5F_ACC_RDONLY);

   try { //For scoping visarray
      std::vector<struct vis> visarray = HDF5_to_struct(file);
      file->close();

      size_t total_sz = visarray.size();
   
      free(in);
      in = (PRECISION2*) malloc(sizeof(PRECISION2)*total_sz);
      free(in_vals);
      in_vals = (PRECISION2*) malloc(sizeof(PRECISION2)*total_sz*POLARIZATIONS);
   
      float inminx = INT_MAX;
      float inminy = INT_MAX;
      float inmaxx = -INT_MAX;
      float inmaxy = -INT_MAX;
      for (int q=0;q<total_sz;q++)
      {
         in[q].x = visarray[q].u;
         in[q].y = visarray[q].v;
         for (int p=0;p<POLARIZATIONS;p++)
         {
            in_vals[POLARIZATIONS*q+p].x = visarray[q].r;
            in_vals[POLARIZATIONS*q+p].y = visarray[q].i;
         }
         inminx = inminx < in[q].x ? inminx : in[q].x;
         inminy = inminy < in[q].y ? inminy : in[q].y;
         inmaxx = inmaxx > in[q].x ? inmaxx : in[q].x;
         inmaxy = inmaxy > in[q].y ? inmaxy : in[q].y;
      
      }
      printf("Image limits: (%f, %f) -- (%f, %f)\n", inminx, inminy, inmaxx, inmaxy);

      npts = total_sz;
      printf("   %d visibilities\n", npts);

   }
   
   catch( FileIException error )
   {
      error.printError();
      return -1;
   }
   // catch failure caused by the DataSet operations
   catch( DataSetIException error )
   {
      error.printError();
      return -1;
   }
   // catch failure caused by the DataSpace operations
   catch( DataSpaceIException error )
   {
      error.printError();
      return -1;
   }
   // catch failure caused by the DataSpace operations
   catch( DataTypeIException error )
   {
      error.printError();
   }

#else
   srand(1541617);
   for(size_t n=0; n<npts; n++) {
      in[n].x = ((float)rand())/RAND_MAX*IMG_SIZE;
      in[n].y = ((float)rand())/RAND_MAX*IMG_SIZE;
      for (int p=0;p<POLARIZATIONS;p++) {
         in_vals[POLARIZATIONS*n+p].x = ((float)rand())/RAND_MAX;
         in_vals[POLARIZATIONS*n+p].y = ((float)rand())/RAND_MAX;
         in_gcfinx[POLARIZATIONS*n+p] = rand()%NGCF;
      }
   }
#endif //HDF5_INPUT
#endif
   //Zero the data in the offset areas
   //for (int x=-IMG_SIZE*GCF_DIM-GCF_DIM;x<0;x++) {
   //   out[x].x = 0.0; out[x].y = 0.0;
  // }
   for (int x=0;x<IMG_SIZE*GCF_DIM*POLARIZATIONS+GCF_DIM*POLARIZATIONS;x++) {
      out[x].x=0.0;
      out[x].y=0.0;
      out[x+(IMG_SIZE*IMG_SIZE+IMG_SIZE*GCF_DIM+GCF_DIM)*POLARIZATIONS].x = 0.0;
      out[x+(IMG_SIZE*IMG_SIZE+IMG_SIZE*GCF_DIM+GCF_DIM)*POLARIZATIONS].y = 0.0;
   }


#ifdef __GATHER
   std::qsort(in, npts, sizeof(PRECISION2), comp_grid<PRECISION2,PRECISION>);
#else
#ifdef __MOVING_WINDOW
   std::qsort(in, npts, sizeof(PRECISION2), w_comp_main<PRECISION2,PRECISION>);
#else
   std::qsort(in, npts, sizeof(PRECISION2), w_comp_sub<PRECISION2,PRECISION>);
#endif
#endif
   
   //auto tmp = in[0];
   //in[0] = in[204];
   //in[204]=tmp;
   std::cout << "Computing on GPU..." << std::endl;
   gridGPU(out,in,in_vals,in_gcfinx,npts,IMG_SIZE,gcf,GCF_DIM);
#ifdef __CPU_CHECK
   std::cout << "Computing on CPU..." << std::endl;
   PRECISION2 *out_cpu=(PRECISION2*)malloc(sizeof(PRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*POLARIZATIONS);
   memset(out_cpu, 0, sizeof(PRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*POLARIZATIONS);
   
   gridCPU_pz(out_cpu+IMG_SIZE*GCF_DIM+GCF_DIM,in,in_vals,in_gcfinx,npts,IMG_SIZE,gcf,GCF_DIM);
   //gridCPU(out+IMG_SIZE*GCF_DIM+GCF_DIM,in,in_vals,in_gcfinx,npts,IMG_SIZE,gcf,GCF_DIM);
#endif


#ifdef __CPU_CHECK
   std::cout << "Checking results against CPU:" << std::endl;
   for (size_t yy = 0; yy < IMG_SIZE; yy++) {
   for (size_t xx = 0; xx < IMG_SIZE; xx++) {
     int n = GCF_DIM+IMG_SIZE*GCF_DIM+yy*IMG_SIZE+xx;
     for (int p = 0; p < IMG_SIZE*IMG_SIZE*POLARIZATIONS; p+=IMG_SIZE*IMG_SIZE) {
        if (fabs(out[n+p].x-out_cpu[n+p].x) > 0.0000001 ||
            fabs(out[n+p].y-out_cpu[n+p].y) > 0.0000001 )
           std::cout << xx << ", " << yy << "[" << p/IMG_SIZE/IMG_SIZE << "] : " 
                     << "(" << n+p-(GCF_DIM+IMG_SIZE*GCF_DIM) << ") "
                     << out[n+p].x << ", " << out[n+p].y 
                     << " vs. " << out_cpu[n+p].x << ", " << out_cpu[n+p].y 
                     << std::endl;
     }
   }
   }
   //std::cout << "free out_cpu" << std::endl;
   //free(out_cpu);out_cpu=NULL;
#endif
#ifdef __MANAGED
   cudaFree(out);out=NULL;
   cudaFree(in);in=NULL;
   cudaFree(in_vals);in_vals=NULL;
   cudaFree(gcf);gcf=NULL;
#else
   free(out);out=NULL;
   free(in);in=NULL;
   free(in_vals);in_vals=NULL;
   free(gcf);gcf=NULL;
#endif
}
