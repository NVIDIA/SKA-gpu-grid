#ifndef __GRID_CUH
#define __GRID_CUH
template <class CmplxType>
void gridGPU(CmplxType* out, CmplxType* in, CmplxType* in_vals, size_t npts, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim); 
#endif

