#ifndef __GRID_CUH
#define __GRID_CUH
template <class CmplxType>
void gridGPU(CmplxType* out, CmplxType* in, size_t npts, CmplxType *img, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim); 
#endif

