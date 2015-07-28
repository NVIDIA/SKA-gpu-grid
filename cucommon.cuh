#include <iostream>
#ifndef __CUCOMMON_CUH
#define __CUCOMMON_CUH
void CUDA_CHECK_ERR(unsigned lineNumber, const char* fileName);
float getElapsed(cudaEvent_t start, cudaEvent_t stop);

//typedef struct {float x,y;} float2;
//typedef struct {double x,y;} double2;


#endif
