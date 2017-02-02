# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of NVIDIA CORPORATION nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
HDF5_HOME = /usr/local/hdf5
ARCH ?= sm_60
PRECISION ?= double
OUTPRECISION ?= double
ifeq ($(MANAGED),1)
	CFLAGS += -D__MANAGED 
endif
ifeq ($(DEBUG),1)
	CFLAGS += -g -G -lineinfo -D__CPU_CHECK
endif
ifeq ($(GATHER),1)
	CFLAGS += -D__GATHER
endif
ifeq ($(CPU_CHECK),1)
	CFLAGS += -D__CPU_CHECK
endif
ifeq ($(MOVING_WINDOW),1)
	CFLAGS += -D__MOVING_WINDOW
endif
ifeq ($(COMPUTE_GCF),1)
	CFLAGS += -D__COMPUTE_GCF
endif
ifeq ($(FILE_INPUT),1)
	CFLAGS += -D__FILE_INPUT
endif
ifeq ($(CAS_ATOMIC),1)
	CFLAGS += -D__CASATOMIC
endif
ifeq ($(FAST_MATH),1)
	CFLAGS += -use_fast_math
endif
ifeq ($(HDF5_INPUT),1)
	CFLAGS += -D__HDF5_INPUT -I$(HDF5_HOME)/include -L$(HDF5_HOME)/lib -lhdf5_cpp -lhdf5
        OBJS += hdf52struct.o
endif
CFLAGS += -Xcompiler -fopenmp -Xptxas -v,-abi=no
CFLAGS += $(USERFLAGS)

all:  grid GPUGrid.so

clean:
	rm *.o
	rm GPUGrid.so
	rm grid

grid: grid.cu cucommon.cuh grid_gpu.cuh grid_gpu.o Defines.h $(OBJS)
	nvcc -arch=${ARCH} -std=c++11 -DPRECISION=${PRECISION} -DOUTPRECISION=${OUTPRECISION} $(CFLAGS) -o grid grid.cu grid_gpu.o $(OBJS)

grid_gpu.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=${ARCH} -std=c++11 $(CFLAGS) -o grid_gpu.o grid_gpu.cu

grid_gpu_pic.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h 
	nvcc -Xcompiler -fPIC -c -arch=${ARCH} -std=c++11 $(CFLAGS) -o grid_gpu_pic.o grid_gpu.cu 

grid-debug: grid.cu grid_gpu-debug.o cucommon.cuh Defines.h $(OBJS)
	nvcc -arch=${ARCH} -std=c++11 -DPRECISION=${PRECISION} -DOUTPRECISION=${OUTPRECISION} -g -G -lineinfo $(CFLAGS) -o grid-debug grid_gpu-debug.o grid.cu $(OBJS)

grid_gpu-debug.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h 
	nvcc -c -arch=${ARCH} -std=c++11 -g -G -lineinfo $(CFLAGS) -o grid_gpu-debug.o grid_gpu.cu 

GPUGrid.so: GPUGrid.cpp grid_gpu_pic.o $(OBJS:.o=_pic.o)
	nvcc -std=c++11 -shared -Xcompiler -fPIC -I/usr/include/python2.7/ -lpython2.7 -o GPUGrid.so GPUGrid.cpp  grid_gpu_pic.o $(OBJS:.o=_pic.o)

hdf52struct.o: hdf52struct.cpp vis.h
	nvcc -c -o hdf52struct.o hdf52struct.cpp $(CFLAGS)

hdf52struct_pic.o: hdf52struct.cpp vis.h
	nvcc -Xcompiler -fPIC -c -o hdf52struct_pic.o hdf52struct.cpp $(CFLAGS)

