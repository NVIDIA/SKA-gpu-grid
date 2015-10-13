PRECISION ?= double
ifeq ($(MANAGED),1)
	USERFLAGS += -D__MANAGED 
endif
ifeq ($(DEBUG),1)
	USERFLAGS += -g -G -lineinfo -D__CPU_CHECK
endif
ifeq ($(GATHER),1)
	USERFLAGS += -D__GATHER
endif
ifeq ($(CPU_CHECK),1)
	USERFLAGS += -D__CPU_CHECK
endif
ifeq ($(MOVING_WINDOW),1)
	USERFLAGS += -D__MOVING_WINDOW
endif
ifeq ($(COMPUTE_GCF),1)
	USERFLAGS += -D__COMPUTE_GCF
endif
ifeq ($(FAST_MATH),1)
	USERFLAGS += -use_fast_math
endif
USERFLAGS += -Xcompiler -fopenmp 

all:  grid GPUGrid.so

clean:
	rm *.o
	rm GPUGrid.so
	rm grid

grid: grid.cu cucommon.cuh grid_gpu.cuh grid_gpu.o Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} $(USERFLAGS) -o grid grid.cu grid_gpu.o

grid_gpu.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o grid_gpu.o grid_gpu.cu

grid_gpu_pic.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h
	nvcc -Xcompiler -fPIC -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o grid_gpu_pic.o grid_gpu.cu

grid-debug: grid.cu grid_gpu-debug.o cucommon.cuh Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} -g -G -lineinfo $(USERFLAGS) -o grid-debug grid_gpu-debug.o grid.cu

grid_gpu-debug.o: grid_gpu.cu grid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 -g -G -lineinfo $(USERFLAGS) -o grid_gpu-debug.o grid_gpu.cu

GPUGrid.so: GPUGrid.cpp grid_gpu_pic.o
	nvcc -std=c++11 -shared -Xcompiler -fPIC -I/usr/include/python2.7/ -lpython2.7 -o GPUGrid.so GPUGrid.cpp  grid_gpu_pic.o
