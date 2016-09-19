# GPUGrid
Gridding for the GPU

Build options (more parameter options in Defines.h):

PRECISION=double,float - double or single precision (default: double)

MANAGED=0,1            - Turns on managed memory

DEBUG=0,1              - Adds debug flags for both GPU and CPU

CPU_CHECK=0,1          - Checks results against a CPU-only method

COMPUTE_GCF=0,1        - Computes a simple GCF on the fly, rather
                           then load from memory

GATHER=0,1             - "Gather" method in which each thread gathers
                           contributions to a single image point

MOVING_WINDOW=0,1      - "Moving window" approach.

FILE_INPUT=0,1         - Rather than generate random input, read from 
                           the file called UVW_in.dat

FAST_MATH=0,1          - Use fast math intrinsics
