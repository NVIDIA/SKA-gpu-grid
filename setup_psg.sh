#! /bin/sh
module load cuda/7.0
module load gcc/4.8.2

export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -ac 3004,875
