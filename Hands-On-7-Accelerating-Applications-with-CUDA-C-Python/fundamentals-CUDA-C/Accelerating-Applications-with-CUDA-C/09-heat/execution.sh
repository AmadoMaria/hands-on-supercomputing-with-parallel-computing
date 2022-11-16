#!/bin/sh

nvcc -arch=sm_70 -o heat 01-heat-condution.cu -run

nvcc -arch=sm_70 -o heat_gpu 01-heat-condution_gpu.cu -run 