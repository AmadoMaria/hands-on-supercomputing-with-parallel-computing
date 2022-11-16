#!/bin/sh

nvcc -arch=sm_70 -o heat 01-heat-conduction.cu -run

nvcc -arch=sm_70 -o heat_gpu 01-heat-conduction_gpu.cu -run 