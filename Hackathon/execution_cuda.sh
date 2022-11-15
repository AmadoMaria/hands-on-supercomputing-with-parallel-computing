#!/bin/sh
nvcc brute_force_cuda.cu -o bruteForce-cuda -x cu
./bruteForce-cuda senhate