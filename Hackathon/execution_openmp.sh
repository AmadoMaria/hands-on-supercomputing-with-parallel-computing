#!/bin/sh
gcc brute_force_openmp.c -o bruteForce-omp -fopenmp -std=c99 -O3
OMP_NUM_THREADS=16 ./bruteForce-omp senhate