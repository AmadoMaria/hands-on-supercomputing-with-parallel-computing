#!/bin/sh
gcc brute_force_openmp.c -o bruteForce-omp -fopenmp
OMP_NUM_THREADS=16 ./bruteForce-omp senhate