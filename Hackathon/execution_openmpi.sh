#!/bin/sh
mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
OMP_NUM_THREADS=16 mpirun -np 4 ./bruteForce-openmpi senhate