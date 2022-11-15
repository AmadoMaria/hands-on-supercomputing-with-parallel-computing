#!/bin/sh
mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3
mpirun --allow-run-as-root -np 4 ./bruteForce-mpi senhate