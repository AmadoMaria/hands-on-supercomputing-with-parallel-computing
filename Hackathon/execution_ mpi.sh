#!/bin/sh
mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp
mpirun -np 4 ./bruteForce-mpi senhate