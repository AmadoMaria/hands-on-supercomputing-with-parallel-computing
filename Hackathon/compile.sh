#!/bin/sh

#SBATCH --job-name=compile_files                        # Job name
#SBATCH --nodes=1                              # Run all processes on 2 nodes
#SBATCH --partition=gpushortc                   # Partition OGBON
#SBATCH --output=out_%j.log                    # Standard output and error log
#SBATCH --ntasks-per-node=1                    # 1 job per node
#SBATCH --account=interact                   # Account of the group

module load openmpi/4.1.1-cuda-11.6-ofed-5.4

gcc brute_force_sequential.c -o bruteForce -std=c99 -O3

gcc brute_force_openmp.c -o bruteForce-omp -fopenmp -std=c99 -O3

mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3

mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
    
nvcc brute_force_cuda.cu -o bruteForceGPU -x cu