#!/bin/sh

#SBATCH --job-name=553072                         # Job name
#SBATCH --nodes=1                              # Run all processes on 2 nodes
#SBATCH --partition=gpushortc                   # Partition OGBON
#SBATCH --output=out_%j.log                    # Standard output and error log
#SBATCH --ntasks-per-node=1                    # 1 job per node
#SBATCH --account=treinamento                   # Account of the group

module load openmpi/4.1.1-cuda-11.6-ofed-5.4

nvcc -I/opt/share/openmpi/4.1.1-cuda/include -L/opt/share/openmpi/4.1.1-cuda/lib64 -DprintLabel -lnccl -lmpi -Xcompiler -fopenmp -o bruteForce-mpi brute_force_mpi.c

