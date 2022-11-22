#!/bin/sh

#SBATCH --job-name=553072                         # Job name
#SBATCH --nodes=1                              # Run all processes on 2 nodes
#SBATCH --partition=gpushortc                   # Partition OGBON
#SBATCH --output=out_%j.log                    # Standard output and error log
#SBATCH --ntasks-per-node=1                    # 1 job per node
#SBATCH --account=treinamento                   # Account of the group


module load openmpi/4.1.1-cuda-11.6-ofed-5.4

mpirun -np 16 -x UCX_MEMTYPE_CACHE=n  -mca pml ucx -mca btl ^vader,tcp,openib,smcuda -x UCX_NET_DEVICES=mlx5_0:1  ./bruteForce-mpi $1
