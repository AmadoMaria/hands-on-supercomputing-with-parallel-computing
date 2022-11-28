#!/bin/sh
dir="results_multiple"
if [ ! -d "$dir" ]
then
 mkdir $dir
fi

creating_files(){
    echo "password;seq;time;" >> ./${dir}/seq
    echo "password;num_threads;time;" >> ./${dir}/omp
    echo "password;num_process;time;" >> ./${dir}/mpi
    echo "password;num_threads;num_process;time;" >> ./${dir}/openmpi
    echo "password;num_blocks;time;" >> ./${dir}/cuda
}

seq_execution(){

    seq=$(./bruteForce $1 | grep "seconds" | cut -d " " -f 1)

    
    echo "${1};1;${seq};" >> ./${dir}/seq${1}

}

omp_execution(){

    omp=$(OMP_NUM_THREADS=$2 ./bruteForce-omp $1 | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};${omp};" >> ./${dir}/omp

}

mpi_execution(){
    mpirun -x MXM_LOG_LEVEL=error -np $2 --allow-run-as-root ./bruteForce-mpi $1 2>/dev/null > output_mpi
    mpi=$(cat output_mpi | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};${mpi};" >> ./${dir}/mpi    
}

openmpi_execution(){
    OMP_NUM_THREADS=$2 mpirun -x MXM_LOG_LEVEL=error -np $3 ./bruteForce-openmpi $1 2>/dev/null > output_openmpi
    ompi=$(cat output_openmpi | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};${3};${ompi}" >> ./${dir}/openmpi
}

cuda_execution(){
    cuda=$(./bruteForceGPU $1 | grep "seconds" | cut -d " " -f 1)
    echo "${1};32;${cuda};" >> ./${dir}/cuda
}

main(){
    creating_files
    pass=$(
    awk -F ";" '{
        print $1
    }' passwords)
    for p in $pass;
    do
    echo "Executing brute force algorithm for $p"
    # cuda_execution $p
    mpi_execution $p 30
    # omp_execution $p 32
    openmpi_execution $p 4 22
    # seq_execution $p
    
    done
}

main