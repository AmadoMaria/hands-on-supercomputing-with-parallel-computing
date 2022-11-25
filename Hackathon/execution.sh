#!/bin/sh

#SBATCH --job-name=553072                         # Job name
#SBATCH --nodes=1                              # Run all processes on 2 nodes
#SBATCH --partition=gpushortc                   # Partition OGBON
#SBATCH --output=out_%j.log                    # Standard output and error log
#SBATCH --ntasks-per-node=1                    # 1 job per node
#SBATCH --account=treinamento                   # Account of the group

module load openmpi/4.1.1-cuda-11.6-ofed-5.4

dir="results_${1}"
if [ ! -d "$dir" ]
then
 mkdir $dir
fi

seq_execution(){

    gcc brute_force_sequential.c -o bruteForce -std=c99 -O3

    seq=$(./bruteForce $1 | grep "seconds" | cut -d " " -f 1)

    echo "seq;time;" >> ./${dir}/seq
    echo "1;${seq};" >> ./${dir}/seq

}

omp(){
    echo "          Compiling OpenMP paralelization..."
    gcc brute_force_openmp.c -o bruteForce-omp -fopenmp -std=c99 -O3
    echo "          Executing OpenMP..."
    echo "num_threads;time;" >> ./${dir}/omp
    #for j in {2..4..2};
    for ((j=2; j<=2048; j*=2 ));
    do
    
        # echo $j
        # OMP_NUM_THREADS=$j ./bruteForce-omp $1
        omp=$(OMP_NUM_THREADS=$j ./bruteForce-omp $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${omp};" >> ./${dir}/omp
    done
}

mpi(){
    
    # nvcc -I/opt/share/openmpi/4.1.1-cuda/include -L/opt/share/openmpi/4.1.1-cuda/lib64 -DprintLabel -lnccl -lmpi -Xcompiler -fopenmp -o bruteForce-mpi brute_force_mpi.c
    echo "          Compiling MPI paralelization"
    mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3
    echo "          Executing MPI..."
    echo "num_process;time;" >> ./${dir}/mpi
    for j in {2..32..2};
    do
        mpirun -x MXM_LOG_LEVEL=error -np $j --allow-run-as-root ./bruteForce-mpi $1 2>/dev/null > output_mpi
        mpi=$(cat output_mpi | grep "seconds" | cut -d " " -f 1)
        # mpirun -x MXM_LOG_LEVEL=error -np 32 --allow-run-as-root ./bruteForce-mpi senha 2>/dev/null
        # mpi=$(mpirun -np $j ./bruteForce-mpi $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${mpi};" >> ./${dir}/mpi
    done
}

openmpi(){
    echo "          Compiling OpenMPI paralelization"
    mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
    echo "          Executing OpenMPI"
    echo "num_threads;num_process;time;" >> ./${dir}/openmpi
    best_mpi=$1
    best_omp=$2
    process=$best_mpi-6
    max=$best_mpi+6
    thread=$best_omp/8
    max_t=$best_omp*8

    for ((j=$process; j <= $max && j<34; j+=2));
    do
        for ((i=$thread; i <=$max_t && i<128; i*=2))
        do
            openmpi=$(OMP_NUM_THREADS=$i mpirun -x MXM_LOG_LEVEL=error -np $j ./bruteForce-openmpi $1 2>/dev/null | grep "seconds" | cut -d " " -f 1)
            echo "${1};${2};${openmpi}" >> ./${dir}/openmpi
        done
    done
    
}

getting_best_value(){
    best=$(awk -F ";" 'BEGIN{
        min=1000;
        th=1;
        }
        {
            if(min > $2){
                min=$2; 
                th=$1;
            }
        }
        END{
            print th;
            }' $1
            )
    echo $best
}

hybrid(){

    best_omp=$(getting_best_value ./${dir}/omp)
    best_mpi=$(getting_best_value ./${dir}/mpi)

    openmpi $best_omp $best_mpi $1    
}

cuda(){
    echo "Compiling GPU CUDA paralelization"
    echo "num_blocks;time;" >> ./${dir}/cuda
    nvcc brute_force_cuda.cu -o bruteForceGPU -x cu
    echo "Executing CUDA..."
    cuda=$(./bruteForceGPU $1 | grep "seconds" | cut -d " " -f 1)
    echo "${j};${cuda};" >> ./${dir}/cuda
}

execution(){
    # seq_execution $1
    # omp $1
    mpi $1
    # hybrid $1
    cuda $1
}

# plotting functions
plot_script() {
cat <<EOF >plot_script.py 
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_time_exec_graph(df_, col, save_path, title, subtitles):
    df = df_.copy(deep=True)
    df.plot(kind='line', title=title, legend=None)

    plt.xticks(df.index.values.tolist(), df.index.values.tolist())
    plt.xlabel(col)
    plt.ylabel('Time execution')
    plt.savefig(save_path, dpi=200)
    plt.close()

def generate_speedups_graph(dfs, save_path, title, subtitles):
    fig, ax = plt.subplots()
    for d in range(len(dfs)):
        dfs[d].plot(kind='line', ax=ax)

    plt.legend(subtitles)
    plt.xlabel('Threads/Process')
    plt.ylabel('Speedup')
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_speedup_table(df, seq_value,  col_name):
    speed_up = pd.DataFrame({col_name: [1], 'time': [seq_value]})

    if col_name != 'seq':
        speed_up = pd.concat([speed_up, df])
        speed_up['time'] = df['time']

    speed_up['S'] = df['time'] / seq_value
    speed_up['S'] = speed_up['S']
    speed_up.set_index(col_name, inplace=True)

    return speed_up

def data_final(dfs, col, title, type='max'):
    path_table = f"./${dir}/{title}_table.csv"
    path_img = "./${dir}/" + title + ".png"

    seq = dfs[0][col].values[0]
    if type == "max":
        omp = dfs[1][col].max()
        mpi = dfs[2][col].max()
        openMPI = dfs[3][col].max()
    else:
        omp = dfs[1][col].min()
        mpi = dfs[2][col].min()
        openMPI = dfs[3][col].min()
    data_speed = pd.DataFrame({ 'Password': ["${1}"],
                                'Sequential': [seq],
                                'OpenMP': [omp],
                                'MPI': [mpi],
                                'OpenMPI':[openMPI]
                                }
                                )    
    if os.path.exists(path_table):
        dt = pd.read_csv(path_table, sep=";")

        data_speed = pd.concat([dt, data_speed])
        print(data_speed.head())

    data_speed.to_csv(path_table, sep=";", index=False)
    data_speed.set_index('Password', inplace=True)
    data_speed.plot(kind='bar', rot=0, title=title, width=0.35)
    plt.ylabel(title.lower())
    plt.savefig(path_img, dpi=200)
    plt.close()


omp = pd.read_csv("./${dir}/omp", sep=";")

mpi = pd.read_csv("./${dir}/mpi", sep=";")

openmpi = pd.read_csv("./${dir}/openmpi", sep=";")

seq = pd.read_csv("./${dir}/seq", sep=";")
seq.dropna(axis=1, inplace=True)

seq_value = seq['time'][0]

subtitles = ['OpenMP', 'MPI']
dfs = []
dfs.append(generate_speedup_table(seq, seq_value, 'seq'))
dfs.append(generate_speedup_table(omp, seq_value, 'num_threads'))
dfs.append(generate_speedup_table(mpi, seq_value, 'num_process'))

generate_time_exec_graph(dfs[1], 'num_threads', "./${dir}/omp_time.png", 'Time execution by threads', 'threads')
generate_time_exec_graph(dfs[2], 'num_process', "./${dir}/mpi_time.png", 'Time execution by process', 'process')

generate_speedups_graph(dfs, "./${dir}/speed_up.png", 'Speedups in OpenMP & MPI', subtitles)
dfs.append(generate_speedup_table(openmpi, seq_value, 'num_process'))

# the row that contains the best speedup gotten, its the same which had the minor time
data_final(dfs, 'S', 'Speedups', 'max')
data_final(dfs, 'time', 'Time', 'min')


EOF
python plot_script.py
}

remove_unnecessary_files() {
    echo " "
    echo "[Remove unnecessary files] "
    rm -f *file ./${dir}/omp ./${dir}/mpi ./${dir}/seq bruteForce bruteForce-mpi bruteForce-omp bruteForce-cuda bruteForce-openmpi *.py
    echo " "

}

main(){
    execution $1
    # echo "Plotting graphs..."
    # plot_script $1
    # remove_unnecessary_files
}

main $1

