#!/bin/sh
dir="results_"
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
    gcc brute_force_openmp.c -o bruteForce-omp -fopenmp -std=c99 -O3

    echo "num_threads;time;" >> ./${dir}/omp
    for j in {2..8..2};
    do
        # echo $j
        # OMP_NUM_THREADS=$j ./bruteForce-omp $1
        omp=$(OMP_NUM_THREADS=$j ./bruteForce-omp $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${omp};" >> ./${dir}/omp
    done
}

mpi(){
    mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3
    echo "num_process;time;" >> ./${dir}/mpi
    for j in {2..4..2};
    do
        mpi=$(mpirun -np $j ./bruteForce-mpi $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${mpi};" >> ./${dir}/mpi
    done
}

openmpi(){
    mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
    echo "num_threads;num_process;time;" >> ./${dir}/openmpi
    openmpi=$(OMP_NUM_THREADS=$1 mpirun -np $2 ./bruteForce-openmpi $3 | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};${openmpi}" >> ./${dir}/openmpi
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

hybdrid(){

    best_omp=$(getting_best_value ./${dir}/omp)
    best_mpi=$(getting_best_value ./${dir}/mpi)
    openmpi $best_omp $best_mpi $1
}

cuda(){

    echo "num_blocks;time;" >> ./${dir}/cuda
    nvcc brute_force_cuda.cu -o bruteForceGPU -x cu
    cuda=$(./bruteForce-cuda $1 | grep "seconds" | cut -d " " -f 1)
    echo "${j};${cuda};" >> ./${dir}/cuda
}

execution(){
    seq_execution $1
    omp $1
    mpi $1
    hybdrid $1
    # cuda $1
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
    speed_up = pd.DataFrame(columns=[col_name, 'time', 'S'])

    speed_up['time'] = df['time']
    speed_up['S'] = df['time'] / seq_value
    speed_up['S'] = speed_up['S']
    speed_up[col_name] = df[col_name]
    speed_up.set_index(col_name, inplace=True)
    
    return speed_up

def data_final(dfs, col, title, type='max'):
    path_table = f"./${dir}/{title}_table.csv"
    path_img = "./${dir}/" + title + ".png"

    seq = dfs[0][col][0]
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
        print(dt.head())
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
data_final(dfs, 'S', 'Speedup', 'max')
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
    plot_script $1
    remove_unnecessary_files
}

main $1

