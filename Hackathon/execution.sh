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
        mpirun -x MXM_LOG_LEVEL=error -np $j --allow-run-as-root ./bruteForce-mpi $1 2>/dev/null > output_mpi_
        mpi=$(cat output_mpi_ | grep "seconds" | cut -d " " -f 1)
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
    best_mpi=$2
    best_omp=$1
    process=30
    max=$best_mpi+6
    thread=$best_omp/8
    max_t=$best_omp*8
    
    for ((j=$process; j > 0 && j <= $max && j <= 32; j+=2));
    do
        for ((i=$thread; i <= $max_t && i <= 128 && (i*j) < 3840; i*=2))
        do
            echo "Running with $i threads and $j proccess"
            OMP_NUM_THREADS=$i mpirun -x MXM_LOG_LEVEL=error -np $j ./bruteForce-openmpi $3 2>/dev/null > output_openmpi_
            ompi=$(cat output_openmpi_ | grep "seconds" | cut -d " " -f 1)
            echo "${i};${j};${ompi}" >> ./${dir}/openmpi
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
    hybrid $1
    # cuda $1
}

# plotting functions
plot_script() {
cat <<EOF >plot_script.py 
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def generate_time_exec_graph(df_, col, save_path, title, subtitles):
    df = df_.copy(deep=True)
    x = df.index.values.tolist()
    y = df['time']
    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.ylabel('Time execution')
    plt.xlabel(subtitles)
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()

def generate_speedups_graph(df, save_path, title, subtitles):
    x = df.index.values.tolist()
    y = df['S']
    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)

    plt.xlabel(subtitles)
    plt.ylabel('Speedup')
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_speedup_table(df_, seq_value,  col_name=None):
    df = df_.copy(deep=True)
    if col_name is not None:
        if col_name == 'all':
            name = 'process and threads'
            values = ['1 & 1']
            df_[name] = df['num_threads'].astype(str) + ' & ' + df['num_process'].astype(str)
        else:
            name = col_name
            values = [1]
        speed_up = pd.DataFrame({name: values, 'time': [seq_value]})
    else:
        speed_up = pd.DataFrame({'time': [seq_value]})
    
    if col_name != 'seq':
        if col_name is not None:
            speed_up = pd.concat([speed_up, df])
        speed_up['time'] = df['time']

    speed_up['S'] = seq_value / df['time']
    speed_up['S'] = speed_up['S']
    if col_name is not None:
        speed_up.set_index(name, inplace=True)

    return speed_up

def data_final(dfs, col, title, type='max'):
    path_table = f"./${dir}/{title}_table.csv"
    path_img = "./${dir}/" + title + ".png"

    seq = dfs[0][col].values[0]
    if type == "max":
        omp = dfs[1][col].max()
        mpi = dfs[2][col].max()
        openMPI = dfs[3][col].max()
        cuda = dfs[4][col].max()
    else:
        omp = dfs[1][col].min()
        mpi = dfs[2][col].min()
        openMPI = dfs[3][col].min()
        cuda = dfs[4][col].min()
    data_speed = pd.DataFrame({ 'Password': ["_Hacka1"],
                                'Sequential': [seq],
                                'OpenMP': [omp],
                                'MPI': [mpi],
                                # 'OpenMPI':[openMPI],
                                'Cuda': [cuda]
                                }
                                )    
    if os.path.exists(path_table):
        dt = pd.read_csv(path_table, sep=";")

        data_speed = pd.concat([dt, data_speed])
        print(data_speed.head())

    data_speed.to_csv(path_table, sep=";", index=False)
    data_speed.set_index('Password', inplace=True)
    
    # columns = data_speed.columns.tolist()
    # step = -0.2
    # width=0.1
    # x = np.arange(data_speed.shape[0])

    # colors = ['blue', 'purple', 'orange', 'red', 'green']
    # y_ticks = []
    # for v in x:
    #     y_ticks.extend(data_speed.iloc[0].values)
    # y_ticks.sort()
    # for c in range(len(columns)):
    #     y = data_speed[columns[c]]
    #     plt.bar(x+step, y, width=width, color=colors[c])
    #     step += 0.2
    # plt.xticks(x, data_speed.index.values)
    # plt.yticks(y_ticks)

    ax = data_speed.plot(kind='bar', rot=0, title=title, width=0.35)

    for p in ax.patches:
        height = p.get_height()
        if type == 'max':
            text = f'{round(height)}x'
        else:
            text = f'{round(height)}'
        ax.text(p.get_x() + p.get_width()/2, height, text, fontsize=10,
        color='black', ha='center', va='bottom')
    
    plt.ylabel(title.lower())
    plt.savefig(path_img, dpi=200)
    plt.close()


omp = pd.read_csv("./${dir}/omp", sep=";")

mpi = pd.read_csv("./${dir}/mpi", sep=";")

openmpi = pd.read_csv("./${dir}/openmpi", sep=";")
cuda = pd.read_csv("./${dir}/cuda", sep=";")
cuda.dropna(axis=1, inplace=True)

seq = pd.read_csv("./${dir}/seq", sep=";")
seq.dropna(axis=1, inplace=True)

seq_value = seq['time'][0]

subtitles = ['OpenMP', 'MPI', 'OpenMPI', 'Cuda']
dfs = []
dfs.append(generate_speedup_table(seq, seq_value, 'seq'))
dfs.append(generate_speedup_table(omp, seq_value, 'num_threads'))
dfs.append(generate_speedup_table(mpi, seq_value, 'num_process'))
dfs.append(generate_speedup_table(openmpi, seq_value, 'all'))
dfs.append(generate_speedup_table(cuda, seq_value))

generate_time_exec_graph(dfs[1], 'num_threads', "./${dir}/omp_time.png", 'Time execution by threads', 'threads')
generate_time_exec_graph(dfs[2], 'num_process', "./${dir}/mpi_time.png", 'Time execution by process', 'process')
generate_time_exec_graph(dfs[3], 'process and threads', "./${dir}/openmpi_time.png", 'Time execution by process and threads', 'process & threads')

generate_speedups_graph(dfs, "./${dir}/speed_up.png", 'Speedups in OpenMP & MPI', subtitles)
generate_speedups_graph(dfs[2], "./${dir}/speed_up_mpi.png", 'Speedups in MPI', subtitles[1])
generate_speedups_graph(dfs[3], "./${dir}/speed_up_openmpi.png", 'Speedups in OpenMPI', subtitles[2])
dfs.append(generate_speedup_table(cuda, seq_value, 'num_blocks'))

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

