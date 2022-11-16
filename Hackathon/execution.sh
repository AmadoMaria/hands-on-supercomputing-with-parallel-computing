#!/bin/sh
dir="results"
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
    for j in {2..16..2};
    do
        omp=$(OMP_NUM_THREADS=$j ./bruteForce-omp $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${omp};" >> ./${dir}/omp
    done
}

mpi(){
    mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3
    echo "num_process;time;" >> ./${dir}/mpi
    for j in {2..6..2};
    do
        mpi=$(mpirun -np $j ./bruteForce-mpi $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${mpi};" >> ./${dir}/mpi
    done
}

openmpi(){
    mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
    echo "num_process;time;" >> ./${dir}/openmpi
    openmpi=$(OMP_NUM_THREADS=$1 mpirun -np $2 ./bruteForce-openmpi $3 | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};" >> ./${dir}/openmpi
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

execution(){
    seq_execution $1
    omp $1
    mpi $1
    hybdrid $1
}

# plotting functions
plot_script() {
cat <<EOF >plot_script.py 
import matplotlib.pyplot as plt
import pandas as pd

def generate_plot(df, seq_value, save_path, title, col_name):
    speed_up = pd.DataFrame(columns=[col_name, 'S'])

    speed_up['S'] = df['time'] / seq_value
    speed_up['S'] = speed_up['S']
    speed_up[col_name] = df[col_name]
    speed_up.set_index(col_name, inplace=True)
    
    
    ax = speed_up.plot(kind='line')
    plt.ylabel('Speedup')
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()

omp = pd.read_csv("./${dir}/omp", sep=";")
print(omp.head())

mpi = pd.read_csv("./${dir}/mpi", sep=";")
print(mpi.head())

seq = pd.read_csv("./${dir}/seq", sep=";", index_col=0)
seq.dropna(axis=1, inplace=True)
seq_value = seq.values[0][0]

generate_plot(omp, seq_value, "./${dir}/speed_up_omp.png", 'Speedups in OpenMP', 'num_threads')
generate_plot(mpi, seq_value, "./${dir}/speed_up_mpi.png", 'Speedups in MPI', 'num_process')

EOF
python plot_script.py
}

remove_unnecessary_files() {
    echo " "
    echo "[Remove unnecessary files] "
    rm -f ./${dir}/omp ./${dir}/mpi ./${dir}/seq bruteForce bruteForce-mpi bruteForce-omp bruteForce-cuda bruteForce-openmpi *.py
    echo " "

}

main(){
    execution $1
    plot_script
    remove_unnecessary_files
}

main $1

