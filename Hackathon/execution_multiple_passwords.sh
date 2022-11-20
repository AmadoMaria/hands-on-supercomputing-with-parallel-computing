#!/bin/sh
dir="results"
if [ ! -d "$dir" ]
then
 mkdir $dir
fi

seq_execution(){

    gcc brute_force_sequential.c -o bruteForce -std=c99 -O3

    seq=$(./bruteForce $1 | grep "seconds" | cut -d " " -f 1)

    echo "seq;time;" >> ./${dir}/seq${1}
    echo "1;${seq};" >> ./${dir}/seq${1}

}

omp(){
    gcc brute_force_openmp.c -o bruteForce-omp -fopenmp -std=c99 -O3

    echo "num_threads;time;" >> ./${dir}/omp${1}
    for j in {2..8..2};
    do
        omp=$(OMP_NUM_THREADS=$j ./bruteForce-omp $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${omp};" >> ./${dir}/omp${1}
    done
}

mpi(){
    mpicc brute_force_mpi.c -o bruteForce-mpi -fopenmp -std=c99 -O3
    echo "num_process;time;" >> ./${dir}/mpi${1}
    for j in {2..6..2};
    do
        mpi=$(mpirun -np $j ./bruteForce-mpi $1 | grep "seconds" | cut -d " " -f 1)
        echo "${j};${mpi};" >> ./${dir}/mpi${1}
    done
}

openmpi(){
    mpicc brute_force_openmpi.c -o bruteForce-openmpi -fopenmp
    echo "num_threads;num_process;time;" >> ./${dir}/openmpi${3}
    openmpi=$(OMP_NUM_THREADS=$1 mpirun -np $2 ./bruteForce-openmpi $3 | grep "seconds" | cut -d " " -f 1)
    echo "${1};${2};${openmpi}" >> ./${dir}/openmpi${3}
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

    best_omp=$(getting_best_value ./${dir}/omp${1})
    best_mpi=$(getting_best_value ./${dir}/mpi${1})
    openmpi $best_omp $best_mpi $1
}

cuda(){

    echo "num_blocks;time;" >> ./${dir}/cuda${1}
    cuda=$(nvcc -arch=sm_70 -o bruteForceGPU $1 ./brute_force_cuda.cu -run | grep "seconds" | cut -d " " -f 1)
    echo "${j};${cuda};" >> ./${dir}/cuda${1}

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

def data_final(dfs, col, title):
    path_table = f"./${dir}/{title}_table.csv"
    path_img = "./${dir}/" + title + ".png"

    data_speed = pd.DataFrame({ 'Password': ["${1}"],
                            'OpenMP': [dfs[0][col].max()],
                            'MPI': [dfs[1][col].max()],
                            'OpenMPI':[ dfs[2][col].max()]
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

omp = pd.read_csv("./${dir}/omp${1}", sep=";")

mpi = pd.read_csv("./${dir}/mpi${1}", sep=";")

openmpi = pd.read_csv("./${dir}/openmpi${1}", sep=";")

seq = pd.read_csv("./${dir}/seq${1}", sep=";", index_col=0)
seq.dropna(axis=1, inplace=True)

seq_value = seq.values[0][0]

subtitles = ['OpenMP', 'MPI']
dfs = []
dfs.append(generate_speedup_table(omp, seq_value, 'num_threads'))
dfs.append(generate_speedup_table(mpi, seq_value, 'num_process'))
generate_speedups_graph(dfs, "./${dir}/speed_up${1}.png", 'Speedups in OpenMP & MPI', subtitles)
dfs.append(generate_speedup_table(openmpi, seq_value, 'num_process'))


data_final(dfs, 'S', 'Speedups')
data_final(dfs, 'time', 'Time')


EOF
python plot_script.py
}

plot_times(){
# names=( $(ls ${dir} | grep $1) )
# export files=["${names[0]}"
# for i in "${names[@]:1}"; 
# do
#     files+=,"$i"
# done
# files+=]
# echo "${files[@]}"
cat <<EOF >plot_time.py
import pandas as pd
import matplotlib.pyplot as plt
import os

print('plotting time for number')
legs = []
fig, ax = plt.subplots()
f = []
all_data = pd.DataFrame()
for path, current, files in os.walk("./${dir}"):
    for file in files:
        if file.startswith("${1}"):
            f.append(file)
for n in range(len(f)):
    df = pd.read_csv(f"${dir}/{f[n]}", sep=";")
    df.dropna(axis=1, inplace=True)
    df.set_index(df.columns[0], inplace=True)
    df.plot(kind='line', ax=ax)

    if n == 0:
        all_data = df.copy()
    else:
        all_data = pd.concat([all_data, df])

    legs.append("${1}")

plt.ylabel('time')
plt.legend(legs)
all_data.to_csv(f'${dir}/time_${1}.csv', sep=";", index=True)
plt.savefig(f'${dir}/time_${1}.png', dpi=200)
plt.close()
EOF
# testar python plot_time.py ${files[@]}
 
python plot_time.py
}

remove_unnecessary_files() {
    echo " "
    echo "[Remove unnecessary files] "
    rm -f *file ./${dir}/omp ./${dir}/mpi ./${dir}/seq bruteForce bruteForce-mpi bruteForce-omp bruteForce-cuda bruteForce-openmpi *.py
    echo " "

}

main(){
    pass=$(
    awk -F ";" '{
        print $1
    }' passwords)
    for p in $pass;
    do
    echo "Executing brute force  algorithm for $p"
    # execution $p
    # plot_script $p
    done
    plot_times "omp"
    plot_times "mpi"
    # remove_unnecessary_files
}

main

