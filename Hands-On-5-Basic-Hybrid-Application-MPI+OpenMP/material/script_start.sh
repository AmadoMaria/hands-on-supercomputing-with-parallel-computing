#!/bin/sh

export LC_NUMERIC="en_US.UTF-8"

set -euo pipefail

clear 

###################################
# FUNCTIONS                       #
###################################

showPropeller() {
   
   tput civis
   
   while [ -d /proc/$! ]
   do
      for i in / - \\ \|
      do
         printf "\033[1D$i"
         sleep .1
      done
   done
   
   tput cnorm
}

create_plot_script_time() {
quant=$1
start=2
steps=2
# files=( {$init $quant $steps } )
# echo "files: $files"
cat <<EOF >time.plt
set title "Execution Time" 
set ylabel "Time (Seconds)"
set xlabel "Size"

set style line 1 lt 2 lc rgb "cyan"   lw 2 
set style line 2 lt 2 lc rgb "red"    lw 2
set style line 3 lt 2 lc rgb "gold"   lw 2
set style line 4 lt 2 lc rgb "green"  lw 2
set style line 5 lt 2 lc rgb "blue"   lw 2
set style line 6 lt 2 lc rgb "black"  lw 2
set terminal postscript eps enhanced color
set output 'time.eps'
set output 'time.png'

set xtics nomirror
set ytics nomirror
set key top left
set key box
set style data lines
plot for [f=2:$quant:2] "file".f using 1:2 title "T=".f with lines 

EOF
}

create_plot_script_speedup() {
cat <<EOF >speedup.plt
set title "Speedup" 
set ylabel "Speedup"
set xlabel "Size"

set style line 1 lt 2 lc rgb "cyan"   lw 2 
set style line 2 lt 2 lc rgb "red"    lw 2
set style line 3 lt 2 lc rgb "gold"   lw 2
set style line 4 lt 2 lc rgb "green"  lw 2
set style line 5 lt 2 lc rgb "blue"   lw 2
set style line 6 lt 2 lc rgb "black"  lw 2
set terminal postscript eps enhanced color
set output 'speedup.eps'
set output 'speedup.png'

set xtics nomirror
set ytics nomirror
set key top left
set key box
set style data lines

plot "file_speedup.data" using 1:2 title "T=2"    ls 1 with linespoints,\
     "file_speedup.data" using 1:3 title "T=3"    ls 2 with linespoints,\
     "file_speedup.data" using 1:4 title "T=4"    ls 3 with linespoints
EOF
}


remove_unnecessary_files() {
echo " "
echo "[Remove unnecessary files] "
rm -f *.txt file* fspeed* *.data mm *.plt
echo " "

sleep 7 > /dev/null 2>&1 &

printf "Loading...\040\040" ; showPropeller
echo " "
echo "[END] " 
echo " "
}

compile_and_execute_openmp() {
num_threads=$1
    # module load gcc/11.1.0
gcc mm-openmp.c -o mm -fopenmp -O3
chmod +x mm

###################################
# Experimental Times              #
###################################

sleep 5 > /dev/null 2>&1 &

printf "Loading...\040\040" ; showPropeller
echo " "

for i in 100 200 300 400 500 600 700 800 900 1000
do
printf "\033[1D$i :" 
for (( j=2; j<=$num_threads; j+=2 ))
do
OMP_NUM_THREADS=$j   ./mm               "$i"    >> "file$j"
done
done

clear 
}

ploting_setup() {
quant=$1
#####################
# PLOTING           #
#####################
echo " "
echo "Do you want to plot a graphic (y/n)?"
read resp

if [[ $resp = "y" ]];then
         echo "ploting eps graphic with gnuplot..."
         create_plot_script_time $quant
        #  create_plot_script_speedup
         gnuplot "time.plt"
        #  gnuplot "speedup.plt"
#rename the files
  mv time.eps  time.$(whoami)@$(hostname)-$(date +%F%T).eps
  mv speedup.eps  speedup.$(whoami)@$(hostname)-$(date +%F%T).eps
  mv time.png  time.$(whoami)@$(hostname)-$(date +%F%T).png
  mv speedup.png  speedup.$(whoami)@$(hostname)-$(date +%F%T).png

fi
}


get_speedups() {
quant=$1
command_files=""
command_print='{print $1'
command_echo="\t[#][size]"
for (( k=2; k<=$quant; k+=2 ))
do
command_files+="file${k} "
command_print+='$'
command_print+=$k
if (( $k + 1 < $quant )); then
command_echo+="\t[T${k}]"
command_print+=',\t'
fi
done
command_print+='}'
echo $command_files
echo $command_print
pr -m -t -s\  ${command_files}  | awk ${command_print} >> file_comparison.data

echo " "

cat -n  file_comparison.data

sleep 3
}

# compile_and_execute_openmp 6
get_speedups 6
ploting_setup 6
# remove_unnecessary_files