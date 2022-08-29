#!/bin/sh
echo -n "" > time.txt
for i in $(seq 1 30)
    do
    printf "$i - threads\n"
    OMP_NUM_THREADS=$i	./image "$i"
done

gnuplot "plot.p"