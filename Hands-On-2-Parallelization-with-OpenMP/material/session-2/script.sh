#!/bin/sh
for i in $(seq 1 20)
    do
    printf "$i - threads\n"
    OMP_NUM_THREADS=$i	./image "$i"
done