#!/bin/sh
# echo -n "" > time.txt #cleaning .txt file
for i in $(seq 1 30)
    do
    printf "$i - threads\n"
    OMP_NUM_THREADS=$i	./count_primes "$i"
done