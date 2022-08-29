#!/bin/sh
for j in $(seq 1 10)
do
 printf "threads:$j :\n"
 for i in 100 200 300 400 500 600 700 800 900 1000
  do
   printf "\033[1D$i :"
   OMP_NUM_THREADS=$j	./mm "$i"
  done
done