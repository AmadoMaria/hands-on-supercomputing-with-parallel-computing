#!/bin/sh
for j in $(seq 1 6)
do 
 printf "threads:$j :\n"
 for i in {1000..10000..1000}
 do
  printf "$i - "
   OMP_NUM_THREADS=$j	./integral "$i"
  done
done