/*
File:           count_primes.c
Last changed:   20220303 11:35:00
Purpose:        Parallelize counting of prime numbers using openMP
Author:         Murilo Boratto - muriloboratto@uneb.br
Usage:
HowToCompile:   gcc count_primes.c -o count_primes -fopenmp -lm
HowToExecute:   OMP_NUM_THREADS=${num_threads} ./count_primes
                OMP_NUM_THREADS=4              ./count_primes
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef unsigned long long big_integer;
#define TOP_LIMIT 20000000ULL

int is_prime(big_integer n)
{
  int p;
  big_integer i, s;

  p = (n % 2 != 0 || n == 2);

  if (p)
  {
    s = sqrt(n);
    
    #pragma omp parallel private(i)
    for (i = 3; p && i <= s; i += 2)
      if (n % i == 0)
        p = 0;
  }

  return p;
}

int main(int argc, char **argv)
{
  big_integer i, primes = 2;
  float t1, t2;
  int max;
  FILE *fp;

  fp = fopen("time.txt", "a");
  max = omp_get_max_threads();
  t1 = omp_get_wtime();
  #pragma omp parallel for
  for (i = 3; i <= TOP_LIMIT; i += 2)
    if (is_prime(i))
      primes++;
  
  t2 = omp_get_wtime();

  printf("%llu\n", primes);
  fprintf(fp, "%d\t%f\n", max, t2-t1);
  printf("time execution: %f\n", t2-t1);
  

  fclose(fp);

  return 0;
}
