/****************************************************************************80
 *  Code:
 *   saxpy.c
 *
 *  Purpose:
 *   Implements in C the simple SAXPY stands for Single-Precision.
 *
 *  Modified:
 *   May 08 2022 17:28
 *
 *  Author:
 *    Murilo Boratto  < muriloboratto 'at' fieb.org.br >
 *
 *  HowtoCompile:
 *    gcc saxpy.c -o saxpy
 *
 *  HowtoExecute:
 *    ./saxpy <size>
 *    ./saxpy  10
 *
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void saxpy(int n, float *x, float *y)
{

  int i = threadIdx.x;
  if (i < n)
    y[i] = x[i] + y[i];
}

void printVector(float *vector, int n)
{

  int i;
  for (i = 0; i < n; ++i)
    printf("%1.0f\t", vector[i]);

  printf("\n\n");
}

void generateVector(float *vector, int n)
{
  int i;
  for (i = 0; i < n; ++i)
    vector[i] = i + 1;
}

int main(int argc, char *argv[])
{
  int n = atoi(argv[1]);
  float *x, *y;
  float *xd, *yd;

  cudaMalloc((void **)&xd, sizeof(float) * n);
  cudaMalloc((void **)&yd, sizeof(float) * n);

  x = (float *)malloc(sizeof(float) * n);
  y = (float *)malloc(sizeof(float) * n);

  generateVector(x, n);
  printVector(x, n);

  generateVector(y, n);
  printVector(y, n);

  cudaMemcpy(xd, x, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(yd, x, sizeof(float) * n, cudaMemcpyHostToDevice);

  int NUMBER_OF_BLOCKS = 1;
  int NUMBER_OF_THREADS_PER_BLOCKS = n;

  saxpy<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCKS>>>(n, xd, yd);

  cudaMemcpy(y, yd, sizeof(float) * (n), cudaMemcpyDeviceToHost);
  printVector(y, n);

  free(x);
  free(y);
  cudaFree(xd);
  cudaFree(yd);

  return 0;
}
