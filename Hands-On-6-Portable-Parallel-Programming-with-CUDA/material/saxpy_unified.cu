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
    for (int i = 0; i < n; ++i)
        printf("%1.0f\t", vector[i]);
    printf("\n\n");
}
void generateVector(float *vector, int n)
{
    for (int i = 0; i < n; ++i)
        vector[i] = i + 1;
}
int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    float *x, *y;
    cudaMallocManaged(&x, sizeof(float) * n);
    cudaMallocManaged(&y, sizeof(float) * n);
    generateVector(x, n);
    printVector(x, n);
    generateVector(y, n);
    printVector(y, n);
    int NUMBER_OF_BLOCKS = 1;
    int NUMBER_OF_THREADS_PER_BLOCK = n;
    saxpy<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>(n, x, y);
    cudaDeviceSynchronize();
    printVector(y, n);
    cudaFree(x);
    cudaFree(y);
    return 0;
}