#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

// 97 to 122 use only lowercase letters
// 65 to 90 use only capital letters
// 48 to 57 use only numbers

#define START_CHAR 97
#define END_CHAR 122
#define MAXIMUM_PASSWORD 20
#define MAX_THREADS_PER_BLOCK 1024

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__device__ long long my_pow(long long x, int y)
{
    long long res = 1;
    if (y == 0)
        return res;
    else
        return x * my_pow(x, y - 1);
}

__device__ int my_strlen(char *s)
{
    int sum = 0;
    while (*s++)
        sum++;
    return sum;
}

__global__ void bruteForce(char *pass)
{
    int pass_b26[MAXIMUM_PASSWORD];

    long long int pass_decimal = 0;
    int base = END_CHAR - START_CHAR + 2;

    int size = my_strlen(pass);
    ;

    for (int i = 0; i < size; i++)
        pass_b26[i] = (int)pass[i] - START_CHAR + 1;

    for (int i = size - 1; i > -1; i--)
        pass_decimal += (long long int)pass_b26[i] * my_pow(base, i);

    long long int max = my_pow(base, size);
    char s[MAXIMUM_PASSWORD];
    long long int j = blockIdx.x * blockDim.x + threadIdx.x;

    while (j < max)
    {
        if (j == pass_decimal)
        {
            printf("Found password!\n");
            int index = 0;

            printf("Password in decimal base: %lli\n", j);
            while (j > 0)
            {
                s[index++] = 'a' + j % base - 1;
                j /= base;
            }
            s[index] = '\0';

            printf("Found password: %s\n", s);
            break;
        }
        j += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv)
{
    char password[MAXIMUM_PASSWORD], *password_d;
    time_t t1, t2;
    double dif;

    strcpy(password, argv[1]);

    cudaMalloc(&password_d, MAXIMUM_PASSWORD * sizeof(char));
    cudaMemcpy(password_d, password, MAXIMUM_PASSWORD * sizeof(char), cudaMemcpyHostToDevice);

    int deviceId, numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int number_of_blocks = numberOfSMs * 32;
    int threads_per_block = 1024;

    printf("Try to broke the password: %s\n", password);

    time(&t1);
    bruteForce<<<number_of_blocks, threads_per_block>>>(password_d);
    checkCuda(cudaDeviceSynchronize());
    time(&t2);

    dif = difftime(t2, t1);

    printf("\n%1.2f seconds\n", dif);

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(password_d);

    return 0;
}