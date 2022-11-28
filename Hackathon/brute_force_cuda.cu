#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

// 97 to 122 use only lowercase letters
// 65 to 90 use only capital letters
// 48 to 57 use only numbers

#define START_CHAR 48
#define END_CHAR 122
#define MAXIMUM_PASSWORD 20

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

__global__ void bruteForce(char *pass, long long size)
{
    char force[MAXIMUM_PASSWORD];
    int palavra[MAXIMUM_PASSWORD];
    int pass_b26[MAXIMUM_PASSWORD];

    long long int pass_decimal = 0;
    int base = END_CHAR - START_CHAR + 2;

    for (int i = 0; i < MAXIMUM_PASSWORD; i++)
        force[i] = '\0';

    for (int i = 0; i < size; i++)
        pass_b26[i] = (int)pass[i] - START_CHAR + 1;

    for (int i = size - 1; i > -1; i--)
    {
        pass_decimal += (long long int)pass_b26[i] * my_pow(base, i);
    }

    long long max = my_pow(base, size);

    long long start = threadIdx.x + blockIdx.x * blockDim.x;
    for (long long idx = start; idx < max; idx += gridDim.x * blockDim.x)
    {
        if (idx > pass_decimal)
            return;
        if (idx == pass_decimal)
        {
            int index = 0;
            char s[MAXIMUM_PASSWORD];

            printf("Password in decimal base: %lli\n", idx);
            while ((idx) > 0)
            {
                s[index++] = START_CHAR + idx % base - 1;
                idx /= base;
            }
            s[index] = '\0';
            printf("Found password: %s\n", s);
            return;
        }
    }
}

int main(int argc, char **argv)
{
    char *password;
    time_t t1, t2;
    double dif;

    checkCuda(cudaMallocManaged(&password, sizeof(char) * MAXIMUM_PASSWORD));

    cudaError_t syncErr, asyncErr;

    strcpy(password, argv[1]);
    int size = strlen(password);

    int deviceId, numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int number_of_blocks = numberOfSMs * 32;
    int threads_per_block = 1024;

    printf("Try to broke the password: %s\n", password);

    time(&t1);
    bruteForce<<<number_of_blocks, threads_per_block>>>(password, size);
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    time(&t2);

    if (syncErr != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(asyncErr));

    dif = difftime(t2, t1);

    printf("\n%1.20f seconds\n", dif);
    cudaFree(password);

    return 0;
}