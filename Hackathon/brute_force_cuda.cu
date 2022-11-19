#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// 97 to 122 use only lowercase letters
// 65 to 90 use only capital letters
// 48 to 57 use only numbers

#define START_CHAR 97
#define END_CHAR 122
#define MAXIMUM_PASSWORD 20

__device__ long long my_pow(long long x, int y)
{
    long long res = 1;
    if (y == 0)
        return res;
    else
        return x * my_pow(x, y - 1);
}

long long my_pow_host(long long x, int y)
{
    long long res = 1;
    if (y == 0)
        return res;
    else
        return x * my_pow_host(x, y - 1);
}

__global__ void bruteForce(char *pass, long long size, long long *result)
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
        if (idx >= pass_decimal)
            return;
        if (idx == pass_decimal)
        {
            *result = idx;
        }
    }
}

int main(int argc, char **argv)
{
    char *password;
    time_t t1, t2;
    double dif;
    long long *result;

    cudaMallocManaged(&result, sizeof(long long) * 282429536481); // verificar o tamanho
    cudaMallocManaged(&password, sizeof(char) * MAXIMUM_PASSWORD);

    strcpy(password, argv[1]);
    int size = strlen(password);
    int base = END_CHAR - START_CHAR + 2;
    long long max = my_pow_host(base, size);

    // int threadsPerBlock = 1024;
    // dim3 gridDim(ceil(max / (float)threadsPerBlock), 1, 1);
    // dim3 blockDim(threadsPerBlock, 1, 1);

    int deviceId, numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int number_of_blocks = numberOfSMs * 32;
    int threads_per_block = 1024;

    printf("Try to broke the password: %s\n", password);

    time(&t1);
    bruteForce<<<number_of_blocks, threads_per_block>>>(password, size, result);
    cudaDeviceSynchronize();
    time(&t2);

    long long finalRes = *result;

    printf("Found password!\n");
    int index = 0;
    char s[MAXIMUM_PASSWORD];

    printf("Password in decimal base: %lli\n", finalRes);
    while ((finalRes) > 0)
    {
        s[index++] = 'a' + finalRes % base - 1;
        finalRes /= base;
    }
    s[index] = '\0';
    printf("Found password: %s\n", s);

    dif = difftime(t2, t1);

    printf("\n%1.2f seconds\n", dif);
    cudaFree(result);

    return 0;
}