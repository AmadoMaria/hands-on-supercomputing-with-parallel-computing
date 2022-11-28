#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// 97 to 122 use only lowercase letters
// 65 to 90 use only capital letters
// 48 to 57 use only numbers

#define START_CHAR 97
#define END_CHAR 122
#define MAXIMUM_PASSWORD 20

long long my_pow(long long x, int y)
{
    long long res = 1;
    if (y == 0)
        return res;
    else
        return x * my_pow(x, y - 1);
}

void bruteForce(char *pass, int rank, int numberProcess)
{
    time_t t1, t2;
    double dif;
    time(&t1);

    char force[MAXIMUM_PASSWORD];
    int palavra[MAXIMUM_PASSWORD];
    int pass_b26[MAXIMUM_PASSWORD];

    long long int j;
    long long int pass_decimal = 0;
    int base = END_CHAR - START_CHAR + 2;

    int size = strlen(pass);

    for (int i = 0; i < MAXIMUM_PASSWORD; i++)
        force[i] = '\0';

    for (int i = 0; i < size; i++)
        pass_b26[i] = (int)pass[i] - START_CHAR + 1;

    for (int i = size - 1; i > -1; i--)
        pass_decimal += (long long int)pass_b26[i] * my_pow(base, i);

    long long int max = my_pow(base, size);
    char s[MAXIMUM_PASSWORD];

    long long partition = ceil(max / numberProcess);
    long long int rest = max % numberProcess;

    if (partition == 0 && rest != 0)
    {
        partition = rest;
    }

    long long lower_bound = rank * partition;
    long long upper_bound = (rank + 1) * partition;
    int rankThatFound;

#pragma omp parallel for shared(rankThatFound)
    for (j = lower_bound; j < upper_bound; j++)
    {
        if (j == pass_decimal)
        {
            rankThatFound = rank;
            printf("Found password!\n");
            int index = 0;

            printf("Password in decimal base: %lli\n", j);
            while (j > 0)
            {
                s[index++] = START_CHAR + j % base - 1;
                j /= base;
            }
            s[index] = '\0';
            printf("Found password: %s\n", s);
            time(&t2);
            dif = difftime(t2, t1);

            printf("\n%1.2f seconds\n", dif);
            MPI_Finalize();
            exit(0);
        }
    }
    if (rankThatFound != rank)
    {
        MPI_Finalize();
        exit(0);
    }
}

int main(int argc, char **argv)
{
    char password[MAXIMUM_PASSWORD];
    strcpy(password, argv[1]);

    int numberProcess, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberProcess);
    MPI_Status status;

    if (rank == 0)
        printf("Try to broke the password: %s\n", password);

    bruteForce(password, rank, numberProcess);

    return 0;
}