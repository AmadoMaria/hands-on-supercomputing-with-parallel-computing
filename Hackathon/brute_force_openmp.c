#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// 97 to 122 use only lowercase letters
// 65 to 90 use only capital letters
// 48 to 57 use only numbers

#define START_CHAR 97
#define END_CHAR 122
#define MAXIMUM_PASSWORD 20

long long my_pow(long long x, int y)
{
    long long res = 1;
    long long i;

    for (i = 0; i < y; i++)
    {
        res *= x;
    }

    return res;
}

void bruteForce(char *pass)
{
    char force[MAXIMUM_PASSWORD];
    int palavra[MAXIMUM_PASSWORD];
    int pass_b26[MAXIMUM_PASSWORD];

    long long int j;
    long long int pass_decimal = 0;
    int base = END_CHAR - START_CHAR + 2;

    int size = strlen(pass);

    for (int i = 0; i < MAXIMUM_PASSWORD; i++)
        force[i] = '\0';

    printf("Try to broke the password: %s\n", pass);

    for (int i = 0; i < size; i++)
        pass_b26[i] = (int)pass[i] - START_CHAR + 1;

    for (int i = size - 1; i > -1; i--)
        pass_decimal += (long long int)pass_b26[i] * my_pow(base, i);

    long long int max = my_pow(base, size);
    char s[MAXIMUM_PASSWORD];
    int go = 1;

#pragma omp parallel for shared(go)
    for (j = 0; j < max; j++)
    {
        if (go)
        {
            if (j == pass_decimal)
            {
                go = 0;
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
            }
        }
    }
}

int main(int argc, char **argv)
{
    char password[MAXIMUM_PASSWORD];
    strcpy(password, argv[1]);
    time_t t1, t2;
    double dif;

    time(&t1);
    bruteForce(password);
    time(&t2);

    dif = difftime(t2, t1);

    printf("\n%1.2f seconds\n", dif);

    return 0;
}
