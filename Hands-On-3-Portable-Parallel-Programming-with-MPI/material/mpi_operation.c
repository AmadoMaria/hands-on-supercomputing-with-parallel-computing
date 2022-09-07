#include <stdio.h>
#include <mpi.h>
#define SIZE 12

int getOperationResult(int operation, int array[SIZE])
{
    int value;
    switch (operation)
    {
    case 1:
        value = 0;
        for (int i = 0; i < SIZE; i++)
            value += array[i];
        break;
    case 2:
        value = 0;
        for (int i = 0; i < SIZE; i++)
            value -= array[i];
        break;
    case 3:
        value = 1;
        for (int i = 0; i < SIZE; i++)
            value *= array[i];
        break;
    }
    return value;
}

char getOperation(int id)
{
    switch (id)
    {
    case 1:
        return '+';
    case 2:
        return '-';
    case 3:
        return '*';
    default:
        break;
    }
}

int main(int argc, char **argv)
{
    int i, sum = 0, subtraction = 0, mult = 1, result, value;
    int array[SIZE];
    int numberProcess, id, to, from, tag = 1000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &numberProcess);
    MPI_Status status;

    if (id == 0)
    {
        for (i = 0; i < SIZE; i++)
        {
            array[i] = i + 1;
            printf("%d %d\t", i, array[i]);
        }
        printf("\n");
        for (to = 1; to < numberProcess; to++)
        {
            MPI_Send(&array, SIZE, MPI_INT, to, tag, MPI_COMM_WORLD);
        }

        for (from = 1; from < numberProcess; from++)
        {
            MPI_Recv(&result, 1, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
            char operation = getOperation(from);
            printf("(%c) = %d\n", operation, result);
        }
    }
    else
    {
        MPI_Recv(&array, SIZE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

        for (int j = 1; j < numberProcess; j++)
        {
            if (id == j)
            {
                value = getOperationResult(id, array);
            }
        }

        MPI_Send(&value, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}