#include <stdio.h>
#include <mpi.h>
#define SIZE 5

double getOperationResult(int processIndex, double array[SIZE])
{
  double term;
  double x = array[0];
  switch (processIndex)
  {
  case 1:
    term = (x * x * x) * array[1];
    break;
  case 2:
    term = (x * x) * array[2];
    break;
  case 3:
    term = x * array[3] + array[4];
    break;
  }
  return term;
}

int main(int argc, char **argv)
{
  int i;
  double result, value, x, total = 0;
  double array[SIZE];
  int numberProcess, id, to, from, tag = 1000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &numberProcess);
  MPI_Status status;

  if (id == 0)
  {
    char c;
    printf("\nf(x) = a*x^3 + b*x^2 + c*x + d\n");
    for (c = 'a'; c < 'e'; c++)
    {
      printf("\nEnter the value of the 'constants' %c:\n", c);
      scanf("%lf", &array[(c - 'a') + 1]);
    }

    printf("\nf(x)=%lf*x^3+%lf*x^2+%lf*x+%lf\n", array[1], array[2], array[3], array[4]);
    printf("\nEnter the value of 'x':\n");
    scanf("%lf", &x);
    array[0] = x;

    for (to = 1; to < numberProcess; to++)
    {
      MPI_Send(&array, SIZE, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
    }

    for (from = 1; from < numberProcess; from++)
    {
      MPI_Recv(&result, 1, MPI_DOUBLE, from, tag, MPI_COMM_WORLD, &status);
      total = total + result;
    }
    printf("\nf(%lf) = %lf*x^3 + %lf*x^2 + %lf*x + %lf = %lf\n", array[0], array[1], array[2], array[3], array[4], total);
  }
  else
  {
    MPI_Recv(&array, SIZE, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);

    for (int j = 1; j < numberProcess; j++)
    {
      if (id == j)
      {
        value = getOperationResult(id, array);
      }
    }

    MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}