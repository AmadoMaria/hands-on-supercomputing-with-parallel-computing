#include <stdio.h>
#include <mpi.h>
#define SIZE 12

int main(int argc, char **argv)
{
  int i, sum = 0, subtraction = 0, mult = 1, result, value;
  int array[SIZE];
  char ops[] = {'+', '-', '*'};
  char operationsRec;
  int numberProcess, id, to, from, tag = 1000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &numberProcess);
  MPI_Status status;
  printf("id: %d\n\n", id);

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
      MPI_Send(&ops[to - 1], 1, MPI_CHAR, to, tag, MPI_COMM_WORLD);
    }

    for (to = 1; to < numberProcess; to++)
    {
      MPI_Recv(&result, 1, MPI_INT, to, tag, MPI_COMM_WORLD, &status);
      MPI_Recv(&operationsRec, 1, MPI_CHAR, to, tag, MPI_COMM_WORLD, &status);
      printf("(%c) = %d\n", operationsRec, result);
    }
  }
  else
  {
    MPI_Recv(&array, SIZE, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&operationsRec, 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);

    switch (operationsRec)
    {
    case '+':
      value = 0;
      for (i = 0; i < SIZE; i++)
        value += array[i];
      break;
    case '-':
      value = 0;
      for (i = 0; i < SIZE; i++)
        value -= array[i];
      break;
    case '*':
      value = 1;
      for (i = 0; i < SIZE; i++)
        value *= array[i];
      break;
    }
    MPI_Send(&value, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    MPI_Send(&operationsRec, 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  // printf("Sum = %d\n", sum);
  // printf("Subtraction = %d\n", subtraction);
  // printf("Multiply = %d\n", mult);

  MPI_Finalize();
  return 0;
}