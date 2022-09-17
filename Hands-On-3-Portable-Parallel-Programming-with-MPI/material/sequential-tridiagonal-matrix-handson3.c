#include <stdio.h>
#include <mpi.h>
#define ORDER 4

int matrix[ORDER][ORDER];
int ma[ORDER][ORDER]; // matriz auxiliar
int k[3] = {100, 200, 300};

void printMatrix(int m[][ORDER])
{
  int i, j;
  for (i = 0; i < ORDER; i++)
  {
    printf("| ");
    for (j = 0; j < ORDER; j++)
    {
      printf("%3d ", m[i][j]);
    }
    printf("|\n");
  }
  printf("\n");
}

void getOperationResult(int operation, int m[][ORDER])
{
  int i;
  switch (operation)
  {
  case 1:
    for (i = 0; i < ORDER; i++)
    {
      ma[i][i] = m[i][i] + k[0]; // main diagonal
    }
    break;
  case 2:
    for (i = 0; i < ORDER; i++)
    {
      ma[i + 1][i] = m[i + 1][i] + k[1]; // subdiagonal
    }
    break;
  case 3:
    for (i = 0; i < ORDER; i++)
    {
      ma[i][i + 1] = m[i][i + 1] + k[2]; // superdiagonal
    }
    break;
  }
}

void buildMatrix(int operation, int m[][ORDER])
{
  int i;
  switch (operation)
  {
  case 1:
    for (i = 0; i < ORDER; i++)
    {
      matrix[i][i] = m[i][i];
    }
    break;
  case 2:
    for (i = 0; i < ORDER; i++)
    {
      matrix[i + 1][i] = m[i + 1][i];
    }
    break;
  case 3:
    for (i = 0; i < ORDER; i++)
    {
      matrix[i][i + 1] = m[i][i + 1];
    }
    break;
  }
}

int main(int argc, char **argv)
{
  int i, result[ORDER][ORDER], matrixInit[ORDER][ORDER];
  int numberProcess, id, to, from, tag = 1000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &numberProcess);
  MPI_Status status;

  if (id == 0)
  {
    int i, j;
    for (i = 0; i < ORDER; i++)
    {
      for (j = 0; j < ORDER; j++)
      {
        if (i == j)
          matrixInit[i][j] = i + j + 1;
        else if (i == (j + 1))
        {
          matrixInit[i][j] = i + j + 1;
          matrixInit[j][i] = matrixInit[i][j];
        }
        else
          matrixInit[i][j] = 0;
      }
    }
    printMatrix(matrixInit);
    for (to = 1; to < numberProcess; to++)
    {
      MPI_Send(&matrixInit, ORDER * ORDER, MPI_INT, to, tag, MPI_COMM_WORLD);
    }

    for (from = 1; from < numberProcess; from++)
    {
      MPI_Recv(&result, ORDER * ORDER, MPI_INT, from, tag, MPI_COMM_WORLD, &status);
      buildMatrix(from, result);
    }
    printMatrix(matrix);
  }
  else
  {
    int matrixSent[ORDER][ORDER];

    MPI_Recv(&matrixSent, ORDER * ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

    for (int j = 1; j < numberProcess; j++)
    {
      if (id == j)
      {
        getOperationResult(id, matrixSent);
      }
    }

    MPI_Send(&ma, ORDER * ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}