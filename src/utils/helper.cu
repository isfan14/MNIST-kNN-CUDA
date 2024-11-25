#include <stdio.h>
#include <stdlib.h>

#include "helper.cuh"

void cudaCheck(cudaError_t error_code, const char *file, int line)
{
  if (error_code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error %d: '%s'. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);

    fflush(stderr);
    exit(error_code);
  }
}
