#include <stdio.h>
#include <stdlib.h>

#include "helper.cuh"

void cudaCheck(cudaError_t error_code, const char *file, int line)
{
  if (error_code != cudaSuccess)
  {
    std::cerr << "Cuda Error " << error_code << ": '" << cudaGetErrorString(error_code) << "' In file '" << file << "' on line " << line << std::endl;
    // fprintf(stderr, "CUDA Error %d: '%s'. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);

    fflush(stderr);
    exit(error_code);
  }
}
