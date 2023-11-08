/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc hello.cu -o hello -arch=sm_20
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void hello()
{
  // Obtener el blockIdx
  int threadId= blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello world\n");
  printf("blockIdx: %d\n", blockIdx.x);
  printf("blockDim: %d\n", blockDim.x);
  printf("threadIdx: %d\n", threadIdx.x);
  printf("threadId: %d\n", threadId);

}

int main()
{
  hello<<<1,10>>>();
  cudaThreadSynchronize();
  return 0;
}
