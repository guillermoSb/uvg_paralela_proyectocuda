#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "./common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 90;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accumulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * 180 / degreeInc];
  memset(*acc, 0, sizeof(int) * rBins * 180 / degreeInc);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
    {
      int idx = j * w + i;
      if (pic[idx] > 00)
      {
        int xCoord = i - xCent;
        int yCoord = yCent - j; // y-coord has to be reversed
        float theta = 0;        // actual angle
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          float r = xCoord * cos(theta) + yCoord * sin(theta);
          int rIdx = (r + rMax) / rScale;
          (*acc)[rIdx * degreeBins + tIdx]++;
          theta += radInc;
        }
      }
    }
}

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;

    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    extern __shared__ int localAcc[]; // Use shared memory as local accumulator
    int locID = threadIdx.x;

    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        localAcc[i] = 0;
    }

    __syncthreads(); // Barrier to ensure all threads have completed initialization

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }

    __syncthreads(); // Barrier to ensure all threads have completed local accumulation

    for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        atomicAdd(&acc[i], localAcc[i]);
    }
}

//*****************************************************************
int main(int argc, char **argv)
{
 int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // compute values to be stored in device constant memory
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // copy precomputed values to constant memory
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels; // h_in is just an alias here

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // ! ========================================================================
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // ! ========================================================================

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    int blockNum = ceil(w * h / 256.0);
    int sharedMemorySize = sizeof(int) * degreeBins * rBins;
    GPU_HoughTran<<<blockNum, 256, sharedMemorySize>>>(d_in, w, h, d_hough, rMax, rScale);

    cudaDeviceSynchronize(); // Wait for the GPU launched work to complete

    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);


  // ! ========================================================================
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // ! ========================================================================

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }

  // ! ========================================================================
  printf("Kernel execution time: %f milliseconds\n", milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // ! ========================================================================

  // clean-up
  cudaFree((void *)d_in);
  cudaFree((void *)d_hough);
  free(h_hough);
  free(cpuht);
  cudaDeviceReset();

  return 0;
}
