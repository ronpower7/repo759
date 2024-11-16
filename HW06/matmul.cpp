#include "matmul.cuh"
#include <cuda.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
  
    int row  = blockIdx.y * blockDim.y + threadIdx.y ;
    int col  = blockIdx.x * blockDim.x + threadIdx.x ;

    if (row < n && col < n) {
      float sum = 0.0f;
      for (int k = 0 ; k < n ; ++k) {
         sum += A[(n * row) + k]  * B[(k*n) + col];
      }
      C[(row*n) + col] = sum ;
    }
 }


void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Calculate grid and block dimensions depending upon the threads per block , max threads per block = 1024 ,so taking BLOCK_SIZE = 32
    unsigned int blockDimX = BLOCK_SIZE ;
    unsigned int blockDimY = threads_per_block / BLOCK_SIZE ;
    dim3 blockDim(blockDimX,blockDimY);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, n);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
}        
