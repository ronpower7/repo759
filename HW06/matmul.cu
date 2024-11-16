#include "matmul.cuh"
#include <cuda.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
 // 1D kernel configuration so deduce row and col from idx 
    int idx  = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx < n * n) {
      int row = idx / n ;
      int col = idx % n ;
      float sum = 0.0f;
      for (int k = 0 ; k < n ; ++k) {
         sum += A[(n * row) + k]  * B[(k*n) + col];
      }
      C[idx] = sum ;
    }
 }


void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
   // as it is 1D kernel, we need to deduce the num of blocks present 	
    unsigned int num_elements = n * n ;
    int num_blocks = (num_elements + threads_per_block - 1)/threads_per_block;
    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
}        
