
#include <cuda_runtime.h>
#include <iostream>
#include "reduce.cuh"



__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Perform first level of reduction upon reading from global memory
    if (i < n) {
        sdata[tid] = g_idata[i] + (i + blockDim.x < n ? g_idata[i + blockDim.x] : 0);
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}



__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    unsigned int shared_mem_size = threads_per_block * sizeof(float);

    float *d_input = *input;
    float *d_output = *output;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    while (N > 1) {
        reduce_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(d_input, d_output, N);

        N = num_blocks;
        num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

        // Swap input and output pointers
        float *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();
    printf("Reduction Time: %f ms\n", milliseconds);

    // Copy the result back to the original input pointer
    *input = d_input;
    *output = d_output;
}
