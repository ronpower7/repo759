
#include <cuda_runtime.h>
#include <iostream>
#include "reduce.cuh"



__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Perform first level of reduction upon reading from global memory
    if(i+blockDim.x < n)
   	 sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    else if(i < n)
	 sdata[tid] = g_idata[i];
    else
	 sdata[tid] = 0 ;

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



// by using float **input, I can modify the pointer itself, allowing it to point to a new location which is needed inside the function

__host__ void reduce(float **input, float **output, unsigned int N,unsigned int threads_per_block)
{
    while(N > 1)
    {
    unsigned int num_blocks = (N + threads_per_block * 2 - 1)/ (threads_per_block * 2 );

    reduce_kernel<<<num_blocks,threads_per_block,threads_per_block*sizeof(float)>>>(*input,*output,N);

    cudaDeviceSynchronize();

    *input = *output;

    N = num_blocks;
	   
    }

}

