#include "stencil.cuh"
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    // // ShMem allocation determined at run time
    extern __shared__ float shared_memory[];

    // Pointers within shared memory
    float* shared_image = shared_memory;                     
    float* shared_mask = &shared_memory[blockDim.x+2*R]; 
    float* shared_output = &shared_memory[blockDim.x + 4*R + 1];

    // Global thread ID and local thread ID
    int l_tid = threadIdx.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;



    if(l_tid < R)
    {
	    int left_halo_index = g_tid - R;
	    int right_halo_index = g_tid + blockDim.x;
	    shared_image[threadIdx.x] = (left_halo_index >= 0 )?(image[left_halo_index]):1.0;
	    shared_image[blockDim.x + R + threadIdx.x] = (right_halo_index < n) ? (image[right_halo_index]):1.0;
    }
    if(g_tid < n)
      shared_image[R+threadIdx.x] = image[g_tid];
    if(l_tid <= 2*R)
      shared_mask[threadIdx.x] = mask[threadIdx.x];




    // make sure all threads have completed loading into shared memory
    __syncthreads();

    // convolution operation
    if (g_tid < n) 
    {
        float result = 0.0;
        for (int i = -(int)R; i <= (int)R; i++) 
        {
            result += shared_image[l_tid + R + i] * shared_mask[i+R];
        }
        shared_output[l_tid] = result;
    }
    __syncthreads();
    if(g_tid < n) 
      output[g_tid] = shared_output[threadIdx.x];
}



__host__ void stencil(const float* image,
    const float* mask,
    float* output,
    unsigned int n,
    unsigned int R,
    unsigned int threads_per_block) 
{
     
    // number of required blocks
    unsigned int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    
    // Shared memory includes:
    // - threads_per_block + 2 * R elements for shared_image
    // - 2 * R + 1 elements for shared_mask
    // size_t shared_memory_size = (2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float);

    
    stencil_kernel<<<blocks_per_grid, threads_per_block, (2*R + threads_per_block + (2*R+1) + threads_per_block)*sizeof(float)>>>(image, mask, output, n, R);

    
    cudaDeviceSynchronize();
}


