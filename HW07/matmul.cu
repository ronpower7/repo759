#include "matmul.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    // Declare shared memory dynamically
   // extern __shared__ T shared_mem[];  // Shared memory for both A and B tiles
    extern __shared__ char shared_mem[];
    T *tile_A = reinterpret_cast<T *>(shared_mem);
    T *tile_B = reinterpret_cast<T *>(shared_mem) + block_dim * block_dim ;
   // T *tile_A = shared_mem;
   // T *tile_B = shared_mem + block_dim * block_dim;

    int row = blockIdx.y * block_dim + threadIdx.y;
    int col = blockIdx.x * block_dim + threadIdx.x;
    T temp = 0;

    // Matrix multiplication with tiling
    for (int i = 0; i < (int) ((n + block_dim - 1) / block_dim) ; ++i) {
        // Load data into shared memory (tile A)
        if (row < n && (i * block_dim + threadIdx.x) < n) {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = A[row * n + i * block_dim + threadIdx.x];
        } else {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

        // Load data into shared memory (tile B)
        if (col < n && (i * block_dim + threadIdx.y) < n) {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = B[(i * block_dim + threadIdx.y) * n + col];
        } else {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

        // Synchronize to make sure all threads have loaded their data
        __syncthreads();

        // Perform multiplication for the current tile
        for (int j = 0; j < block_dim; ++j) {
            temp += tile_A[threadIdx.y * block_dim + j] * tile_B[j * block_dim + threadIdx.x];
        }

        // Synchronize again before loading new tiles
        __syncthreads();
    }

    // Store the result in C
    if (row < n && col < n) {
        C[row * n + col] = temp;
    }
}

// Host functions for different data types
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);

    matmul_kernel<int><<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    matmul_kernel<float><<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);

    matmul_kernel<double><<<dimGrid, dimBlock, shared_mem_size>>>(A, B, C, n, block_dim);
}
