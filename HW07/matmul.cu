
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    extern __shared__ T shared_mem[];
    T *tile_A = shared_mem;
    T *tile_B = shared_mem + block_dim * block_dim;

    int row = blockIdx.y * block_dim + threadIdx.y;
    int col = blockIdx.x * block_dim + threadIdx.x;
    T temp = 0;

    for (int i = 0; i < (n + block_dim - 1) / block_dim; ++i) {
      // Boundary condition
        if (row < n && (i * block_dim + threadIdx.x) < n) {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = A[row * n + i * block_dim + threadIdx.x];
        } else {
            tile_A[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

      // Boundary condition
        if (col < n && (i * block_dim + threadIdx.y) < n) {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = B[(i * block_dim + threadIdx.y) * n + col];
        } else {
            tile_B[threadIdx.y * block_dim + threadIdx.x] = 0;
        }

        __syncthreads();

        for (int j = 0; j < block_dim; ++j) {
            temp += tile_A[threadIdx.y * block_dim + j] * tile_B[j * block_dim + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = temp;
    }
}



__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    int *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<int><<<dimGrid, dimBlock, shared_mem_size>>>(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    printf("Time: %f ms\n", milliseconds);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<float><<<dimGrid, dimBlock, shared_mem_size>>>(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    printf("Time: %f ms\n", milliseconds);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<double><<<dimGrid, dimBlock, shared_mem_size>>>(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    printf("Time: %f ms\n", milliseconds);
}















