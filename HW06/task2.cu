#include <iostream>
#include <cuda.h>
#include "stencil.cuh"

#include <random>           // To generate random numbers
using namespace std;


int main(int argc, char *argv[])
{
    // command line arguments
    unsigned int n = std::stoi(argv[1]);                        // number of elements
    unsigned int R = std::stoi(argv[2]);
    unsigned int threads_per_block = std::stoi(argv[3]);        // threads per block
    
    // declarations for calculating time
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // random number generation
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist1(-1.0, 1.0);
    std::uniform_real_distribution<float> dist2(-1.0, 1.0);


    // host arrays 
    float *image= (float*)malloc(n*(sizeof(float)));
    float *output= (float*)malloc(n*(sizeof(float)));
    float *mask= (float*)malloc((2*R+1)*(sizeof(float)));

    for (size_t i = 0; i < n; i++)
    {
        image[i] = dist1(generator);
    }

    for (size_t i = 0; i < (2*R+1); i++)
    {
        mask[i] = dist2(generator);
    }

    // device arrays
    float *d_image, *d_mask, *d_output;

    // allocate memory on the device (GPU)
    cudaMalloc((void **)&d_image, n * sizeof(float));
    cudaMalloc((void **)&d_mask, (2*R+1) * sizeof(float));
    cudaMalloc((void **)&d_output, n * sizeof(float));

    // copy data into device blocks
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2*R+1) * sizeof(float), cudaMemcpyHostToDevice);


    // #############################################
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);
    // #############################################
    
    cudaEventSynchronize(stop);
    float elapsedtime;                                                  
    cudaEventElapsedTime(&elapsedtime, start, stop);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);




    std::cout << output[n - 1] << std::endl;
    std::cout << elapsedtime << "ms"  << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // deallocate memory from Host and Device
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    free(image);
    free(mask);
    free(output);


    return 0;
    
}

