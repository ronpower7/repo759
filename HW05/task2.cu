#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>

__global__ void computeAxpyKernel(int* devArray , int a) {

	int x = threadIdx.x;
	int y = blockIdx.x;
	int idx = y * blockDim.x + x ;
        devArray[idx] = a * x + y ; 

}

int main() {
    
    const int numElems = 16 ;
    int randN;
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distribution(1, 20);
    randN = distribution(gen) ;
    int hA[numElems] , *dA;

    // allocate memory on the device (GPU) and reset all entries in the device array 
     cudaMalloc((void**) &dA,numElems * sizeof(int));
     cudaMemset(dA,0,numElems * sizeof(int));

    // invoke GPU kernel with two blocks and each block having 8 threads 
     computeAxpyKernel<<<2,8>>>(dA,randN);

    // bring the result back from the GPU into host  
     cudaMemcpy(&hA,dA,sizeof(int) * numElems,cudaMemcpyDeviceToHost);

   // print the result 
    for (int i = 0 ; i < numElems ; i++) {
       
       printf("%d ",hA[i]);


    }	    
   printf("\n ");
   std::cout << std::endl;
   cudaFree(dA);
   return 0 ;

} 

