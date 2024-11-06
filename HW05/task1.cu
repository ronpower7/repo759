#include <cstdio>
#include <cuda_runtime.h>

 
__global__ void factorialKernel() {

   int x ;
   int y = 1 ;
   
   x = threadIdx.x + 1 ;

   for (int i  = 1 ; i <= x ; ++i) {
     
      y *= i ;


   }
   
   printf("%d!=%d \n",x,y);
}

 

int main()  {
  
  factorialKernel<<<1,8>>>();
  cudaDeviceSynchronize();
  return 0;

}
