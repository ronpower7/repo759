#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <random>

__global__ void computeAxpy(int* dA , int a) {




}

int main() {

    int randN;
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distribution(1, 20);
    randN = distribution(gen) ;
 


 return 0 ;

} 

