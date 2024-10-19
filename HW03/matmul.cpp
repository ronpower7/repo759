#include <vector>
#include "matmul.h"
#include <omp.h>


void mmul (const float* A , const float* B, float* C, const std::size_t n) {
  unsigned int i;
  unsigned int j;
  unsigned int k;
  #pragma omp parallel for 
    for (i = 0 ; i < n ; ++i ) {
         for ( k = 0 ; k < n ; ++k) {         
             for (j = 0 ; j < n ; ++j) {
                   C[i * n + j] += A[i * n + k] * B[k * n + j] ;
             }
         }
    }
}   

