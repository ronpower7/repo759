#include "convolution.h"
#include <algorithm>
#include <omp.h>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
 {
   unsigned int offset = (m - 1) / 2;
   unsigned int x;
   unsigned int y;
   unsigned int i;
   unsigned int j;
   unsigned int xi; 
   unsigned int yj;
   float sum;

  #pragma omp parallel for collapse(2)  
    for ( x= 0; x < n; ++x) {
        for ( y = 0; y < n; ++y) {
            sum = 0.0f;
            for ( i = 0; i < m; ++i) {
                for ( j = 0; j < m; ++j) {
                   xi = x + i - offset;
                   yj = y + j - offset;
                    float value = 0.0f;

                    if (xi >= 0 && xi < n && yj >= 0 && yj < n) {
                        value = image[n * xi + yj];
                    } else if ((xi >= 0 && xi < n) || (yj >= 0 && yj < n)) {
                        value = 1.0f;
                    }

                    sum += mask[m * i + j] * value;
                }
            }
            output[n * x + y] = sum;
        }
    }

}
