#include "convolution.h"
#include <algorithm>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
 {
   unsigned int offset = (m - 1) / 2;

    for (unsigned int x = 0; x < n; ++x) {
        for (unsigned int y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < m; ++i) {
                for (unsigned int j = 0; j < m; ++j) {
                   unsigned int xi = x + i - offset;
                   unsigned int yj = y + j - offset;
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
