// sqrt_avx2.h
#ifndef SQRT_AVX2_H
#define SQRT_AVX2_H

#include <immintrin.h>

// Constants
extern const float kThreshold;

// Function declaration
void sqrt_avx2(int N, float initialGuess, const float* values, float* output);

#endif // SQRT_AVX2_H
