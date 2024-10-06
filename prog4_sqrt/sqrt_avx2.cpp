// sqrt_avx2.cpp
#include "sqrt_avx2.h"
#include <cmath>
#include <immintrin.h> // Ensure this is included for AVX intrinsics

// Define the threshold
const float kThreshold = 0.00001f;

// Inline function to compute absolute value using AVX2
inline __m256 abs256_ps(__m256 x) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(x, mask);
}

// AVX2 sqrt function implementation
void sqrt_avx2(int N, float initialGuess, const float* values, float* output) {
    __m256 vThreshold = _mm256_set1_ps(kThreshold);
    __m256 vThree = _mm256_set1_ps(3.0f);
    __m256 vHalf = _mm256_set1_ps(0.5f);
    __m256 vInitialGuess = _mm256_set1_ps(initialGuess);

    for (int i = 0; i < N; i += 8) {
        __m256 vX = _mm256_loadu_ps(&values[i]);
        __m256 vGuess = vInitialGuess;

        // Compute the initial prediction
        __m256 vPred = abs256_ps(_mm256_sub_ps(
            _mm256_mul_ps(_mm256_mul_ps(vGuess, vGuess), vX),
            _mm256_set1_ps(1.0f)
        ));

        // Iterative approximation loop
        while (_mm256_movemask_ps(_mm256_cmp_ps(vPred, vThreshold, _CMP_GT_OQ))) {
            __m256 vGuessCubed = _mm256_mul_ps(_mm256_mul_ps(vGuess, vGuess), vGuess);
            __m256 vNewGuess = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_mul_ps(vThree, vGuess), _mm256_mul_ps(vX, vGuessCubed)),
                vHalf
            );
            vGuess = vNewGuess;

            // Recompute the prediction after updating the guess
            vPred = abs256_ps(_mm256_sub_ps(
                _mm256_mul_ps(_mm256_mul_ps(vGuess, vGuess), vX),
                _mm256_set1_ps(1.0f)
            ));
        }

        // Store the result
        __m256 vResult = _mm256_mul_ps(vX, vGuess);
        _mm256_storeu_ps(&output[i], vResult);
    }
}
