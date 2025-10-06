#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "quantization.h"
#include "random.h"

static void measure_metrics(const float *orig, const float *deq, uint64_t N,
                double *mae, double *mse, double *max_abs) {
    double m = 0.0, s = 0.0, mx = 0.0;
    for (uint64_t i = 0; i < N; ++i) {
        double e = (double)deq[i] - (double)orig[i];
        double ae = fabs(e);
        m   += ae;
        s   += e * e;
        if (ae > mx) mx = ae;
    }
    *mae     = m / (double)N;
    *mse     = s / (double)N;
    *max_abs = mx;
}

int main(void)
{
    /* ---- configuration --------------------------------------------------- */
    const uint64_t X   = 10;            /* number of random arrays            */
    const uint64_t N   = 4194304;          /* length of each array               */
    const float  MINV = -10.0f;
    const float  MAXV =  10.0f;
    const unsigned int SEED = 12345;    /* deterministic seed                */

    /* ---- generate random inputs ----------------------------------------- */
    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    /* ---- loop over each array ------------------------------------------- */
    for (uint64_t k = 0; k < X; ++k) {
        /* ---- q4_0 ------------------------------------------------------- */
        quantized_array_t *qa4 = NULL;
        if (quantize(inputs[k], N, 1 /*q4_0*/, &qa4) || !qa4) {
            fprintf(stderr, "q4_0 quantisation failed on array %lu\n", k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *y4 = malloc(N * sizeof(float));
        if (!y4) {
            fprintf(stderr, "malloc failed for q4_0 dequant buffer\n");
            free_quantized_array(qa4);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        if (dequantize(qa4, y4)) {
            fprintf(stderr, "q4_0 dequantisation failed on array %lu\n", k);
            free(y4);
            free_quantized_array(qa4);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae4, mse4, maxabs4;
        measure_metrics(inputs[k], y4, N, &mae4, &mse4, &maxabs4);

        /* ---- q8_0 ------------------------------------------------------- */
        quantized_array_t *qa8 = NULL;
        if (quantize(inputs[k], N, 0 /*q8_0*/, &qa8) || !qa8) {
            fprintf(stderr, "q8_0 quantisation failed on array %lu\n", k);
            free(y4);
            free_quantized_array(qa4);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *y8 = malloc(N * sizeof(float));
        if (!y8) {
            fprintf(stderr, "malloc failed for q8_0 dequant buffer\n");
            free(y4);
            free_quantized_array(qa4);
            free_quantized_array(qa8);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        if (dequantize(qa8, y8)) {
            fprintf(stderr, "q8_0 dequantisation failed on array %lu\n", k);
            free(y4); free(y8);
            free_quantized_array(qa4);
            free_quantized_array(qa8);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae8, mse8, maxabs8;
        measure_metrics(inputs[k], y8, N, &mae8, &mse8, &maxabs8);

        /* ---- report ------------------------------------------------------ */
        double size4_kb = get_quantized_array_size(qa4) / 1024.0;
        double size8_kb = get_quantized_array_size(qa8) / 1024.0;
        double bw4 = 8.0 * size4_kb * 1024.0 / (double)N;   /* bits per weight */
        double bw8 = 8.0 * size8_kb * 1024.0 / (double)N;

        printf("[array %lu] N=%lu, blocks=%lu, original_size=%.3f KB\n", k, N, qa4->num_blocks, N * sizeof(float) / 1024.0);
        printf("   Q8_0:  size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size8_kb, bw8, mae8, mse8, maxabs8);
        printf("   Q4_0:  size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size4_kb, bw4, mae4, mse4, maxabs4);

        /* ---- clean ------------------------------------------------------- */
        free(y4);   
        free(y8);
        free_quantized_array(qa4);
        free_quantized_array(qa8);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
