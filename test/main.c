#include <stdio.h>

#include "quantized.h"
#include "random.h"

int main(void) {
    const uint64_t X   = 10;   /* number of arrays */
    const uint64_t N   = 4096; /* length of each array */
    const float  MINV = -10.0f;
    const float  MAXV =  10.0f;
    const unsigned int SEED = 12345;

    /* --- generate inputs ------------------------------------- */
    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    /* --- loop over each array --------------------------------- */
    for (uint64_t k = 0; k < X; ++k) {
        quantized_array_t *qa = NULL;
        if (quantize(inputs[k], N, 1, &qa) || !qa) {
            fprintf(stderr, "quantization failed on array %llu\n", k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *y = malloc(N * sizeof(float));
        if (!y) {
            fprintf(stderr, "malloc failed for dequant buffer\n");
            free_quantized_array(qa);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        if (dequantize(qa, y)) {
            fprintf(stderr, "dequantization failed on array %llu\n", k);
            free(y);
            free_quantized_array(qa);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        /* --- compute metrics ---------------------------------- */
        double mae = 0.0, mse = 0.0, max_abs = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double e = (double)y[i] - (double)inputs[k][i];
            double ae = fabs(e);
            mae += ae;
            mse += e * e;
            if (ae > max_abs) max_abs = ae;
        }
        mae /= N;
        mse /= N;

        printf("[array %llu] N=%llu, blocks=%llu, orig=%.1f KB, "
               "quant=%.1f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               k, N, qa->num_blocks,
               N * sizeof(float) / 1024.0,
               get_quantized_array_size(qa) / 1024.0,
               8 * get_quantized_array_size(qa) / (double)N,
               mae, mse, max_abs);

        free_quantized_array(qa);
        free(y);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
