# Quantization and Sparsity Playground

A tiny C library that implements block-wise 8-bit (q8_0) and 4-bit (q4_0) quantization in the style of GGML, along with sparsity compression using a zero-based COO format for 2D arrays. The quantization logic is in `include/quantization.h` and `src/quantization.c`. The sparsity logic is in `include/sparsity.h` and `src/sparsity.c`. Test scripts are provided in the `test/` directory to evaluate quantization and sparsity on random data and a real example.

## Quick Start

```bash
# Build the library and tests
make

# Run quantization test on random arrays
./build/test_quantization

# Run sparsity test on random arrays
./build/test_sparsity

# Run real example test (requires example/example.bin)
./build/test_real_example
```

The `test_quantization` executable prints per-array reports for quantization metrics, such as:

```plaintext
[array 0] N=4194304, blocks=131072, original_size=16384.000 KB
   Q8_0:  size=4608.000 KB, B/W=9.00000, MAE=0.019052, MSE=0.000495, MaxAbs=0.039370
   Q4_0:  size=2560.000 KB, B/W=5.00000, MAE=0.339177, MSE=0.157560, MaxAbs=0.714286
```

The `test_sparsity` executable prints similar reports for sparsity:

```plaintext
[array 0] N=4194304 (tokens=512, features=8192), original_size=16384.000 KB
   Sparse0.25: sparsity=0.250, size=5120.000 KB, B/W=10.00000, MAE=0.339177, MSE=0.157560, MaxAbs=9.999999
   Sparse0.125: sparsity=0.125, size=2560.000 KB, B/W=5.00000, MAE=0.509918, MSE=0.354529, MaxAbs=9.999999
```

The `test_real_example` processes a binary file (`example/example.bin`) with both quantization and sparsity, outputs recovered binaries, and prints metrics.

## API Reference

The library provides APIs for quantization and sparsity. Memory management helpers are included for allocating, freeing, sizing, and loading from buffers.

### Quantization API

```c
/* ---- Allocation / Free / Size / Load ---------------------------------- */
quantized_array_t *allocate_q8_0_array(uint64_t num_elements,
                                       uint64_t block_size);

quantized_array_t *allocate_q4_0_array(uint64_t num_elements,
                                       uint64_t block_size);

void free_quantized_array(quantized_array_t *quantized_array);

int64_t get_quantized_array_size(const quantized_array_t *quantized_array);

quantized_array_t *load_quantized_array_from_buffer(const void *buffer, int64_t buffer_size);

/* ---- Quantization / Dequantization ------------------------------------ */
int quantize(const float *float_array,
             uint64_t num_elements,
             uint8_t quantized_type,          /* 0 = q8_0, 1 = q4_0 */
             quantized_array_t **quantized_array);   /* out */

int dequantize(const quantized_array_t *quantized_array,
               float *float_array);            /* out */

/* ---- Quantized array struct ------------------------------------------- */
typedef struct {
    uint8_t  quantized_type; /* 0: q8_0, 1: q4_0, … */
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of blocks (for block-wised formats) */
    uint64_t block_size;     /* elements per block */
    float  *scales;          /* length = num_blocks (or num_superblocks for kquant formats) */
    int8_t *data;            /* for kquant, here need to contain quantized scale value + quantized value, otherwise it only need to store quantized value*/
} quantized_array_t;
```

### Sparsity API

```c
/* ---- Allocation / Free / Size / Load ---------------------------------- */
sparse_array_t *allocate_sparse_array(uint16_t num_tokens, uint16_t num_features, float sparse_ratio);

void free_sparse_array(sparse_array_t *sparse_array);

uint64_t get_sparse_array_size(const sparse_array_t *sparse_array);

sparse_array_t *load_sparse_array_from_buffer(const void *buffer, uint64_t buffer_size);

/* ---- Compression / Decompression -------------------------------------- */
int compress(const float *float_array, uint16_t num_tokens, uint16_t num_features,  float sparse_ratio, sparse_array_t **sparse_array);

int decompress(const sparse_array_t *sparse_array, float *float_array);

/* ---- Sparse array struct ---------------------------------------------- */
/**
 * @brief Represents a sparse array in zero-based COO format for 2D data with shape [num_tokens, num_features].
 *
 * Sparsity is applied along the features dimension. Since each token retains the same number of sparse features,
 * token indices are not stored explicitly. The structure holds the selected feature indices and corresponding values
 * for all tokens in a flattened manner.
 */
typedef struct {
    uint16_t num_tokens;                /* Number of tokens (rows in the 2D shape). */
    uint16_t num_features;              /* Number of features per token (columns in the 2D shape). */
    uint16_t num_sparse_features;       /* Number of retained sparse features per token (must be <= num_features). */
    uint16_t *sparse_indices;           /* Flattened array of selected feature indices; length is (num_tokens * num_sparse_features). */
    float *values;                      /* Flattened array of corresponding sparse values; length is (num_tokens * num_sparse_features). */
} sparse_array_t;
```

### Example Usage: Quantization

```c
#include "quantization.h"

float src[1000];          /* fill with data */
quantized_array_t *qa = NULL;

/* Quantize to q4_0 */
if (quantize(src, 1000, 1, &qa) != 0) {
    /* handle error */
}

/* Dequantize */
float dst[1000];
if (dequantize(qa, dst) != 0) {
    /* handle error */
}

/* Clean up */
free_quantized_array(qa);
```

The `quantize` function allocates the quantized array; provide a pointer to receive it.

### Example Usage: Sparsity

```c
#include "sparsity.h"

float src[1000];          /* fill with data; assume 2D shape [num_tokens, num_features] */
sparse_array_t *sa = NULL;

/* Compress with 25% sparsity */
if (compress(src, 10 /* num_tokens */, 100 /* num_features */, 0.25f, &sa) != 0) {
    /* handle error */
}

/* Decompress */
float dst[1000];
if (decompress(sa, dst) != 0) {
    /* handle error */
}

/* Clean up */
free_sparse_array(sa);
```

The `compress` function allocates the sparse array; provide a pointer to receive it. Input is treated as a flattened 2D array [num_tokens, num_features].

## License

MIT License – see the LICENSE file for details.