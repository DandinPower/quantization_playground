# Quantization Playground

A tiny C library that implements block‑wise 8‑bit (q8_0) and 4‑bit (q4_0) quantization in the style of the GGML. The core logic lives in include/quantization.h and src/quantization.c. A short evaluation script is provided in test/main.c.

## Quick Start

```bash
# Build the library + demo
make

# Run the demo
./build/demo
```

The executable build/demo is built from test/main.c. It prints a per‑array report that includes:

```plaintext
[array 0] N=4096, blocks=128, original_size=16.000 KB
   Q4_0:  size=2.547 KB, B/W=5.09375, MAE=0.331545, MSE=0.152191, MaxAbs=0.713930
   Q8_0:  size=4.524 KB, B/W=9.04883, MAE=0.018560, MSE=0.000472, MaxAbs=0.039356
```

## API Reference

The library exposes a small, generic interface.

```c
/* ---- Allocation helpers ------------------------------------------------ */
quantized_array_t *allocate_q8_0_array(uint64_t num_elements,
                                       uint64_t block_size);

quantized_array_t *allocate_q4_0_array(uint64_t num_elements,
                                       uint64_t block_size);

/* ---- Core quantisation / dequantisation ------------------------------- */
int quantize(const float *float_array,
             uint64_t num_elements,
             uint8_t quantized_type,          /* 0 = q8_0, 1 = q4_0 */
             quantized_array_t **quantized_array);   /* out */

int dequantize(const quantized_array_t *quantized_array,
               float *float_array);            /* out */

/* ---- Helpers ----------------------------------------------------------- */
void free_quantized_array(quantized_array_t *quantized_array);
size_t get_quantized_array_size(const quantized_array_t *quantized_array);

/* ---- Quantised array struct -------------------------------------------- */
typedef struct {
    uint8_t  quantized_type;  /* 0: q8_0, 1: q4_0, … */
    uint64_t num_elements;    /* original float count            */
    uint64_t num_blocks;      /* number of blocks                */
    uint64_t block_size;      /* elements per block              */
    float  *scales;           /* length = num_blocks             */
    int8_t *data;             /* packed quantised data           */
} quantized_array_t;
```

### Example Usage

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

The quantize function automatically allocates the quantized array for you; you only need to provide the pointer to the resulting quantized_array_t*.

## License

MIT License – see the LICENSE file for details.