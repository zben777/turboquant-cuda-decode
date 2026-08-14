// V6 Stage1: V5 plus vectorized packed K/V decode.
#define TQ4_DIRECT_WRITE 1
#define TQ4_VECTOR_DECODE 1
#define TQ4_STAGE1_KERNEL tq4_cuda_v6_kernel
#define TQ4_STAGE1_ENTRY tq4_cuda_v6_cuda
#include "tq4_cuda_stage1_template.cuh"
