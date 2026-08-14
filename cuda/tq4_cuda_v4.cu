// V4 Stage1: fixed-workload specialization of the V3 Tensor Core path.
#define TQ4_STAGE1_KERNEL tq4_cuda_v4_kernel
#define TQ4_STAGE1_ENTRY tq4_cuda_v4_cuda
#include "tq4_cuda_stage1_template.cuh"
