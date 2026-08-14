// V5 Stage1: V4 plus direct WMMA fragment writeback.
#define TQ4_DIRECT_WRITE 1
#define TQ4_STAGE1_KERNEL tq4_cuda_v5_kernel
#define TQ4_STAGE1_ENTRY tq4_cuda_v5_cuda
#include "tq4_cuda_stage1_template.cuh"
