// V7 Stage1: V6 plus fused tile barriers.
#define TQ4_DIRECT_WRITE 1
#define TQ4_VECTOR_DECODE 1
#define TQ4_FUSED_TILE_BARRIER 1
#define TQ4_STAGE1_KERNEL tq4_cuda_v7_stage1_kernel
#define TQ4_STAGE1_ENTRY tq4_cuda_v7_stage1_cuda
#include "tq4_cuda_stage1_template.cuh"

namespace {

__global__
__launch_bounds__(THREADS) void tq4_cuda_v7_stage2_kernel(const float *__restrict__ mid_o,
                                                          float *__restrict__ output,
                                                          float *__restrict__ lse, int batch_size) {
    const int b = blockIdx.x;
    const int qh = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    if (b >= batch_size)
        return;
    __shared__ float weights[NUM_SPLITS];
    __shared__ float inv_sum;
    __shared__ float merged_lse;
    const int64_t head_stride = static_cast<int64_t>(NUM_SPLITS) * (HEAD_DIM + 1);
    const int64_t mid_base = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride;
    if (tid < NUM_SPLITS) {
        weights[tid] = mid_o[mid_base + static_cast<int64_t>(tid) * (HEAD_DIM + 1) + HEAD_DIM];
    }
    __syncthreads();
    if (tid < NUM_SPLITS) {
        const float split_lse = weights[lane];
        const float max_lse = __shfl_sync(FULL_MASK, warp_max(split_lse), 0);
        const float weight = expf(split_lse - max_lse);
        const float weight_sum = __shfl_sync(FULL_MASK, warp_sum(weight), 0);
        weights[lane] = weight;
        if (lane == 0) {
            inv_sum = 1.0f / weight_sum;
            merged_lse = max_lse + logf(weight_sum);
        }
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < NUM_SPLITS; ++split) {
        acc +=
            weights[split] * mid_o[mid_base + static_cast<int64_t>(split) * (HEAD_DIM + 1) + tid];
    }
    output[(static_cast<int64_t>(b) * Q_HEADS + qh) * HEAD_DIM + tid] = acc * inv_sum;
    if (tid == 0) {
        lse[static_cast<int64_t>(b) * Q_HEADS + qh] = merged_lse;
    }
}

} // namespace

torch::Tensor tq4_cuda_v7_stage2_cuda(torch::Tensor mid_o, torch::Tensor output,
                                      torch::Tensor lse) {
    TORCH_CHECK(mid_o.is_cuda() && mid_o.is_contiguous(), "invalid mid_o");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "invalid output");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous(), "invalid lse");
    TORCH_CHECK(mid_o.scalar_type() == torch::kFloat32, "mid_o must be fp32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be fp32");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be fp32");
    const int B = mid_o.size(0);
    TORCH_CHECK(mid_o.sizes() == torch::IntArrayRef({B, Q_HEADS, NUM_SPLITS, HEAD_DIM + 1}),
                "invalid mid_o shape");
    TORCH_CHECK(output.sizes() == torch::IntArrayRef({B, Q_HEADS, HEAD_DIM}),
                "invalid output shape");
    TORCH_CHECK(lse.sizes() == torch::IntArrayRef({B, Q_HEADS}), "invalid lse shape");
    c10::cuda::CUDAGuard guard(mid_o.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tq4_cuda_v7_stage2_kernel<<<dim3(B, Q_HEADS), THREADS, 0, stream>>>(
        mid_o.data_ptr<float>(), output.data_ptr<float>(), lse.data_ptr<float>(), B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor tq4_cuda_v7_full_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
                                    torch::Tensor block_table, torch::Tensor seq_lens,
                                    torch::Tensor centroids, torch::Tensor mid_o,
                                    torch::Tensor output, torch::Tensor lse) {
    tq4_cuda_v7_stage1_cuda(q_rot, kv_cache, block_table, seq_lens, centroids, mid_o);
    return tq4_cuda_v7_stage2_cuda(mid_o, output, lse);
}
