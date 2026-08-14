#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

constexpr int Q_HEADS = 32;
constexpr int KV_HEADS = 8;
constexpr int GQA = 4;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_SIZE = 16;
constexpr int NUM_SPLITS = 32;
constexpr int K_BYTES = 64;
constexpr int DATA_BYTES_PER_SLOT = 128;
constexpr int SLOT_SIZE = 134;
constexpr int NUM_SOA_FIELDS = 3;
constexpr int SOA_K_NORM = 0;
constexpr int SOA_V_SCALE = 1;
constexpr int SOA_V_ZERO = 2;
constexpr int META_REGION_OFFSET = BLOCK_SIZE * KV_HEADS * DATA_BYTES_PER_SLOT;
constexpr int THREADS = 128;
constexpr float ATTN_SCALE = 0.08838834764831845f;
constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

__device__ __forceinline__ float load_meta(const uint8_t *cache, int64_t block_base, int kv_head,
                                           int field, int token_offset) {
    const int index = (kv_head * NUM_SOA_FIELDS + field) * BLOCK_SIZE + token_offset;
    const __half *ptr =
        reinterpret_cast<const __half *>(cache + block_base + META_REGION_OFFSET + index * 2);
    return __half2float(*ptr);
}

__global__ __launch_bounds__(THREADS) void tq4_cuda_v2_kernel(
    const float *__restrict__ q_rot, const uint8_t *__restrict__ kv_cache,
    const int32_t *__restrict__ block_table, const int32_t *__restrict__ seq_lens,
    const float *__restrict__ centroids, float *__restrict__ mid_o, int batch_size,
    int blocks_per_seq, int64_t cache_block_stride) {
    const int b = blockIdx.x;
    const int kvh = blockIdx.y;
    const int sid = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (b >= batch_size) {
        return;
    }
    const int qh = kvh * GQA + warp;
    const int seq_len = seq_lens[b];
    const int split_len = (seq_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int split_start = sid * split_len;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) {
        return;
    }
    float centroid_lane = lane < 16 ? centroids[lane] : 0.0f;
    float q[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = lane + j * 32;
        q[j] = q_rot[(static_cast<int64_t>(b) * Q_HEADS + qh) * HEAD_DIM + d];
    }
    float running_m = -CUDART_INF_F;
    float running_l = 0.0f;
    for (int token = split_start; token < split_end; ++token) {
        const int page_idx = token / BLOCK_SIZE;
        const int page_off = token % BLOCK_SIZE;
        int block_num = 0;
        if (lane == 0) {
            block_num = block_table[static_cast<int64_t>(b) * blocks_per_seq + page_idx];
        }
        block_num = __shfl_sync(FULL_MASK, block_num, 0);
        const int64_t block_base = static_cast<int64_t>(block_num) * cache_block_stride;
        const int64_t data_base = block_base +
                                  static_cast<int64_t>(page_off) * KV_HEADS * DATA_BYTES_PER_SLOT +
                                  static_cast<int64_t>(kvh) * DATA_BYTES_PER_SLOT;
        float k_norm =
            lane == 0 ? load_meta(kv_cache, block_base, kvh, SOA_K_NORM, page_off) : 0.0f;
        k_norm = __shfl_sync(FULL_MASK, k_norm, 0);
        float dot = 0.0f;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned packed = 0;
            if (lane < 16) {
                packed = kv_cache[data_base + j * 16 + lane];
            }
            packed = __shfl_sync(FULL_MASK, packed, lane >> 1);
            const int index = (packed >> ((lane & 1) * 4)) & 0xF;
            const float centroid = __shfl_sync(FULL_MASK, centroid_lane, index);
            dot += q[j] * centroid * k_norm;
        }
        dot = warp_sum(dot);
        float alpha = 0.0f;
        float weight = 0.0f;
        if (lane == 0) {
            const float score = dot * ATTN_SCALE * RCP_LN2;
            const float new_m = fmaxf(running_m, score);
            alpha = running_l == 0.0f ? 0.0f : exp2f(running_m - new_m);
            weight = exp2f(score - new_m);
            running_l = running_l * alpha + weight;
            running_m = new_m;
        }
        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        weight = __shfl_sync(FULL_MASK, weight, 0);
        float v_scale =
            lane == 0 ? load_meta(kv_cache, block_base, kvh, SOA_V_SCALE, page_off) : 0.0f;
        float v_zero =
            lane == 0 ? load_meta(kv_cache, block_base, kvh, SOA_V_ZERO, page_off) : 0.0f;
        v_scale = __shfl_sync(FULL_MASK, v_scale, 0);
        v_zero = __shfl_sync(FULL_MASK, v_zero, 0);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned packed = 0;
            if (lane < 16) {
                packed = kv_cache[data_base + K_BYTES + j * 16 + lane];
            }
            packed = __shfl_sync(FULL_MASK, packed, lane >> 1);
            const int index = (packed >> ((lane & 1) * 4)) & 0xF;
            const float value = index * v_scale + v_zero;
            acc[j] = acc[j] * alpha + weight * value;
        }
    }
    float inv_l = 0.0f;
    float lse = 0.0f;
    if (lane == 0) {
        const float safe_l = running_l > 0.0f ? running_l : 1.0f;
        inv_l = 1.0f / safe_l;
        lse = running_m * LN2 + logf(safe_l);
    }
    inv_l = __shfl_sync(FULL_MASK, inv_l, 0);
    const int64_t head_stride = static_cast<int64_t>(NUM_SPLITS) * (HEAD_DIM + 1);
    const int64_t out_base = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride +
                             static_cast<int64_t>(sid) * (HEAD_DIM + 1);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        mid_o[out_base + lane + j * 32] = acc[j] * inv_l;
    }
    if (lane == 0) {
        mid_o[out_base + HEAD_DIM] = lse;
    }
}

} // namespace

torch::Tensor tq4_cuda_v2_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
                               torch::Tensor block_table, torch::Tensor seq_lens,
                               torch::Tensor centroids, torch::Tensor mid_o) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.is_contiguous(), "invalid q_rot");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous(), "invalid cache");
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(), "invalid block table");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous(), "invalid seq_lens");
    TORCH_CHECK(centroids.is_cuda() && centroids.is_contiguous(), "invalid centroids");
    TORCH_CHECK(mid_o.is_cuda() && mid_o.is_contiguous(), "invalid mid_o");
    TORCH_CHECK(q_rot.scalar_type() == torch::kFloat32, "q_rot must be fp32");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block table must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be fp32");
    TORCH_CHECK(mid_o.scalar_type() == torch::kFloat32, "mid_o must be fp32");
    TORCH_CHECK(q_rot.size(1) == Q_HEADS && q_rot.size(2) == HEAD_DIM, "requires Hq=32,D=128");
    TORCH_CHECK(kv_cache.size(1) == BLOCK_SIZE && kv_cache.size(2) == KV_HEADS &&
                    kv_cache.size(3) == SLOT_SIZE,
                "requires cache [blocks,16,8,134]");
    const int B = q_rot.size(0);
    TORCH_CHECK(mid_o.sizes() == torch::IntArrayRef({B, Q_HEADS, NUM_SPLITS, HEAD_DIM + 1}),
                "invalid mid_o shape");
    c10::cuda::CUDAGuard guard(q_rot.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tq4_cuda_v2_kernel<<<dim3(B, KV_HEADS, NUM_SPLITS), THREADS, 0, stream>>>(
        q_rot.data_ptr<float>(), kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(), mid_o.data_ptr<float>(), B,
        block_table.size(1), kv_cache.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mid_o;
}
