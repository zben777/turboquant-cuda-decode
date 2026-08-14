#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdint>

#ifndef TQ4_STAGE1_KERNEL
#error "Define TQ4_STAGE1_KERNEL before including tq4_cuda_stage1_template.cuh"
#endif

#ifndef TQ4_STAGE1_ENTRY
#error "Define TQ4_STAGE1_ENTRY before including tq4_cuda_stage1_template.cuh"
#endif

namespace {
using namespace nvcuda;

constexpr int Q_HEADS = 32, KV_HEADS = 8, GQA = 4;
constexpr int HEAD_DIM = 128, BLOCK_SIZE = 16, NUM_SPLITS = 32;
constexpr int K_BYTES = 64, DATA_BYTES = 128, SLOT_SIZE = 134;
constexpr int NUM_FIELDS = 3, K_NORM = 0, V_SCALE = 1, V_ZERO = 2;
constexpr int META_OFFSET = BLOCK_SIZE * KV_HEADS * DATA_BYTES;
constexpr int THREADS = 128, TILE = 16, SPLIT_TOKENS = 128;
constexpr float ATTN_SCALE = 0.08838834764831845f;
constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, d));
    }
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        x += __shfl_down_sync(FULL_MASK, x, d);
    }
    return x;
}

__device__ __forceinline__ float load_meta(const uint8_t *cache, int64_t base, int kvh, int field,
                                           int pos) {
    const int i = (kvh * NUM_FIELDS + field) * BLOCK_SIZE + pos;
    return __half2float(*reinterpret_cast<const __half *>(cache + base + META_OFFSET + i * 2));
}

#ifdef TQ4_DIRECT_WRITE
struct WmmaScratch {
    float qk[TILE][TILE];
};
#else
union WmmaScratch {
    float qk[TILE][TILE];
    float output[TILE][HEAD_DIM];
};
#endif

__global__ __launch_bounds__(THREADS) void TQ4_STAGE1_KERNEL(
    const float *__restrict__ q_rot, const uint8_t *__restrict__ cache,
    const int32_t *__restrict__ block_table, const int32_t *__restrict__ seq_lens,
    const float *__restrict__ centroids, float *__restrict__ mid_o, int batch_size,
    int blocks_per_seq, int64_t cache_stride) {
    const int b = blockIdx.x, kvh = blockIdx.y, sid = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (b >= batch_size)
        return;
    const int seq_len = seq_lens[b];
    const int split_len = (seq_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int split_start = sid * split_len;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_end - split_start != SPLIT_TOKENS || (split_start & 15))
        return;
    __shared__ __align__(32) __half q_s[TILE][HEAD_DIM];
    __shared__ __align__(32) __half k_s[TILE][HEAD_DIM];
    __shared__ __align__(32) __half v_s[TILE][HEAD_DIM];
    __shared__ __align__(32) __half p_s[TILE][TILE];
    __shared__ __align__(32) WmmaScratch scratch;
    __shared__ int64_t data_base[TILE];
    __shared__ float k_norm[TILE], v_scale[TILE], v_zero[TILE];
    __shared__ float tile_alpha[TILE];
#ifdef TQ4_DIRECT_WRITE
    __shared__ float split_inv_l[GQA];
#else
    __shared__ float split_lse[GQA], split_inv_l[GQA];
#endif
    // Only the four real query rows are consumed by softmax and output.
    // The remaining WMMA rows may contain arbitrary values.
    for (int i = tid; i < GQA * HEAD_DIM; i += THREADS) {
        const int row = i / HEAD_DIM, d = i % HEAD_DIM;
        const int qh = kvh * GQA + row;
        q_s[row][d] = __float2half(q_rot[(static_cast<int64_t>(b) * Q_HEADS + qh) * HEAD_DIM + d]);
    }
    for (int i = tid; i < TILE * TILE; i += THREADS) {
        reinterpret_cast<__half *>(p_s)[i] = __float2half(0.0f);
    }
    __syncthreads();
    const float centroid_lane = lane < 16 ? centroids[lane] : 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out0, out1;
    wmma::fill_fragment(out0, 0.0f);
    wmma::fill_fragment(out1, 0.0f);
    float running_m = -CUDART_INF_F;
    float running_l = 0.0f;

#pragma unroll
    for (int tile = 0; tile < SPLIT_TOKENS; tile += TILE) {
        int block = 0;
        if (warp == 0) {
            if (lane == 0) {
                block = block_table[static_cast<int64_t>(b) * blocks_per_seq +
                                    (split_start + tile) / BLOCK_SIZE];
            }
            block = __shfl_sync(FULL_MASK, block, 0);
            if (lane < TILE) {
                const int64_t base = static_cast<int64_t>(block) * cache_stride;
                data_base[lane] = base + static_cast<int64_t>(lane) * KV_HEADS * DATA_BYTES +
                                  static_cast<int64_t>(kvh) * DATA_BYTES;
                k_norm[lane] = load_meta(cache, base, kvh, K_NORM, lane);
                v_scale[lane] = load_meta(cache, base, kvh, V_SCALE, lane);
                v_zero[lane] = load_meta(cache, base, kvh, V_ZERO, lane);
            }
        }
        __syncthreads();

#ifdef TQ4_VECTOR_DECODE
        constexpr int PACKED_WORDS = K_BYTES / 4;
        for (int i = tid; i < TILE * PACKED_WORDS; i += THREADS) {
            const int t = i / PACKED_WORDS, word = i % PACKED_WORDS;
            const int byte = word * 4;
            const uint32_t packed_k4 =
                *reinterpret_cast<const uint32_t *>(cache + data_base[t] + byte);
            const uint32_t packed_v4 =
                *reinterpret_cast<const uint32_t *>(cache + data_base[t] + K_BYTES + byte);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t packed_k = static_cast<uint8_t>(packed_k4 >> (j * 8));
                const uint8_t packed_v = static_cast<uint8_t>(packed_v4 >> (j * 8));
                const float c_lo = __shfl_sync(FULL_MASK, centroid_lane, packed_k & 15);
                const float c_hi = __shfl_sync(FULL_MASK, centroid_lane, packed_k >> 4);
                const int d = byte * 2 + j * 2;
                *reinterpret_cast<__half2 *>(&k_s[t][d]) =
                    __halves2half2(__float2half(c_lo * k_norm[t]), __float2half(c_hi * k_norm[t]));
                *reinterpret_cast<__half2 *>(&v_s[t][d]) =
                    __halves2half2(__float2half((packed_v & 15) * v_scale[t] + v_zero[t]),
                                   __float2half((packed_v >> 4) * v_scale[t] + v_zero[t]));
            }
        }
#else
        for (int i = tid; i < TILE * K_BYTES; i += THREADS) {
            const int t = i / K_BYTES, byte = i % K_BYTES;
            const uint8_t packed_k = cache[data_base[t] + byte];
            const uint8_t packed_v = cache[data_base[t] + K_BYTES + byte];
            const int k_lo = packed_k & 15, k_hi = packed_k >> 4;
            const float c_lo = __shfl_sync(FULL_MASK, centroid_lane, k_lo);
            const float c_hi = __shfl_sync(FULL_MASK, centroid_lane, k_hi);
            const int d = byte * 2;
            k_s[t][d] = __float2half(c_lo * k_norm[t]);
            k_s[t][d + 1] = __float2half(c_hi * k_norm[t]);
            v_s[t][d] = __float2half((packed_v & 15) * v_scale[t] + v_zero[t]);
            v_s[t][d + 1] = __float2half((packed_v >> 4) * v_scale[t] + v_zero[t]);
        }
#endif
        __syncthreads();
        if (warp == 0) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk;
            wmma::fill_fragment(qk, 0.0f);
#pragma unroll
            for (int k = 0; k < HEAD_DIM; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
                wmma::load_matrix_sync(qf, &q_s[0][k], HEAD_DIM);
                wmma::load_matrix_sync(kf, &k_s[0][k], HEAD_DIM);
                wmma::mma_sync(qk, qf, kf, qk);
            }
            wmma::store_matrix_sync(&scratch.qk[0][0], qk, TILE, wmma::mem_row_major);
        }
        __syncthreads();
        float score = -CUDART_INF_F;
        if (lane < TILE) {
            score = scratch.qk[warp][lane] * ATTN_SCALE * RCP_LN2;
        }
        const float tile_m = __shfl_sync(FULL_MASK, warp_max(score), 0);
        const float new_m = fmaxf(running_m, tile_m);
        const float alpha = running_l == 0.0f ? 0.0f : exp2f(running_m - new_m);
        const float p = lane < TILE ? exp2f(score - new_m) : 0.0f;
        const float tile_l = __shfl_sync(FULL_MASK, warp_sum(p), 0);
        running_l = running_l * alpha + tile_l;
        running_m = new_m;
        if (lane < TILE)
            p_s[warp][lane] = __float2half(p);
        if (lane == 0)
            tile_alpha[warp] = alpha;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < out0.num_elements; ++i) {
            const int row = (lane >> 2) + ((i & 2) ? 8 : 0);
            const float row_alpha = row < GQA ? tile_alpha[row] : 0.0f;
            out0.x[i] *= row_alpha;
            out1.x[i] *= row_alpha;
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, &p_s[0][0], TILE);
        const int col = warp * 32;
        wmma::load_matrix_sync(vf, &v_s[0][col], HEAD_DIM);
        wmma::mma_sync(out0, pf, vf, out0);
        wmma::load_matrix_sync(vf, &v_s[0][col + 16], HEAD_DIM);
        wmma::mma_sync(out1, pf, vf, out1);
#ifndef TQ4_FUSED_TILE_BARRIER
        __syncthreads();
#endif
    }
    const int64_t head_stride = static_cast<int64_t>(NUM_SPLITS) * (HEAD_DIM + 1);

#ifdef TQ4_DIRECT_WRITE
    if (lane == 0) {
        split_inv_l[warp] = 1.0f / running_l;
        const int qh = kvh * GQA + warp;
        const int64_t out = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride +
                            static_cast<int64_t>(sid) * (HEAD_DIM + 1);
        mid_o[out + HEAD_DIM] = running_m * LN2 + logf(running_l);
    }
    __syncthreads();
    if (lane < 16) {
        const int row = lane >> 2;
        const int qh = kvh * GQA + row;
        const int64_t out = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride +
                            static_cast<int64_t>(sid) * (HEAD_DIM + 1);
        const float inv_l = split_inv_l[row];
#pragma unroll
        for (int i = 0; i < out0.num_elements; ++i) {
            if ((i & 2) == 0) {
                const int fragment_col = 2 * (lane & 3) + (i & 1) + ((i & 4) ? 8 : 0);
                const int d0 = warp * 32 + fragment_col;
                mid_o[out + d0] = out0.x[i] * inv_l;
                mid_o[out + d0 + 16] = out1.x[i] * inv_l;
            }
        }
    }
#else
    if (lane == 0) {
        split_inv_l[warp] = 1.0f / running_l;
        split_lse[warp] = running_m * LN2 + logf(running_l);
    }
    const int col = warp * 32;
    wmma::store_matrix_sync(&scratch.output[0][col], out0, HEAD_DIM, wmma::mem_row_major);
    wmma::store_matrix_sync(&scratch.output[0][col + 16], out1, HEAD_DIM, wmma::mem_row_major);
    __syncthreads();
    for (int q = 0; q < GQA; ++q) {
        const int qh = kvh * GQA + q;
        const int64_t out = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride +
                            static_cast<int64_t>(sid) * (HEAD_DIM + 1);
        mid_o[out + tid] = scratch.output[q][tid] * split_inv_l[q];
        if (tid == 0)
            mid_o[out + HEAD_DIM] = split_lse[q];
    }
#endif
}
} // namespace

torch::Tensor TQ4_STAGE1_ENTRY(torch::Tensor q_rot, torch::Tensor kv_cache,
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
    TQ4_STAGE1_KERNEL<<<dim3(B, KV_HEADS, NUM_SPLITS), THREADS, 0, stream>>>(
        q_rot.data_ptr<float>(), kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(), mid_o.data_ptr<float>(), B,
        block_table.size(1), kv_cache.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mid_o;
}
