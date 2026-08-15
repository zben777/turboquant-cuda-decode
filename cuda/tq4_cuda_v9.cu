#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

constexpr int Q_HEADS = 32, KV_HEADS = 8, GQA = 4;
constexpr int HEAD_DIM = 128, BLOCK_SIZE = 16, NUM_SPLITS = 32;
constexpr int K_BYTES = 64, DATA_BYTES = 128, SLOT_SIZE = 134;
constexpr int NUM_FIELDS = 3, K_NORM = 0, V_SCALE = 1, V_ZERO = 2;
constexpr int META_OFFSET = BLOCK_SIZE * KV_HEADS * DATA_BYTES;
constexpr int THREADS = 128, TILE = 16, MMA_N = 8, SPLIT_TOKENS = 128;
constexpr float ATTN_SCALE = 0.08838834764831845f;
constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float load_meta(const uint8_t *cache, int64_t base, int kvh, int field,
                                           int pos) {
    const int i = (kvh * NUM_FIELDS + field) * BLOCK_SIZE + pos;
    return __half2float(*reinterpret_cast<const __half *>(cache + base + META_OFFSET + i * 2));
}

__device__ __forceinline__ uint32_t pack_half2(__half lo, __half hi) {
    union {
        __half2 value;
        uint32_t bits;
    } packed;
    packed.value = __halves2half2(lo, hi);
    return packed.bits;
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// PTX m16n8k16 maps each lane to two rows and two adjacent columns. QK uses
// token as M and query head as N, so four GQA heads occupy half of N=8.
__device__ __forceinline__ void load_qk_fragments(const __half (&q)[GQA][HEAD_DIM],
                                                   const __half (&k)[TILE][HEAD_DIM], int k_base,
                                                   int lane, uint32_t (&a)[4], uint32_t (&b)[2]) {
    const int group = lane >> 2;
    const int thread = lane & 3;
    const int k0 = k_base + 2 * thread;
    a[0] = pack_half2(k[group][k0], k[group][k0 + 1]);
    a[1] = pack_half2(k[group + 8][k0], k[group + 8][k0 + 1]);
    a[2] = pack_half2(k[group][k0 + 8], k[group][k0 + 9]);
    a[3] = pack_half2(k[group + 8][k0 + 8], k[group + 8][k0 + 9]);
    if (group < GQA) {
        b[0] = pack_half2(q[group][k0], q[group][k0 + 1]);
        b[1] = pack_half2(q[group][k0 + 8], q[group][k0 + 9]);
    } else {
        b[0] = b[1] = 0;
    }
}

// PV is transposed for the same reason: dimension is M and query head is N.
__device__ __forceinline__ void load_pv_fragments(const __half (&v)[TILE][HEAD_DIM],
                                                   const __half (&p)[TILE][MMA_N], int d_base,
                                                   int lane, uint32_t (&a)[4], uint32_t (&b)[2]) {
    const int group = lane >> 2;
    const int thread = lane & 3;
    const int t0 = 2 * thread;
    const int d0 = d_base + group;
    const int d1 = d0 + 8;
    a[0] = pack_half2(v[t0][d0], v[t0 + 1][d0]);
    a[1] = pack_half2(v[t0][d1], v[t0 + 1][d1]);
    a[2] = pack_half2(v[t0 + 8][d0], v[t0 + 9][d0]);
    a[3] = pack_half2(v[t0 + 8][d1], v[t0 + 9][d1]);
    if (group < GQA) {
        b[0] = pack_half2(p[t0][group], p[t0 + 1][group]);
        b[1] = pack_half2(p[t0 + 8][group], p[t0 + 9][group]);
    } else {
        b[0] = b[1] = 0;
    }
}

__global__ __launch_bounds__(THREADS) void tq4_cuda_v9_kernel(
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

    __shared__ __align__(32) __half q_s[GQA][HEAD_DIM];
    __shared__ __align__(32) __half k_s[TILE][HEAD_DIM];
    __shared__ __align__(32) __half v_s[TILE][HEAD_DIM];
    __shared__ __align__(32) __half p_s[TILE][MMA_N];
    __shared__ int64_t data_base[TILE];
    __shared__ float k_norm[TILE], v_scale[TILE], v_zero[TILE];
    __shared__ float tile_alpha[GQA], split_inv_l[GQA];

    for (int i = tid; i < GQA * HEAD_DIM; i += THREADS) {
        const int q = i / HEAD_DIM, d = i % HEAD_DIM;
        const int qh = kvh * GQA + q;
        q_s[q][d] = __float2half(q_rot[(static_cast<int64_t>(b) * Q_HEADS + qh) * HEAD_DIM + d]);
    }
    __syncthreads();

    const float centroid_lane = lane < 16 ? centroids[lane] : 0.0f;
    float out0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Warp 0 owns two online-softmax states per lane class. Lanes with the
    // same lane%4 collectively hold all 16 tokens of one pair of GQA heads.
    float running_m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float running_l[2] = {0.0f, 0.0f};

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
        __syncthreads();

        if (warp == 0) {
            float qk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < HEAD_DIM; k += 16) {
                uint32_t a[4], q[2];
                load_qk_fragments(q_s, k_s, k, lane, a, q);
                mma_m16n8k16(qk, a, q);
            }

            const int group = lane >> 2;
            const int thread = lane & 3;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float score0 = qk[h] * ATTN_SCALE * RCP_LN2;
                float score8 = qk[h + 2] * ATTN_SCALE * RCP_LN2;
                float tile_m = fmaxf(score0, score8);
#pragma unroll
                for (int d = 4; d <= 16; d <<= 1)
                    tile_m = fmaxf(tile_m, __shfl_xor_sync(FULL_MASK, tile_m, d));

                const float new_m = fmaxf(running_m[h], tile_m);
                const float alpha = running_l[h] == 0.0f
                                        ? 0.0f
                                        : exp2f(running_m[h] - new_m);
                const float p0 = exp2f(score0 - new_m);
                const float p8 = exp2f(score8 - new_m);
                float tile_l = p0 + p8;
#pragma unroll
                for (int d = 4; d <= 16; d <<= 1)
                    tile_l += __shfl_xor_sync(FULL_MASK, tile_l, d);

                running_l[h] = running_l[h] * alpha + tile_l;
                running_m[h] = new_m;
                if (thread < 2) {
                    const int q = 2 * thread + h;
                    p_s[group][q] = __float2half(p0);
                    p_s[group + 8][q] = __float2half(p8);
                    if (group == 0)
                        tile_alpha[q] = alpha;
                }
            }
        }
        __syncthreads();

        const int q0 = 2 * (lane & 3);
        const float alpha0 = q0 < GQA ? tile_alpha[q0] : 0.0f;
        const float alpha1 = q0 + 1 < GQA ? tile_alpha[q0 + 1] : 0.0f;
        out0[0] *= alpha0;
        out0[1] *= alpha1;
        out0[2] *= alpha0;
        out0[3] *= alpha1;
        out1[0] *= alpha0;
        out1[1] *= alpha1;
        out1[2] *= alpha0;
        out1[3] *= alpha1;
        uint32_t a[4], prob[2];
        const int d_base = warp * 32;
        load_pv_fragments(v_s, p_s, d_base, lane, a, prob);
        mma_m16n8k16(out0, a, prob);
        load_pv_fragments(v_s, p_s, d_base + 16, lane, a, prob);
        mma_m16n8k16(out1, a, prob);
    }

    if (warp == 0 && (lane >> 2) == 0 && (lane & 3) < 2) {
        const int thread = lane & 3;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int q = 2 * thread + h;
            split_inv_l[q] = 1.0f / running_l[h];
            const int qh = kvh * GQA + q;
            const int64_t head_stride = static_cast<int64_t>(NUM_SPLITS) * (HEAD_DIM + 1);
            const int64_t out = (static_cast<int64_t>(b) * Q_HEADS + qh) * head_stride +
                                static_cast<int64_t>(sid) * (HEAD_DIM + 1);
            mid_o[out + HEAD_DIM] = running_m[h] * LN2 + logf(running_l[h]);
        }
    }
    __syncthreads();

    const int thread = lane & 3;
    if (thread < 2) {
        const int group = lane >> 2;
        const int q0 = 2 * thread;
        const int q1 = q0 + 1;
        const int d_base = warp * 32;
        const int64_t head_stride = static_cast<int64_t>(NUM_SPLITS) * (HEAD_DIM + 1);
        const int64_t out0_base =
            (static_cast<int64_t>(b) * Q_HEADS + kvh * GQA + q0) * head_stride +
            static_cast<int64_t>(sid) * (HEAD_DIM + 1);
        const int64_t out1_base =
            (static_cast<int64_t>(b) * Q_HEADS + kvh * GQA + q1) * head_stride +
            static_cast<int64_t>(sid) * (HEAD_DIM + 1);
        const float inv0 = split_inv_l[q0], inv1 = split_inv_l[q1];
        mid_o[out0_base + d_base + group] = out0[0] * inv0;
        mid_o[out1_base + d_base + group] = out0[1] * inv1;
        mid_o[out0_base + d_base + group + 8] = out0[2] * inv0;
        mid_o[out1_base + d_base + group + 8] = out0[3] * inv1;
        mid_o[out0_base + d_base + group + 16] = out1[0] * inv0;
        mid_o[out1_base + d_base + group + 16] = out1[1] * inv1;
        mid_o[out0_base + d_base + group + 24] = out1[2] * inv0;
        mid_o[out1_base + d_base + group + 24] = out1[3] * inv1;
    }
}

} // namespace

torch::Tensor tq4_cuda_v9_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
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
    tq4_cuda_v9_kernel<<<dim3(B, KV_HEADS, NUM_SPLITS), THREADS, 0, stream>>>(
        q_rot.data_ptr<float>(), kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), centroids.data_ptr<float>(), mid_o.data_ptr<float>(), B,
        block_table.size(1), kv_cache.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mid_o;
}
