import sys
from pathlib import Path

import torch


# ------------------------------------------------------------
# Import upstream reference code
#
# 当前文件:
#   baseline/common.py
#
# upstream reference:
#   reference/centroids.py
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference"

sys.path.insert(0, str(REFERENCE_DIR))

from centroids import get_centroids


# ============================================================
# 1. Fixed workload
# ============================================================

# 我们现在只研究:
#
#   turboquant_4bit_nc
#
# 并固定 Qwen3-4B attention shape。
#
# 后续 Triton V2 / CUDA V1 / CUDA V2 ...
# 全部使用同一组参数。

BATCH_SIZE = 64
CONTEXT_LEN = 4096

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8

HEAD_DIM = 128

GQA_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS


# Paged KV Cache
BLOCK_SIZE = 16

# split-KV
NUM_KV_SPLITS = 32


# ============================================================
# 2. turboquant_4bit_nc
# ============================================================

MSE_BITS = 4
VALUE_BITS = 4

NORM_CORRECTION = True

N_CENTROIDS = 1 << MSE_BITS


# ============================================================
# 3. Packed storage size
# ============================================================

# K:
#
# 128 dimensions * 4 bit
#
# = 512 bit
# = 64 bytes

MSE_BYTES = (HEAD_DIM * MSE_BITS + 7) // 8


# V:
#
# 128 dimensions * 4 bit
#
# = 512 bit
# = 64 bytes

VAL_DATA_BYTES = (HEAD_DIM * VALUE_BITS + 7) // 8


KEY_DATA_BYTES = MSE_BYTES


# SoA data region 中，
# 每个 token / KV head 只放:
#
# [packed K | packed V]
#
# metadata 不放这里。

DATA_BYTES_PER_SLOT = KEY_DATA_BYTES + VAL_DATA_BYTES


# ============================================================
# 4. SoA metadata
# ============================================================

# 对 turboquant_4bit_nc:
#
# field 0:
#   K norm     FP16
#
# field 1:
#   V scale    FP16
#
# field 2:
#   V zero     FP16

NUM_SOA_FIELDS = 3

SOA_K_NORM = 0
SOA_V_SCALE = 1
SOA_V_ZERO = 2


# 每个 metadata field:
#
# FP16 = 2 bytes

META_BYTES_PER_SLOT = NUM_SOA_FIELDS * 2


# Logical storage:
#
# K       64 B
# V       64 B
# K norm   2 B
# V scale  2 B
# V zero   2 B
#
# = 134 B

SLOT_SIZE = DATA_BYTES_PER_SLOT + META_BYTES_PER_SLOT


# upstream 要求 slot size 为偶数。
#
# 134 已经是偶数。

SLOT_SIZE_ALIGNED = SLOT_SIZE + SLOT_SIZE % 2


# ============================================================
# 5. One physical block layout
# ============================================================

# 一个 physical block:
#
# BLOCK_SIZE = 16 tokens
# NUM_KV_HEADS = 8
#
#
# byte 0
# |
# |  DATA REGION
# |
# |  [token][kv_head][K64 | V64]
# |
# |
# |  METADATA REGION
# |
# |  [kv_head][field][token]
# |
# |  field 0 = K norm
# |  field 1 = V scale
# |  field 2 = V zero
# |
# end


# Data:
#
# 16 * 8 * 128
# = 16384 bytes

DATA_REGION_BYTES = BLOCK_SIZE * NUM_KV_HEADS * DATA_BYTES_PER_SLOT


META_REGION_OFFSET = DATA_REGION_BYTES


# Metadata:
#
# 8 heads
# * 3 fields
# * 16 tokens
# * 2 B
#
# = 768 B

META_REGION_BYTES = NUM_KV_HEADS * NUM_SOA_FIELDS * BLOCK_SIZE * 2


# Physical allocation仍然维持:
#
# [BLOCK_SIZE, Hk, slot_size_aligned]
#
# 所以一个 block:
#
# 16 * 8 * 134
# = 17152 B

BLOCK_BYTES = BLOCK_SIZE * NUM_KV_HEADS * SLOT_SIZE_ALIGNED


# ============================================================
# 6. Sequence / physical page counts
# ============================================================

BLOCKS_PER_SEQ = (CONTEXT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE


TOTAL_PHYSICAL_BLOCKS = BATCH_SIZE * BLOCKS_PER_SEQ


# ============================================================
# 7. Pair LUT
# ============================================================


def build_pair_lut(
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Same logical layout as upstream decode_v2.
    pair_lut[i, j] =
        [centroid[i], centroid[j]]
    For 4-bit:
        16 * 16 * 2 * 4 bytes
        = 2048 bytes
    """
    n = centroids.numel()
    lut = torch.empty(
        n,
        n,
        2,
        dtype=torch.float32,
        device=centroids.device,
    )
    c = centroids.float()
    lut[:, :, 0] = c[:, None]
    lut[:, :, 1] = c[None, :]
    return lut.reshape(
        n * n,
        2,
    ).contiguous()


# ============================================================
# 8. block_table
# ============================================================


def build_block_table(
    device: torch.device,
) -> torch.Tensor:
    """
    第一版 baseline 使用最简单 physical layout:
        logical page == physical page
    Sequence 0:
        physical block 0 ... 255
    Sequence 1:
        physical block 256 ... 511
    ...
    shape:
        [B, BLOCKS_PER_SEQ]
    后续可以增加 random-page benchmark，
    测真正 paged/random access 对 L2/DRAM 的影响。
    """
    block_table = torch.arange(
        TOTAL_PHYSICAL_BLOCKS,
        dtype=torch.int32,
        device=device,
    )
    block_table = block_table.reshape(
        BATCH_SIZE,
        BLOCKS_PER_SEQ,
    )
    return block_table


# ============================================================
# 9. seq_lens
# ============================================================


def build_seq_lens(
    device: torch.device,
) -> torch.Tensor:
    return torch.full(
        (BATCH_SIZE,),
        CONTEXT_LEN,
        dtype=torch.int32,
        device=device,
    )


# ============================================================
# 10. Rotated Query
# ============================================================


def build_query_rot(
    device: torch.device,
) -> torch.Tensor:
    """
    第一阶段只测试 TQ Decode Stage1。
    upstream launcher 中:
        query
          ↓
        query.float()
          ↓
        q @ PiT
          ↓
        q_rot
          ↓
        Stage1
    现在我们直接 synthetic 生成 q_rot。
    所以 Q rotation GEMM 不计入第一阶段 kernel latency。
    """
    torch.manual_seed(1234)
    q_rot = torch.randn(
        BATCH_SIZE,
        NUM_Q_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    return q_rot.contiguous()


# ============================================================
# 11. Real Lloyd-Max centroids
# ============================================================


def build_centroids(
    device: torch.device,
) -> torch.Tensor:
    """
    centroid table 不随机。
    直接使用 upstream Lloyd-Max implementation:
        get_centroids(D=128, bits=4)
    """
    centroids = get_centroids(
        HEAD_DIM,
        MSE_BITS,
    )
    return centroids.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()


# ============================================================
# 12. Synthetic SoA KV Cache
# ============================================================


def build_synthetic_soa_kv_cache(
    device: torch.device,
) -> torch.Tensor:
    """
    构造 synthetic TurboQuant SoA KV Cache。
    数据的“值”无意义，
    但是下面这些必须完全正确:
      * allocation size
      * K/V packing byte region
      * metadata offset
      * metadata order
      * metadata dtype
      * paged cache physical layout

    Physical PyTorch tensor shape:
        [num_blocks,
         block_size,
         num_kv_heads,
         slot_size_aligned]

    但 decode_v2 内部把每个 physical block
    看成一整块连续 bytes:
        DATA REGION
        +
        METADATA REGION
    """
    # --------------------------------------------------------
    # Allocate exact physical TQ cache size.
    # --------------------------------------------------------
    kv_cache = torch.empty(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        SLOT_SIZE_ALIGNED,
        dtype=torch.uint8,
        device=device,
    )
    # --------------------------------------------------------
    # Flatten each physical block.
    #
    # shape:
    #
    # [num_blocks, BLOCK_BYTES]
    # --------------------------------------------------------
    block_bytes = kv_cache.view(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_BYTES,
    )
    # --------------------------------------------------------
    # DATA REGION
    #
    # Random uint8 is enough.
    #
    # Every byte naturally represents:
    #
    # K:
    #
    #   low nibble  = centroid index
    #   high nibble = centroid index
    #
    #
    # V:
    #
    #   low nibble  = INT4
    #   high nibble = INT4
    #
    # Therefore:
    #
    # 0 ... 255 random byte
    #
    # perfectly exercises the real unpack path.
    # --------------------------------------------------------
    block_bytes[
        :,
        :DATA_REGION_BYTES,
    ].random_(
        0,
        256,
    )
    # --------------------------------------------------------
    # Metadata region
    #
    # Cannot use random bytes directly.
    #
    # Arbitrary FP16 bit patterns may produce:
    #
    # NaN / Inf
    #
    # which would poison attention softmax.
    # --------------------------------------------------------
    block_bytes[:, META_REGION_OFFSET:].zero_()
    # --------------------------------------------------------
    # Generate valid K norm.
    #
    # Real NC store approximately contains:
    #
    #     ||K|| / ||centroid vector||
    #
    # For performance benchmarking,
    # exact model semantics are unnecessary.
    #
    # We only need finite positive FP16.
    # --------------------------------------------------------
    k_norm = (
        0.5
        + 1.5
        * torch.rand(
            TOTAL_PHYSICAL_BLOCKS,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.float16)
    # --------------------------------------------------------
    # V scale
    #
    # Must be positive.
    # --------------------------------------------------------
    v_scale = (
        0.01
        + 0.09
        * torch.rand(
            TOTAL_PHYSICAL_BLOCKS,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.float16)
    # --------------------------------------------------------
    # V zero / minimum
    # --------------------------------------------------------
    v_zero = (
        -0.5
        + torch.rand(
            TOTAL_PHYSICAL_BLOCKS,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.float16)
    # --------------------------------------------------------
    # Reinterpret same KV cache memory as FP16.
    #
    # BLOCK_BYTES = 17152
    #
    # divisible by 2.
    # --------------------------------------------------------
    block_fp16 = kv_cache.view(torch.float16).view(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_BYTES // 2,
    )
    meta_fp16_offset = META_REGION_OFFSET // 2
    # --------------------------------------------------------
    # Metadata logical view:
    #
    # [physical_block,
    #  kv_head,
    #  field,
    #  token]
    #
    # field:
    #
    # 0 -> K norm
    # 1 -> V scale
    # 2 -> V zero
    # --------------------------------------------------------
    metadata = block_fp16[
        :, meta_fp16_offset : meta_fp16_offset + NUM_KV_HEADS * NUM_SOA_FIELDS * BLOCK_SIZE
    ].view(
        TOTAL_PHYSICAL_BLOCKS,
        NUM_KV_HEADS,
        NUM_SOA_FIELDS,
        BLOCK_SIZE,
    )
    metadata[:, :, SOA_K_NORM, :].copy_(k_norm)
    metadata[:, :, SOA_V_SCALE, :].copy_(v_scale)
    metadata[:, :, SOA_V_ZERO, :].copy_(v_zero)
    return kv_cache


def convert_soa_to_aos_kv_cache(
    soa_cache: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Convert the synthetic SoA cache to the equivalent AoS layout.
    SoA stores the norm-correction factor in the K norm. AoS V1 applies
    that correction while decoding, so its stored raw norm is reconstructed
    from the packed centroid indices. Both layouts then decode identical K/V.
    """
    assert soa_cache.shape == (
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        SLOT_SIZE_ALIGNED,
    )
    aos_cache = torch.empty_like(soa_cache)
    soa_blocks = soa_cache.view(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_BYTES,
    )
    soa_data = soa_blocks[
        :,
        :DATA_REGION_BYTES,
    ].view(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        DATA_BYTES_PER_SLOT,
    )
    soa_meta = (
        soa_cache.view(torch.float16)
        .view(
            TOTAL_PHYSICAL_BLOCKS,
            BLOCK_BYTES // 2,
        )[:, META_REGION_OFFSET // 2 :]
        .view(
            TOTAL_PHYSICAL_BLOCKS,
            NUM_KV_HEADS,
            NUM_SOA_FIELDS,
            BLOCK_SIZE,
        )
    )
    # AoS slot: [K64 | K norm2 | V64 | V scale2 | V zero2].
    aos_cache[..., :MSE_BYTES].copy_(soa_data[..., :MSE_BYTES])
    aos_cache[
        ...,
        MSE_BYTES + 2 : MSE_BYTES + 2 + VAL_DATA_BYTES,
    ].copy_(
        soa_data[
            ...,
            KEY_DATA_BYTES : KEY_DATA_BYTES + VAL_DATA_BYTES,
        ]
    )
    aos_half = aos_cache.view(torch.float16)
    aos_half[..., 65].copy_(
        soa_meta[
            :,
            :,
            SOA_V_SCALE,
            :,
        ].permute(0, 2, 1)
    )
    aos_half[..., 66].copy_(
        soa_meta[
            :,
            :,
            SOA_V_ZERO,
            :,
        ].permute(0, 2, 1)
    )
    # Reconstruct the raw norm expected by AoS V1 in bounded chunks.
    chunk_blocks = 128
    for start in range(
        0,
        TOTAL_PHYSICAL_BLOCKS,
        chunk_blocks,
    ):
        end = min(
            start + chunk_blocks,
            TOTAL_PHYSICAL_BLOCKS,
        )
        packed = soa_data[
            start:end,
            ...,
            :MSE_BYTES,
        ]
        lo = (packed & 0xF).to(torch.long)
        hi = ((packed >> 4) & 0xF).to(torch.long)
        indices = torch.stack(
            [lo, hi],
            dim=-1,
        ).reshape(
            end - start,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        centroid_norm = torch.linalg.vector_norm(
            centroids[indices],
            dim=-1,
        )
        folded_norm = (
            soa_meta[
                start:end,
                :,
                SOA_K_NORM,
                :,
            ]
            .permute(0, 2, 1)
            .float()
        )
        raw_norm = (folded_norm * centroid_norm).to(torch.float16)
        # K norm occupies bytes 64:66, i.e. FP16 slot 32.
        aos_half[
            start:end,
            ...,
            MSE_BYTES // 2,
        ].copy_(raw_norm)
    return aos_cache


# ============================================================
# 13. Build all inputs
# ============================================================


def build_inputs(
    device: torch.device,
):
    q_rot = build_query_rot(device)
    kv_cache = build_synthetic_soa_kv_cache(device)
    block_table = build_block_table(device)
    seq_lens = build_seq_lens(device)
    centroids = build_centroids(device)
    pair_lut = build_pair_lut(centroids)
    return {
        "q_rot": q_rot,
        "kv_cache": kv_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "centroids": centroids,
        "pair_lut": pair_lut,
    }


# ============================================================
# 14. Print configuration
# ============================================================


def print_config():
    print("TurboQuant 4bit_nc Synthetic Benchmark")
    print()
    print("Preset              : turboquant_4bit_nc")
    print()
    print("Batch               :", BATCH_SIZE)
    print("Context             :", CONTEXT_LEN)
    print("Q heads             :", NUM_Q_HEADS)
    print("KV heads            :", NUM_KV_HEADS)
    print("GQA group           :", GQA_GROUP_SIZE)
    print("Head dim            :", HEAD_DIM)
    print()
    print("MSE bits            :", MSE_BITS)
    print("Value bits          :", VALUE_BITS)
    print("Norm correction     :", NORM_CORRECTION)
    print()
    print("K packed bytes      :", KEY_DATA_BYTES)
    print("V packed bytes      :", VAL_DATA_BYTES)
    print("Logical slot bytes  :", SLOT_SIZE)
    print("Aligned slot bytes  :", SLOT_SIZE_ALIGNED)
    print()
    print("KV block size       :", BLOCK_SIZE)
    print("Blocks / sequence   :", BLOCKS_PER_SEQ)
    print("Physical blocks     :", TOTAL_PHYSICAL_BLOCKS)
    print()
    print(
        "Data region / block :",
        DATA_REGION_BYTES,
        "bytes",
    )
    print(
        "Meta region / block :",
        META_REGION_BYTES,
        "bytes",
    )
    print(
        "Meta region offset  :",
        META_REGION_OFFSET,
        "bytes",
    )
    print(
        "Total bytes / block :",
        BLOCK_BYTES,
        "bytes",
    )
    print()
    print("KV splits           :", NUM_KV_SPLITS)


# ============================================================
# 15. Self-test
# ============================================================


def main():
    # --------------------------------------------------------
    # Static layout checks
    # --------------------------------------------------------
    assert NUM_Q_HEADS % NUM_KV_HEADS == 0
    assert GQA_GROUP_SIZE == 4
    assert MSE_BYTES == 64
    assert VAL_DATA_BYTES == 64
    assert DATA_BYTES_PER_SLOT == 128
    assert SLOT_SIZE == 134
    assert SLOT_SIZE_ALIGNED == 134
    assert DATA_REGION_BYTES == 16384
    assert META_REGION_BYTES == 768
    assert META_REGION_OFFSET == 16384
    assert BLOCK_BYTES == 17152
    assert DATA_REGION_BYTES + META_REGION_BYTES == BLOCK_BYTES
    assert BLOCKS_PER_SEQ == 256
    assert TOTAL_PHYSICAL_BLOCKS == 16384
    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------
    device = torch.device("cuda")
    print(
        "GPU                  :",
        torch.cuda.get_device_name(0),
    )
    print(
        "Compute Capability   :",
        torch.cuda.get_device_capability(0),
    )
    print()
    print_config()
    print()
    print("Building synthetic inputs...")
    inputs = build_inputs(device)
    torch.cuda.synchronize()
    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------
    print()
    print("Input Shapes")
    print(
        "q_rot       :",
        tuple(inputs["q_rot"].shape),
        inputs["q_rot"].dtype,
    )
    print(
        "kv_cache    :",
        tuple(inputs["kv_cache"].shape),
        inputs["kv_cache"].dtype,
    )
    print(
        "block_table :",
        tuple(inputs["block_table"].shape),
        inputs["block_table"].dtype,
    )
    print(
        "seq_lens    :",
        tuple(inputs["seq_lens"].shape),
        inputs["seq_lens"].dtype,
    )
    print(
        "centroids   :",
        tuple(inputs["centroids"].shape),
        inputs["centroids"].dtype,
    )
    print(
        "pair_lut    :",
        tuple(inputs["pair_lut"].shape),
        inputs["pair_lut"].dtype,
    )
    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------
    kv_bytes = inputs["kv_cache"].numel() * inputs["kv_cache"].element_size()
    print()
    print(
        "KV Cache bytes       :",
        kv_bytes,
    )
    print(
        "KV Cache MiB         :",
        f"{kv_bytes / (1024 ** 2):.2f}",
    )
    # --------------------------------------------------------
    # Finite checks
    # --------------------------------------------------------
    print()
    print("Finite Checks")
    print(
        "q_rot finite         :",
        bool(torch.isfinite(inputs["q_rot"]).all().item()),
    )
    print(
        "centroids finite     :",
        bool(torch.isfinite(inputs["centroids"]).all().item()),
    )
    print(
        "pair LUT finite      :",
        bool(torch.isfinite(inputs["pair_lut"]).all().item()),
    )
    # --------------------------------------------------------
    # block_table sanity
    # --------------------------------------------------------
    print()
    print("Block Table Sanity")
    print(
        "seq0 first block     :",
        inputs["block_table"][0, 0].item(),
    )
    print(
        "seq0 last block      :",
        inputs["block_table"][0, -1].item(),
    )
    print(
        "seq1 first block     :",
        inputs["block_table"][1, 0].item(),
    )
    print(
        "seq1 last block      :",
        inputs["block_table"][1, -1].item(),
    )
    print()
    print("Synthetic SoA input generation: OK")


if __name__ == "__main__":
    main()
