import math

import torch
import triton
import triton.language as tl

from tq4_common import (
    BATCH_SIZE,
    CONTEXT_LEN,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GQA_GROUP_SIZE,
    BLOCK_SIZE,
    NUM_KV_SPLITS,
    MSE_BITS,
    VALUE_BITS,
    MSE_BYTES,
    VAL_DATA_BYTES,
    KEY_DATA_BYTES,
    META_REGION_OFFSET,
    NUM_SOA_FIELDS,
    SOA_K_NORM,
    SOA_V_SCALE,
    SOA_V_ZERO,
    N_CENTROIDS,
    SLOT_SIZE_ALIGNED,
    build_inputs,
)


# ============================================================
# TurboQuant 4bit_nc
# SoA Triton Decode V2 - Stage 1
#
# Standalone extraction from upstream vLLM:
#
#   reference/soa_decode_v2.py
#
# Benchmark scope:
#
#   q_rot
#      +
#   compressed SoA KV
#      ↓
#   Stage1
#      ↓
#   mid_o
#
# Not included:
#
#   Q rotation GEMM
#   pair-LUT construction
#   Stage2 split merge
# ============================================================


@triton.jit
def tq4_decode_stage1_v2(
    Q_rot_ptr,
    KV_cache_ptr,
    KV_cache_u16_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    Pair_lut_ptr,
    Mid_o_ptr,

    # Runtime strides
    stride_qb,
    stride_qh,

    stride_cache_block,

    stride_bt_b,

    stride_mid_b,
    stride_mid_h,
    stride_mid_s,

    # Dimensions
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,

    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,

    PADDED_SLOT: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,

    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,

    # TurboQuant
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,

    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,

    # SoA layout
    KEY_DATA_BYTES: tl.constexpr,
    META_REGION_OFFSET: tl.constexpr,

    NUM_SOA_FIELDS: tl.constexpr,

    SOA_K_NORM: tl.constexpr,
    SOA_V_SCALE: tl.constexpr,
    SOA_V_ZERO: tl.constexpr,

    N_CENTROIDS: tl.constexpr,

    # Attention
    ATTN_SCALE: tl.constexpr,

    # Tiles
    BLOCK_D: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,

    # These are kept to match upstream v2 semantics.
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    USE_PAIR_LUT: tl.constexpr = 0,
    USE_BF16_DOT: tl.constexpr = 0,
):
    # --------------------------------------------------------
    # Program mapping
    #
    # grid:
    #
    #   (B, Hk, NUM_KV_SPLITS)
    #
    # One Triton program:
    #
    #   one sequence
    #   one KV head
    #   one KV split
    #
    # It handles all Q heads that share this KV head.
    # --------------------------------------------------------

    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    sid = tl.program_id(2)


    # --------------------------------------------------------
    # Sequence partition
    # --------------------------------------------------------

    seq_len = tl.load(
        Seq_lens_ptr + bid
    )

    split_len = tl.cdiv(
        seq_len,
        NUM_KV_SPLITS,
    )

    split_start = split_len * sid

    split_end = tl.minimum(
        split_start + split_len,
        seq_len,
    )

    if split_start >= split_end:
        return


    # --------------------------------------------------------
    # Tile offsets
    # --------------------------------------------------------

    d_offs = tl.arange(
        0,
        BLOCK_D,
    )

    d_mask = d_offs < HEAD_DIM

    m_offs = tl.arange(
        0,
        BLOCK_M,
    )

    t_offs = tl.arange(
        0,
        TILE_SIZE,
    )


    # --------------------------------------------------------
    # Grouped-Q
    #
    # Qwen3-4B:
    #
    # KV_GROUP_SIZE = 4
    #
    # kv_head 0:
    #   q0 q1 q2 q3
    #
    # kv_head 1:
    #   q4 q5 q6 q7
    #
    # ...
    #
    # BLOCK_M = 16 for tensor-core tile.
    #
    # Only first 4 rows are valid for our GQA=4.
    # --------------------------------------------------------

    q_head_offs = (
        kv_hid * KV_GROUP_SIZE
        + m_offs
    )

    q_mask = (
        m_offs < KV_GROUP_SIZE
    )


    # --------------------------------------------------------
    # Load Q tile
    #
    # Shape:
    #
    #   [BLOCK_M, BLOCK_D]
    #
    # For us:
    #
    #   [16, 128]
    # --------------------------------------------------------

    q_addrs = (
        bid * stride_qb
        + q_head_offs[:, None] * stride_qh
        + d_offs[None, :]
    )

    Q = tl.load(
        Q_rot_ptr + q_addrs,
        mask=(
            q_mask[:, None]
            & d_mask[None, :]
        ),
        other=0.0,
    )


    # --------------------------------------------------------
    # Pre-warm the centroid table.
    #
    # 4bit:
    #
    # 16 centroids
    # 16 * FP32
    # = 64 bytes
    # --------------------------------------------------------

    if not KEY_FP8:
        _centroid_warm = tl.load(
            Centroids_ptr
            + tl.arange(
                0,
                N_CENTROIDS,
            )
        )


    # --------------------------------------------------------
    # Softmax uses exp2.
    #
    # exp(x)
    #
    # =
    #
    # exp2(x * log2(e))
    # --------------------------------------------------------

    RCP_LN2: tl.constexpr = (
        1.4426950408889634
    )

    LN2: tl.constexpr = (
        0.6931471805599453
    )

    QK_SCALE = (
        ATTN_SCALE
        * RCP_LN2
    )


    # --------------------------------------------------------
    # Online softmax state
    #
    # One state per Q row.
    # --------------------------------------------------------

    M = tl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=tl.float32,
    )

    L = tl.zeros(
        [BLOCK_M],
        dtype=tl.float32,
    )

    acc = tl.zeros(
        [BLOCK_M, BLOCK_D],
        dtype=tl.float32,
    )


    bt_base = (
        bid * stride_bt_b
    )


    # ========================================================
    # KV tile loop
    #
    # TILE_SIZE = 16
    #
    # Context = 4096
    # splits  = 32
    #
    # each split:
    #
    #   4096 / 32 = 128 tokens
    #
    # therefore:
    #
    #   128 / 16 = 8 tile iterations
    # ========================================================

    for start_n in range(
        split_start,
        split_end,
        TILE_SIZE,
    ):
        kv_offs = (
            start_n + t_offs
        )

        kv_mask_1d = (
            kv_offs < split_end
        )


        # ----------------------------------------------------
        # Paged KV mapping
        # ----------------------------------------------------

        page_idx = (
            kv_offs // BLOCK_SIZE
        )

        page_off = (
            kv_offs % BLOCK_SIZE
        )

        block_nums = tl.load(
            Block_table_ptr
            + bt_base
            + page_idx,
            mask=kv_mask_1d,
            other=0,
        ).to(tl.int64)


        # ----------------------------------------------------
        # SoA addressing
        # ----------------------------------------------------

        slot_within_block = (
            page_off.to(tl.int64)
        )

        block_base = (
            block_nums
            * stride_cache_block
        )


        DATA_BYTES_PER_SLOT: tl.constexpr = (
            KEY_DATA_BYTES
            + VAL_DATA_BYTES
        )


        # Packed data region:
        #
        # [token][kv_head][K64 | V64]

        data_bases = (
            block_base
            + slot_within_block
            * (
                NUM_KV_HEADS
                * DATA_BYTES_PER_SLOT
            )
            + tl.cast(
                kv_hid,
                tl.int64,
            )
            * DATA_BYTES_PER_SLOT
        )


        # Metadata region is accessed through uint16.
        #
        # [kv_head][field][token]

        head_meta_u16_base = (
            (
                block_base
                + META_REGION_OFFSET
            )
            // 2
            + tl.cast(
                kv_hid,
                tl.int64,
            )
            * (
                NUM_SOA_FIELDS
                * BLOCK_SIZE
            )
        )


        knorm_u16_addrs = (
            head_meta_u16_base
            + SOA_K_NORM
            * BLOCK_SIZE
            + slot_within_block
        )


        vscale_u16_addrs = (
            head_meta_u16_base
            + SOA_V_SCALE
            * BLOCK_SIZE
            + slot_within_block
        )


        vzero_u16_addrs = (
            head_meta_u16_base
            + SOA_V_ZERO
            * BLOCK_SIZE
            + slot_within_block
        )


        # ====================================================
        # K: packed INT4 index
        #
        # Each byte:
        #
        #   low nibble
        #   high nibble
        #
        # v2 loads each byte exactly once.
        # ====================================================

        if MSE_BITS == 4 and USE_PAIR_LUT:

            HALF_D: tl.constexpr = (
                BLOCK_D // 2
            )

            half_offs = tl.arange(
                0,
                HALF_D,
            )

            byte_mask = (
                half_offs * 2
                < HEAD_DIM
            )


            # [TILE_SIZE, 64]

            byte_addrs = (
                data_bases[:, None]
                + half_offs[None, :]
            )


            byte_raw = tl.load(
                KV_cache_ptr
                + byte_addrs,
                mask=(
                    kv_mask_1d[:, None]
                    & byte_mask[None, :]
                ),
                other=0,
            ).to(tl.int32)


            # One byte -> two centroid indices

            lo_idx = (
                byte_raw & 0xF
            )

            hi_idx = (
                byte_raw >> 4
            ) & 0xF


            # pair LUT index:
            #
            # idx = lo * 16 + hi

            pair_key = (
                lo_idx
                * N_CENTROIDS
                + hi_idx
            )


            pair_slot = tl.arange(
                0,
                2,
            )


            # Result:
            #
            # [TILE_SIZE, 64, 2]

            c_pair = tl.load(
                Pair_lut_ptr
                + pair_key[:, :, None] * 2
                + pair_slot[
                    None,
                    None,
                    :
                ],
                mask=(
                    kv_mask_1d[
                        :,
                        None,
                        None,
                    ]
                    & byte_mask[
                        None,
                        :,
                        None,
                    ]
                ),
                other=0.0,
            )


            # [TILE_SIZE, 64, 2]
            #
            # ->
            #
            # [TILE_SIZE, 128]

            c_vals = tl.reshape(
                c_pair,
                [
                    TILE_SIZE,
                    BLOCK_D,
                ],
            )

        else:
            # This standalone baseline only targets:
            #
            # turboquant_4bit_nc
            #
            # therefore this branch should never execute.

            half_idx = (
                d_offs // 2
            )

            nibble_shift = (
                d_offs % 2
            ) * 4

            mse_addrs = (
                data_bases[:, None]
                + half_idx[None, :]
            )

            mse_raw = tl.load(
                KV_cache_ptr
                + mse_addrs,
                mask=(
                    kv_mask_1d[:, None]
                    & d_mask[None, :]
                ),
                other=0,
            ).to(tl.int32)

            mse_idx = (
                mse_raw
                >> nibble_shift[None, :]
            ) & 0xF

            c_vals = tl.load(
                Centroids_ptr
                + mse_idx,
                mask=(
                    kv_mask_1d[:, None]
                    & d_mask[None, :]
                ),
                other=0.0,
            )


        # ----------------------------------------------------
        # K norm
        #
        # Norm correction has already been folded into the
        # stored scalar by the SoA store path.
        #
        # Decode side therefore only needs:
        #
        # K = centroid * stored_norm
        # ----------------------------------------------------

        norm_u16 = tl.load(
            KV_cache_u16_ptr
            + knorm_u16_addrs,
            mask=kv_mask_1d,
            other=0,
        )


        vec_norms = (
            norm_u16
            .to(
                tl.float16,
                bitcast=True,
            )
            .to(
                tl.float32
            )
        )


        K_recon = (
            c_vals
            * vec_norms[:, None]
        )


        # [TILE_SIZE, D]
        #
        # ->
        #
        # [D, TILE_SIZE]

        K_T = tl.trans(
            K_recon
        )


        # ====================================================
        # QK
        #
        # [16,128]
        #
        # x
        #
        # [128,16]
        #
        # ->
        #
        # [16,16]
        #
        # CUDA path uses FP16 Tensor Core input.
        # FP32 accumulation remains inside tl.dot.
        # ====================================================

        S = (
            QK_SCALE
            * tl.dot(
                Q.to(tl.float16),
                K_T.to(tl.float16),
            )
        )


        S = tl.where(
            kv_mask_1d[None, :],
            S,
            float("-inf"),
        )


        # ====================================================
        # Online softmax
        # ====================================================

        m_j = tl.maximum(
            M,
            tl.max(
                S,
                axis=1,
            ),
        )


        m_j = tl.where(
            m_j > float("-inf"),
            m_j,
            0.0,
        )


        P = tl.math.exp2(
            S
            - m_j[:, None]
        )


        l_j = tl.sum(
            P,
            axis=1,
        )


        alpha = tl.math.exp2(
            M - m_j
        )


        acc = (
            acc
            * alpha[:, None]
        )


        L = (
            L * alpha
            + l_j
        )


        M = m_j


        # ====================================================
        # V INT4
        #
        # V data starts immediately after packed K data.
        # ====================================================

        val_bases = (
            data_bases
            + KEY_DATA_BYTES
        )


        # Build V directly in its final [TILE_SIZE, BLOCK_D]
        # tensor-core layout. Feeding tl.interleave output straight into
        # tl.dot can preserve an incompatible blocked layout on CUDA and
        # silently permute V columns.

        v_byte_idx = d_offs // 2
        v_nibble_shift = (d_offs % 2) * 4

        val_addrs = (
            val_bases[:, None]
            + v_byte_idx[None, :]
        )


        val_byte = tl.load(
            KV_cache_ptr
            + val_addrs,
            mask=(
                kv_mask_1d[:, None]
                & d_mask[None, :]
            ),
            other=0,
        ).to(tl.int32)


        v_idx = (
            (
                val_byte
                >> v_nibble_shift[None, :]
            )
            & 0xF
        ).to(tl.float32)


        # ----------------------------------------------------
        # V metadata
        # ----------------------------------------------------

        scale_u16 = tl.load(
            KV_cache_u16_ptr
            + vscale_u16_addrs,
            mask=kv_mask_1d,
            other=0,
        )


        zero_u16 = tl.load(
            KV_cache_u16_ptr
            + vzero_u16_addrs,
            mask=kv_mask_1d,
            other=0,
        )


        v_scales = (
            scale_u16
            .to(
                tl.float16,
                bitcast=True,
            )
            .to(
                tl.float32
            )
        )


        v_zeros = (
            zero_u16
            .to(
                tl.float16,
                bitcast=True,
            )
            .to(
                tl.float32
            )
        )


        V = (
            v_idx
            * v_scales[:, None]
            + v_zeros[:, None]
        )


        # ====================================================
        # P * V
        #
        # [16,16]
        #
        # x
        #
        # [16,128]
        #
        # ->
        #
        # [16,128]
        #
        # Tensor Core
        # ====================================================

        acc += tl.dot(
            P.to(tl.float16),
            V.to(tl.float16),
        )


    # ========================================================
    # Stage1 epilogue
    # ========================================================

    safe_L = tl.where(
        L > 0.0,
        L,
        1.0,
    )


    acc = (
        acc
        / safe_L[:, None]
    )


    # Stage2 expects natural-log LSE.

    lse = (
        M * LN2
        + tl.log(safe_L)
    )


    # --------------------------------------------------------
    # Write:
    #
    # mid_o[B, Hq, split, D+1]
    # --------------------------------------------------------

    out_addrs = (
        bid * stride_mid_b
        + q_head_offs[:, None]
        * stride_mid_h
        + sid * stride_mid_s
        + d_offs[None, :]
    )


    tl.store(
        Mid_o_ptr
        + out_addrs,
        acc,
        mask=(
            q_mask[:, None]
            & d_mask[None, :]
        ),
    )


    # Last element stores LSE.

    lse_addrs = (
        bid * stride_mid_b
        + q_head_offs
        * stride_mid_h
        + sid
        * stride_mid_s
        + HEAD_DIM
    )


    tl.store(
        Mid_o_ptr
        + lse_addrs,
        lse,
        mask=q_mask,
    )


# ============================================================
# Standalone launcher
# ============================================================

def launch_tq4_v2_stage1(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    pair_lut: torch.Tensor,
    mid_o: torch.Tensor | None = None,
):
    """
    Standalone Stage1 launcher.

    This deliberately excludes:

        query @ PiT
        pair-LUT construction
        Stage2

    so the measured latency is the actual TQ decode Stage1.
    """

    B, Hq, D = q_rot.shape

    Hk = kv_cache.shape[2]

    block_size = kv_cache.shape[1]

    padded_slot = kv_cache.shape[3]

    max_num_blocks = (
        block_table.shape[1]
    )


    # --------------------------------------------------------
    # Fixed workload sanity checks
    # --------------------------------------------------------

    assert B == BATCH_SIZE

    assert Hq == NUM_Q_HEADS

    assert Hk == NUM_KV_HEADS

    assert D == HEAD_DIM

    assert block_size == BLOCK_SIZE

    assert padded_slot == SLOT_SIZE_ALIGNED

    assert (
        Hq // Hk
        == GQA_GROUP_SIZE
    )

    assert MSE_BITS == 4

    assert VALUE_BITS == 4

    assert centroids.numel() == 16

    assert pair_lut.shape == (
        256,
        2,
    )


    # --------------------------------------------------------
    # v2 upstream constants
    # --------------------------------------------------------

    BLOCK_D = triton.next_power_of_2(
        D
    )


    # Upstream:
    #
    # BLOCK_M =
    # max(16, next_power_of_2(kv_group_size))
    #
    # GQA=4 -> BLOCK_M=16.

    BLOCK_M = max(
        16,
        triton.next_power_of_2(
            GQA_GROUP_SIZE
        ),
    )


    TILE_SIZE = 16


    ATTN_SCALE = (
        1.0
        / math.sqrt(D)
    )


    # --------------------------------------------------------
    # Stage1 output
    #
    # [B,Hq,splits,D+1]
    # --------------------------------------------------------

    if mid_o is None:

        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=q_rot.device,
        )


    # Same physical storage reinterpreted as uint16
    # for contiguous FP16 SoA metadata loads.

    kv_cache_u16 = (
        kv_cache.view(
            torch.uint16
        )
    )


    # --------------------------------------------------------
    # Grid
    #
    # B=64
    # Hk=8
    # splits=32
    #
    # = 16384 Triton programs
    # --------------------------------------------------------

    grid = (
        B,
        Hk,
        NUM_KV_SPLITS,
    )


    tq4_decode_stage1_v2[
        grid
    ](
        q_rot,
        kv_cache,
        kv_cache_u16,

        block_table,
        seq_lens,

        centroids,
        pair_lut,

        mid_o,

        q_rot.stride(0),
        q_rot.stride(1),

        kv_cache.stride(0),

        block_table.stride(0),

        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),

        NUM_Q_HEADS=Hq,
        NUM_KV_HEADS=Hk,

        HEAD_DIM=D,

        BLOCK_SIZE=block_size,

        PADDED_SLOT=padded_slot,

        MAX_NUM_BLOCKS=max_num_blocks,

        NUM_KV_SPLITS=NUM_KV_SPLITS,

        KV_GROUP_SIZE=GQA_GROUP_SIZE,

        MSE_BITS=MSE_BITS,

        MSE_BYTES=MSE_BYTES,

        VQB=VALUE_BITS,

        VAL_DATA_BYTES=VAL_DATA_BYTES,

        KEY_DATA_BYTES=KEY_DATA_BYTES,

        META_REGION_OFFSET=META_REGION_OFFSET,

        NUM_SOA_FIELDS=NUM_SOA_FIELDS,

        SOA_K_NORM=SOA_K_NORM,

        SOA_V_SCALE=SOA_V_SCALE,

        SOA_V_ZERO=SOA_V_ZERO,

        N_CENTROIDS=N_CENTROIDS,

        ATTN_SCALE=ATTN_SCALE,

        BLOCK_D=BLOCK_D,

        TILE_SIZE=TILE_SIZE,

        BLOCK_M=BLOCK_M,

        # turboquant_4bit_nc fixed path
        KEY_FP8=0,

        NORM_CORRECTION=1,

        FP8_E4B15=0,

        USE_PAIR_LUT=1,

        # CUDA/NVIDIA path:
        # upstream v2 uses fp16 tl.dot.
        USE_BF16_DOT=0,

        # Same upstream CUDA launch config.
        num_warps=4,

        num_stages=2,
    )


    return mid_o


# ============================================================
# Simple launch test
# ============================================================

def main():

    device = torch.device("cuda")


    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Building synthetic inputs..."
    )


    inputs = build_inputs(
        device
    )


    print(
        "Launching standalone "
        "TurboQuant V2 Stage1..."
    )


    mid_o = launch_tq4_v2_stage1(
        inputs["q_rot"],
        inputs["kv_cache"],
        inputs["block_table"],
        inputs["seq_lens"],
        inputs["centroids"],
        inputs["pair_lut"],
    )


    # Force Triton compilation + kernel completion.
    torch.cuda.synchronize()


    print()

    print(
        "mid_o shape :",
        tuple(mid_o.shape),
    )

    print(
        "mid_o dtype :",
        mid_o.dtype,
    )


    finite = torch.isfinite(
        mid_o
    ).all().item()


    nan_count = torch.isnan(
        mid_o
    ).sum().item()


    inf_count = torch.isinf(
        mid_o
    ).sum().item()


    print(
        "finite      :",
        bool(finite),
    )

    print(
        "NaN count   :",
        nan_count,
    )

    print(
        "Inf count   :",
        inf_count,
    )


    print()

    print(
        "Grid        :",
        (
            BATCH_SIZE,
            NUM_KV_HEADS,
            NUM_KV_SPLITS,
        ),
    )

    print(
        "Programs    :",
        BATCH_SIZE
        * NUM_KV_HEADS
        * NUM_KV_SPLITS,
    )

    print(
        "BLOCK_M     :",
        16,
    )

    print(
        "TILE_SIZE   :",
        16,
    )


    if not finite:
        raise RuntimeError(
            "Stage1 output contains NaN/Inf"
        )


    print()

    print(
        "Standalone TQ4 "
        "SoA Triton V2 Stage1: OK"
    )


if __name__ == "__main__":
    main()
