import argparse
import time

import torch

from tq4_common import (
    BATCH_SIZE,
    CONTEXT_LEN,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GQA_GROUP_SIZE,
    BLOCK_SIZE,
    NUM_KV_SPLITS,
    SLOT_SIZE_ALIGNED,
    TOTAL_PHYSICAL_BLOCKS,
    build_inputs,
)

from tq4_v2_stage1 import (
    launch_tq4_v2_stage1,
)


# ------------------------------------------------------------
# Default timing configuration
# ------------------------------------------------------------

DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "NCU profiling mode: "
            "5 warmup launches + 1 measured launch"
        ),
    )

    return parser.parse_args()


def print_header():
    print("TurboQuant 4bit_nc - SoA Triton V2 Stage1")
    print()

    print("GPU                  :", torch.cuda.get_device_name(0))
    print("Compute Capability   :", torch.cuda.get_device_capability(0))

    print()

    print("Batch                :", BATCH_SIZE)
    print("Context              :", CONTEXT_LEN)

    print("Q heads              :", NUM_Q_HEADS)
    print("KV heads             :", NUM_KV_HEADS)
    print("GQA group            :", GQA_GROUP_SIZE)
    print("Head dim             :", HEAD_DIM)

    print()

    print("KV block size        :", BLOCK_SIZE)
    print("KV splits            :", NUM_KV_SPLITS)
    print("Physical blocks      :", TOTAL_PHYSICAL_BLOCKS)

    print()

    print("K quant              : 4-bit Lloyd-Max")
    print("V quant              : 4-bit uniform")
    print("Norm correction      : ON")
    print("Layout               : SoA")

    print()

    print(
        "Stage1 grid          :",
        (
            BATCH_SIZE,
            NUM_KV_HEADS,
            NUM_KV_SPLITS,
        ),
    )

    num_programs = (
        BATCH_SIZE
        * NUM_KV_HEADS
        * NUM_KV_SPLITS
    )

    print("Stage1 programs      :", num_programs)

    print("BLOCK_M              :", 16)
    print("TILE_SIZE            :", 16)
    print("num_warps            :", 4)
    print("num_stages           :", 2)


def run_benchmark(
    warmup: int,
    iters: int,
):
    device = torch.device("cuda")

    # --------------------------------------------------------
    # Build synthetic inputs once.
    #
    # This is NOT included in timing.
    # --------------------------------------------------------

    print("Building synthetic inputs...")

    inputs = build_inputs(device)

    q_rot = inputs["q_rot"]
    kv_cache = inputs["kv_cache"]
    block_table = inputs["block_table"]
    seq_lens = inputs["seq_lens"]
    centroids = inputs["centroids"]
    pair_lut = inputs["pair_lut"]


    # --------------------------------------------------------
    # Preallocate Stage1 output.
    #
    # Critical:
    #
    # We do NOT allocate mid_o inside timed iterations.
    # --------------------------------------------------------

    mid_o = torch.empty(
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
        dtype=torch.float32,
        device=device,
    )


    # --------------------------------------------------------
    # First launch:
    #
    # Force Triton JIT compilation before warmup/timing.
    # --------------------------------------------------------

    print("Compiling / first launch...")

    launch_tq4_v2_stage1(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        pair_lut,
        mid_o=mid_o,
    )

    torch.cuda.synchronize()


    # --------------------------------------------------------
    # Sanity check before benchmark.
    # --------------------------------------------------------

    finite = bool(
        torch.isfinite(mid_o).all().item()
    )

    nan_count = int(
        torch.isnan(mid_o).sum().item()
    )

    inf_count = int(
        torch.isinf(mid_o).sum().item()
    )

    if not finite:
        raise RuntimeError(
            f"Stage1 output invalid: "
            f"NaN={nan_count}, Inf={inf_count}"
        )


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(f"Warmup               : {warmup}")

    for _ in range(warmup):

        launch_tq4_v2_stage1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o=mid_o,
        )

    torch.cuda.synchronize()


    # --------------------------------------------------------
    # CUDA Event timing
    #
    # Measure only GPU execution.
    #
    # Synthetic input construction, JIT compilation,
    # output allocation are all outside this region.
    # --------------------------------------------------------

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    start.record()

    for _ in range(iters):

        launch_tq4_v2_stage1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o=mid_o,
        )

    end.record()

    torch.cuda.synchronize()


    total_ms = start.elapsed_time(end)

    avg_ms = total_ms / iters

    avg_us = avg_ms * 1000.0


    # ========================================================
    # Derived workload metrics
    # ========================================================

    seconds = avg_ms / 1000.0


    # --------------------------------------------------------
    # Number of attention score evaluations:
    #
    # B * Hq * context
    #
    # = 64 * 32 * 4096
    # = 8,388,608
    #
    # Although Stage1 is grouped by KV head,
    # mathematically every Q head still evaluates one
    # attention score per context token.
    # --------------------------------------------------------

    score_evals = (
        BATCH_SIZE
        * NUM_Q_HEADS
        * CONTEXT_LEN
    )

    score_evals_per_s = (
        score_evals / seconds
    )


    # --------------------------------------------------------
    # Decode query throughput.
    #
    # One Stage1 launch processes B decode query tokens.
    # --------------------------------------------------------

    decode_tokens_per_s = (
        BATCH_SIZE / seconds
    )


    # --------------------------------------------------------
    # Logical compressed KV payload.
    #
    # With Grouped-Q, each:
    #
    #   (batch, kv_head, context token)
    #
    # owns one compressed K+V record.
    #
    # turboquant_4bit_nc logical storage:
    #
    #   134 B / token / KV head
    #
    # So:
    #
    # B * Hk * context * 134
    #
    # IMPORTANT:
    #
    # This is LOGICAL payload.
    #
    # It is NOT actual DRAM traffic.
    #
    # Actual memory transactions must be measured with NCU.
    # --------------------------------------------------------

    logical_kv_bytes = (
        BATCH_SIZE
        * NUM_KV_HEADS
        * CONTEXT_LEN
        * SLOT_SIZE_ALIGNED
    )

    logical_kv_gb = (
        logical_kv_bytes / 1.0e9
    )

    logical_kv_gib = (
        logical_kv_bytes / (1024.0 ** 3)
    )

    logical_kv_bw = (
        logical_kv_bytes
        / seconds
        / 1.0e9
    )


    # --------------------------------------------------------
    # Stage1 partial-output traffic.
    #
    # mid_o:
    #
    # [B, Hq, splits, D+1] FP32
    #
    # This is written once by Stage1.
    # --------------------------------------------------------

    mid_o_bytes = (
        mid_o.numel()
        * mid_o.element_size()
    )

    mid_o_mib = (
        mid_o_bytes / (1024.0 ** 2)
    )


    # --------------------------------------------------------
    # Useful structural values.
    # --------------------------------------------------------

    split_tokens = (
        CONTEXT_LEN // NUM_KV_SPLITS
    )

    tiles_per_program = (
        split_tokens // 16
    )


    # ========================================================
    # Print results
    # ========================================================

    print()

    print("Performance")
    print()

    print(
        f"Iterations            : {iters}"
    )

    print(
        f"Total timed GPU time  : {total_ms:.6f} ms"
    )

    print(
        f"Average Stage1 time   : {avg_ms:.6f} ms"
    )

    print(
        f"Average Stage1 time   : {avg_us:.3f} us"
    )

    print()

    print(
        f"Decode tokens/s       : "
        f"{decode_tokens_per_s:,.2f}"
    )

    print(
        f"Score evaluations/s   : "
        f"{score_evals_per_s / 1.0e9:.3f} G"
    )

    print()

    print(
        f"Logical KV payload    : "
        f"{logical_kv_bytes:,} bytes"
    )

    print(
        f"Logical KV payload    : "
        f"{logical_kv_gb:.3f} GB"
    )

    print(
        f"Logical KV payload    : "
        f"{logical_kv_gib:.3f} GiB"
    )

    print(
        f"Logical KV BW         : "
        f"{logical_kv_bw:.3f} GB/s"
    )

    print()

    print(
        f"Stage1 mid_o write    : "
        f"{mid_o_mib:.2f} MiB"
    )

    print()

    print(
        f"Tokens / split        : "
        f"{split_tokens}"
    )

    print(
        f"KV tiles / program    : "
        f"{tiles_per_program}"
    )

    print()

    print("Correctness sanity")
    print()

    print(
        "Finite output         :",
        finite,
    )

    print(
        "NaN count             :",
        nan_count,
    )

    print(
        "Inf count             :",
        inf_count,
    )

    print()

    print(
        "NOTE: Logical KV BW is not DRAM bandwidth."
    )

    print(
        "      Actual DRAM/L2/L1 traffic will be measured with Nsight Compute."
    )


def main():
    args = parse_args()

    if args.profile:
        warmup = 5
        iters = 1
    else:
        warmup = args.warmup
        iters = args.iters

    print_header()

    print()

    run_benchmark(
        warmup=warmup,
        iters=iters,
    )


if __name__ == "__main__":
    main()