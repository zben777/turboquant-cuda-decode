import argparse
import statistics

import torch

from .stage1 import (
    build_cuda_v7_extension,
    diff_stats,
    event_time_ms,
)
from baseline.common import (
    BATCH_SIZE,
    HEAD_DIM,
    NUM_KV_SPLITS,
    NUM_Q_HEADS,
    build_inputs,
)
from baseline.stage2 import launch_stage2
from baseline.triton_v2 import launch_tq4_v2_stage1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Building fixed decode inputs...")
    inputs = build_inputs(device)
    q_rot = inputs["q_rot"]
    kv_cache = inputs["kv_cache"]
    block_table = inputs["block_table"]
    seq_lens = inputs["seq_lens"]
    centroids = inputs["centroids"]
    pair_lut = inputs["pair_lut"]
    mid_shape = (
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
    )
    out_shape = (BATCH_SIZE, NUM_Q_HEADS, HEAD_DIM)
    lse_shape = (BATCH_SIZE, NUM_Q_HEADS)
    mid_triton = torch.empty(mid_shape, dtype=torch.float32, device=device)
    mid_cuda = torch.empty_like(mid_triton)
    out_triton = torch.empty(out_shape, dtype=torch.float32, device=device)
    out_cuda = torch.empty_like(out_triton)
    out_cuda_stage2_ref = torch.empty_like(out_triton)
    lse_triton = torch.empty(lse_shape, dtype=torch.float32, device=device)
    lse_cuda = torch.empty_like(lse_triton)
    lse_cuda_stage2_ref = torch.empty_like(lse_triton)
    cuda_ext = build_cuda_v7_extension()

    def triton_stage1():
        launch_tq4_v2_stage1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o=mid_triton,
        )

    def triton_stage2():
        launch_stage2(mid_triton, seq_lens, out_triton, lse_triton)

    def triton_full():
        triton_stage1()
        triton_stage2()

    def cuda_stage1():
        cuda_ext.tq4_cuda_v7_stage1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda,
        )

    def cuda_stage2():
        cuda_ext.tq4_cuda_v7_stage2(mid_cuda, out_cuda, lse_cuda)

    def cuda_full():
        cuda_ext.tq4_cuda_v7_full(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda,
            out_cuda,
            lse_cuda,
        )

    print("Compiling and checking complete outputs...")
    triton_full()
    cuda_full()
    cuda_ext.tq4_cuda_v7_stage2(
        mid_triton,
        out_cuda_stage2_ref,
        lse_cuda_stage2_ref,
    )
    torch.cuda.synchronize()
    for name, tensor in (
        ("Triton output", out_triton),
        ("Triton LSE", lse_triton),
        ("CUDA output", out_cuda),
        ("CUDA LSE", lse_cuda),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} produced NaN/Inf")
    print()
    print("Correctness")
    out_max, out_mean = diff_stats(out_triton, out_cuda_stage2_ref)
    lse_max, lse_mean = diff_stats(lse_triton, lse_cuda_stage2_ref)
    print("CUDA Stage2 vs Triton Stage2 output max/mean: " f"{out_max:.8g} / {out_mean:.8g}")
    print("CUDA Stage2 vs Triton Stage2 LSE    max/mean: " f"{lse_max:.8g} / {lse_mean:.8g}")
    out_max, out_mean = diff_stats(out_triton, out_cuda)
    lse_max, lse_mean = diff_stats(lse_triton, lse_cuda)
    print("CUDA V7 full vs Triton full output max/mean: " f"{out_max:.8g} / {out_mean:.8g}")
    print("CUDA V7 full vs Triton full LSE    max/mean: " f"{lse_max:.8g} / {lse_mean:.8g}")
    runners = [
        ("Triton Stage1", triton_stage1),
        ("Triton Stage2", triton_stage2),
        ("Triton full", triton_full),
        ("CUDA V7 Stage1", cuda_stage1),
        ("CUDA V7 Stage2", cuda_stage2),
        ("CUDA V7 full", cuda_full),
    ]
    samples = {name: [] for name, _ in runners}
    print()
    print("Decode performance")
    for round_idx in range(args.rounds):
        order = runners[round_idx % len(runners) :] + runners[: round_idx % len(runners)]
        values = {}
        for name, fn in order:
            value = event_time_ms(
                fn,
                warmup=args.warmup,
                iters=args.iters,
            )
            values[name] = value
            samples[name].append(value)
        print(
            f"round {round_idx + 1}: "
            + ", ".join(f"{name}={values[name]:.6f} ms" for name, _ in runners)
        )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print()
    print("Median")
    for name, _ in runners:
        print(f"{name:18s}: {medians[name]:.6f} ms")
    print()
    print(
        "CUDA V7 full vs Triton full: "
        f"{medians['Triton full'] / medians['CUDA V7 full']:.3f}x faster"
    )
    print(
        "CUDA Stage2 share          : "
        f"{100.0 * medians['CUDA V7 Stage2'] / medians['CUDA V7 full']:.2f}%"
    )


if __name__ == "__main__":
    main()
