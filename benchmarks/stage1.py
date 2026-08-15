import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from baseline.common import (
    BATCH_SIZE,
    HEAD_DIM,
    NUM_KV_SPLITS,
    NUM_Q_HEADS,
    build_inputs,
    convert_soa_to_aos_kv_cache,
)
from baseline.triton_v1 import (
    launch_aos_v1_stage1,
    launch_soa_v1_stage1,
)
from baseline.triton_v2 import launch_tq4_v2_stage1


CUDA_ARCH_LIST = "8.9"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = PROJECT_ROOT / "cuda"


def event_time_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def diff_stats(ref, test):
    diff = (test - ref).abs()
    return float(diff.max()), float(diff.mean())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--include-cuda-v2", action="store_true")
    parser.add_argument("--include-cuda-v3", action="store_true")
    parser.add_argument("--include-cuda-v4", action="store_true")
    parser.add_argument("--include-cuda-v5", action="store_true")
    parser.add_argument("--include-cuda-v6", action="store_true")
    parser.add_argument("--include-cuda-v7", action="store_true")
    parser.add_argument("--include-cuda-v8", action="store_true")
    return parser.parse_args()


def build_cuda_v2_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v2"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v2_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v2_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v2.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v3_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v3"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v3_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v3_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v3.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v4_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v4"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v4_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v4_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v4.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v5_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v5"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v5_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v5_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v5.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v6_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v6"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v6_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v6_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v6.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v7_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v7"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v7_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v7_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v7.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def build_cuda_v8_extension():
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    build_dir = CUDA_DIR / "build_v8"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="tq4_cuda_v8_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v8_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v8.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=True,
    )


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    print("GPU:", torch.cuda.get_device_name(0))
    print("Building shared logical inputs...")
    inputs = build_inputs(device)
    q_rot = inputs["q_rot"]
    soa_cache = inputs["kv_cache"]
    block_table = inputs["block_table"]
    seq_lens = inputs["seq_lens"]
    centroids = inputs["centroids"]
    pair_lut = inputs["pair_lut"]
    aos_cache = convert_soa_to_aos_kv_cache(
        soa_cache,
        centroids,
    )
    shape = (
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
    )
    mid_aos = torch.empty(shape, dtype=torch.float32, device=device)
    mid_soa_v1 = torch.empty_like(mid_aos)
    mid_soa_v2 = torch.empty_like(mid_aos)
    mid_cuda = torch.empty_like(mid_aos)
    mid_cuda_v2 = torch.empty_like(mid_aos)
    mid_cuda_v3 = torch.empty_like(mid_aos)
    mid_cuda_v4 = torch.empty_like(mid_aos)
    mid_cuda_v5 = torch.empty_like(mid_aos)
    mid_cuda_v6 = torch.empty_like(mid_aos)
    mid_cuda_v7 = torch.empty_like(mid_aos)
    mid_cuda_v8 = torch.empty_like(mid_aos)
    cuda_ext = None
    if args.include_cuda:
        from tools.cuda_v1_diagnostic import build_extension

        cuda_ext = build_extension()
    cuda_v2_ext = build_cuda_v2_extension() if args.include_cuda_v2 else None
    cuda_v3_ext = build_cuda_v3_extension() if args.include_cuda_v3 else None
    cuda_v4_ext = build_cuda_v4_extension() if args.include_cuda_v4 else None
    cuda_v5_ext = build_cuda_v5_extension() if args.include_cuda_v5 else None
    cuda_v6_ext = build_cuda_v6_extension() if args.include_cuda_v6 else None
    cuda_v7_ext = build_cuda_v7_extension() if args.include_cuda_v7 else None
    cuda_v8_ext = build_cuda_v8_extension() if args.include_cuda_v8 else None

    def run_aos():
        launch_aos_v1_stage1(
            q_rot,
            aos_cache,
            block_table,
            seq_lens,
            centroids,
            mid_aos,
        )

    def run_soa_v1():
        launch_soa_v1_stage1(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_soa_v1,
        )

    def run_soa_v2():
        launch_tq4_v2_stage1(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o=mid_soa_v2,
        )

    def run_cuda():
        assert cuda_ext is not None
        cuda_ext.tq4_cuda_v1(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda,
        )

    def run_cuda_v2():
        assert cuda_v2_ext is not None
        cuda_v2_ext.tq4_cuda_v2(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v2,
        )

    def run_cuda_v3():
        assert cuda_v3_ext is not None
        cuda_v3_ext.tq4_cuda_v3(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v3,
        )

    def run_cuda_v4():
        assert cuda_v4_ext is not None
        cuda_v4_ext.tq4_cuda_v4(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v4,
        )

    def run_cuda_v5():
        assert cuda_v5_ext is not None
        cuda_v5_ext.tq4_cuda_v5(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v5,
        )

    def run_cuda_v6():
        assert cuda_v6_ext is not None
        cuda_v6_ext.tq4_cuda_v6(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v6,
        )

    def run_cuda_v7():
        assert cuda_v7_ext is not None
        cuda_v7_ext.tq4_cuda_v7_stage1(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v7,
        )

    def run_cuda_v8():
        assert cuda_v8_ext is not None
        cuda_v8_ext.tq4_cuda_v8(
            q_rot,
            soa_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda_v8,
        )

    print("Compiling and checking outputs...")
    run_aos()
    run_soa_v1()
    run_soa_v2()
    if args.include_cuda:
        run_cuda()
    if args.include_cuda_v2:
        run_cuda_v2()
    if args.include_cuda_v3:
        run_cuda_v3()
    if args.include_cuda_v4:
        run_cuda_v4()
    if args.include_cuda_v5:
        run_cuda_v5()
    if args.include_cuda_v6:
        run_cuda_v6()
    if args.include_cuda_v7:
        run_cuda_v7()
    if args.include_cuda_v8:
        run_cuda_v8()
    torch.cuda.synchronize()
    outputs = [
        ("AoS Triton V1", mid_aos),
        ("SoA Triton V1", mid_soa_v1),
        ("SoA Triton V2-fixed", mid_soa_v2),
    ]
    if args.include_cuda:
        outputs.append(("CUDA V1", mid_cuda))
    if args.include_cuda_v2:
        outputs.append(("CUDA V2", mid_cuda_v2))
    if args.include_cuda_v3:
        outputs.append(("CUDA V3", mid_cuda_v3))
    if args.include_cuda_v4:
        outputs.append(("CUDA V4", mid_cuda_v4))
    if args.include_cuda_v5:
        outputs.append(("CUDA V5", mid_cuda_v5))
    if args.include_cuda_v6:
        outputs.append(("CUDA V6", mid_cuda_v6))
    if args.include_cuda_v7:
        outputs.append(("CUDA V7", mid_cuda_v7))
    if args.include_cuda_v8:
        outputs.append(("CUDA V8", mid_cuda_v8))
    for name, output in outputs:
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"{name} produced NaN/Inf")
    print()
    print("Correctness against SoA V1")
    comparisons = [
        ("AoS Triton V1", mid_aos),
        ("SoA Triton V2-fixed", mid_soa_v2),
    ]
    if args.include_cuda:
        comparisons.append(("CUDA V1", mid_cuda))
    if args.include_cuda_v2:
        comparisons.append(("CUDA V2", mid_cuda_v2))
    if args.include_cuda_v3:
        comparisons.append(("CUDA V3", mid_cuda_v3))
    if args.include_cuda_v4:
        comparisons.append(("CUDA V4", mid_cuda_v4))
    if args.include_cuda_v5:
        comparisons.append(("CUDA V5", mid_cuda_v5))
    if args.include_cuda_v6:
        comparisons.append(("CUDA V6", mid_cuda_v6))
    if args.include_cuda_v7:
        comparisons.append(("CUDA V7", mid_cuda_v7))
    if args.include_cuda_v8:
        comparisons.append(("CUDA V8", mid_cuda_v8))
    for name, output in comparisons:
        out_max, out_mean = diff_stats(
            mid_soa_v1[..., :HEAD_DIM],
            output[..., :HEAD_DIM],
        )
        lse_max, lse_mean = diff_stats(
            mid_soa_v1[..., HEAD_DIM],
            output[..., HEAD_DIM],
        )
        print(f"{name:21s} output max/mean: " f"{out_max:.8g} / {out_mean:.8g}")
        print(f"{'':21s} LSE    max/mean: " f"{lse_max:.8g} / {lse_mean:.8g}")
    print()
    print("Stage1 performance")
    runners = [
        ("AoS Triton V1", run_aos),
        ("SoA Triton V1", run_soa_v1),
        ("SoA Triton V2-fixed", run_soa_v2),
    ]
    if args.include_cuda:
        runners.append(("CUDA V1", run_cuda))
    if args.include_cuda_v2:
        runners.append(("CUDA V2", run_cuda_v2))
    if args.include_cuda_v3:
        runners.append(("CUDA V3", run_cuda_v3))
    if args.include_cuda_v4:
        runners.append(("CUDA V4", run_cuda_v4))
    if args.include_cuda_v5:
        runners.append(("CUDA V5", run_cuda_v5))
    if args.include_cuda_v6:
        runners.append(("CUDA V6", run_cuda_v6))
    if args.include_cuda_v7:
        runners.append(("CUDA V7", run_cuda_v7))
    if args.include_cuda_v8:
        runners.append(("CUDA V8", run_cuda_v8))
    samples = {name: [] for name, _ in runners}
    for round_idx in range(args.rounds):
        order = runners[round_idx % len(runners) :] + runners[: round_idx % len(runners)]
        values = {}
        for name, fn in order:
            values[name] = event_time_ms(
                fn,
                warmup=args.warmup,
                iters=args.iters,
            )
            samples[name].append(values[name])
        print(
            f"round {round_idx + 1}: "
            + ", ".join(f"{name}={values[name]:.6f} ms" for name, _ in runners)
        )
    print()
    print("Median")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    for name, _ in runners:
        print(f"{name:21s}: {medians[name]:.6f} ms")
    print()
    print("AoS -> SoA V1 speedup : " f"{medians['AoS Triton V1'] / medians['SoA Triton V1']:.3f}x")
    print(
        "SoA V1 -> V2 speedup  : "
        f"{medians['SoA Triton V1'] / medians['SoA Triton V2-fixed']:.3f}x"
    )
    print(
        "AoS V1 -> V2 speedup  : "
        f"{medians['AoS Triton V1'] / medians['SoA Triton V2-fixed']:.3f}x"
    )
    if args.include_cuda:
        print(
            "CUDA V1 / V2-fixed     : "
            f"{medians['CUDA V1'] / medians['SoA Triton V2-fixed']:.3f}x slower"
        )
    if args.include_cuda_v2:
        print(
            "CUDA V2 / V2-fixed     : "
            f"{medians['CUDA V2'] / medians['SoA Triton V2-fixed']:.3f}x slower"
        )
    if args.include_cuda_v3:
        v3_ratio = medians["CUDA V3"] / medians["SoA Triton V2-fixed"]
        if v3_ratio >= 1.0:
            print(f"CUDA V3 / V2-fixed     : {v3_ratio:.3f}x slower")
        else:
            print(f"CUDA V3 vs V2-fixed    : {1.0 / v3_ratio:.3f}x faster")
    if args.include_cuda_v4:
        v4_ratio = medians["CUDA V4"] / medians["SoA Triton V2-fixed"]
        if v4_ratio >= 1.0:
            print(f"CUDA V4 / V2-fixed     : {v4_ratio:.3f}x slower")
        else:
            print(f"CUDA V4 vs V2-fixed    : {1.0 / v4_ratio:.3f}x faster")
    if args.include_cuda_v5:
        v5_ratio = medians["CUDA V5"] / medians["SoA Triton V2-fixed"]
        if v5_ratio >= 1.0:
            print(f"CUDA V5 / V2-fixed     : {v5_ratio:.3f}x slower")
        else:
            print(f"CUDA V5 vs V2-fixed    : {1.0 / v5_ratio:.3f}x faster")
    if args.include_cuda_v6:
        v6_ratio = medians["CUDA V6"] / medians["SoA Triton V2-fixed"]
        if v6_ratio >= 1.0:
            print(f"CUDA V6 / V2-fixed     : {v6_ratio:.3f}x slower")
        else:
            print(f"CUDA V6 vs V2-fixed    : {1.0 / v6_ratio:.3f}x faster")
    if args.include_cuda_v7:
        v7_ratio = medians["CUDA V7"] / medians["SoA Triton V2-fixed"]
        if v7_ratio >= 1.0:
            print(f"CUDA V7 / V2-fixed     : {v7_ratio:.3f}x slower")
        else:
            print(f"CUDA V7 vs V2-fixed    : {1.0 / v7_ratio:.3f}x faster")
    if args.include_cuda_v8:
        v8_ratio = medians["CUDA V8"] / medians["SoA Triton V2-fixed"]
        if v8_ratio >= 1.0:
            print(f"CUDA V8 / V2-fixed     : {v8_ratio:.3f}x slower")
        else:
            print(f"CUDA V8 vs V2-fixed    : {1.0 / v8_ratio:.3f}x faster")
        if args.include_cuda_v7:
            print(f"CUDA V8 vs CUDA V7     : {medians['CUDA V7'] / medians['CUDA V8']:.3f}x")


if __name__ == "__main__":
    main()
