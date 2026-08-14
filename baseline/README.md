# Standalone Baseline And Validation Harness

[中文版](README_CN.md)

This directory is the main execution entry point for the repository. It
constructs the fixed Qwen3-4B-shaped workload, launches the Triton baselines
and CUDA candidates, checks numerical correctness, and records latency.

The naming convention is:

```text
bench_*.py    executable benchmark or correctness entry point
tq4_*.py      reusable layout, Triton kernel, Store, or Stage2 module
profile_*.py  single-launch profiling entry point
```

## Quick Start

Run commands from the repository root:

```bash
./baseline/run.sh smoke
```

The first CUDA run JIT-compiles a PyTorch extension and can take several
minutes. Later runs reuse `cuda/build_v*/` until `./baseline/run.sh clean` is
called. Those generated directories are ignored by Git.

Available modes:

| Mode | Purpose | Typical cost |
| --- | --- | --- |
| `check` | Parse all Python sources | CPU-only, seconds |
| `layout` | Validate the synthetic SoA layout and tensors | Short GPU check |
| `smoke` | Short Triton/CUDA V7 Stage1 and Full regression | One V7 build |
| `store` | Run raw Q/K/V through the real SoA Store and both decoders | Correctness-focused |
| `benchmark` | Run official five-round Triton and CUDA V1-V7 timings | Builds all CUDA versions |
| `all` | Run the complete release regression | Longest |
| `clean` | Remove generated CUDA extension builds | No benchmark |

Select a Python environment or CUDA toolkit explicitly when needed:

```bash
PYTHON=/path/to/python CUDA_HOME=/path/to/cuda ./baseline/run.sh smoke
```

## Executable Entry Points

| File | Role |
| --- | --- |
| `bench_triton_baselines.py` | Authoritative Stage1 comparison: AoS V1, SoA V1, V2-fixed, and optional CUDA V1-V7 |
| `bench_full_decode.py` | Separately measures Stage1, Stage2, and Stage1+Stage2 Full Decode for Triton V2-fixed and CUDA V7 |
| `bench_store_decode.py` | Starts from raw FP16 Q/K/V, invokes the unmodified vLLM SoA Store, and compares Triton/CUDA final outputs with FP32 attention |
| `bench_v2_stage1.py` | Focused standalone timing/profiling entry for the corrected Triton V2 Stage1 |
| `bench_cuda_v1.py` | Historical CUDA V1 diagnostic, including split-level comparisons with FP32 and a Tensor Core mimic |
| `profile_cuda_v1.py` | Minimal warmed CUDA V1 launch intended for an external profiler such as NCU |

## Reusable Modules

| File | Role |
| --- | --- |
| `tq4_common.py` | Fixed constants, the 134-byte SoA slot contract, synthetic cache construction, block tables, centroids, and pair LUT |
| `tq4_v1_stage1.py` | Loads the unmodified AoS and SoA Triton V1 Stage1 kernels from `reference/` through small import stubs |
| `tq4_v2_stage1.py` | Corrected strong SoA Triton V2 Stage1 baseline; fixes the invalid V-column interleave path |
| `tq4_stage2.py` | Triton log-sum-exp reduction across 32 KV splits |
| `tq4_soa_store.py` | Standalone adapter for the unmodified vLLM SoA TurboQuant Store |

## What Each Test Proves

`bench_triton_baselines.py` uses one logical synthetic cache for every
implementation. It isolates Decode behavior and performance, but does not
exercise the quantization Store.

`bench_store_decode.py` covers the missing producer side:

```text
raw Q/K/V
  -> Hadamard rotation and 4-bit SoA Store
  -> one shared compressed cache tensor
     -> Triton V2-fixed Decode
     -> CUDA V7 Decode
  -> FP32 attention comparison for sequence 0
```

No AoS conversion or repacking is performed between Store and Decode.

## Reading Order

For a first code walkthrough, use this order:

```text
tq4_common.py
  -> tq4_v1_stage1.py
  -> tq4_v2_stage1.py
  -> tq4_stage2.py
  -> bench_triton_baselines.py
  -> bench_full_decode.py
  -> tq4_soa_store.py
  -> bench_store_decode.py
```

The fixed workload requires an NVIDIA CUDA GPU with enough memory for the
batch-64, context-4096 cache. CUDA V3-V7 are currently specialized for
`sm_86`; see `../cuda/README.md` for version-specific constraints.
