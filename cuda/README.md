# CUDA Kernel Implementations

[中文版](README_CN.md)

This directory contains the versioned CUDA experiments. It is implementation
code rather than a standalone executable; the Python launchers in `baseline/`
JIT-compile these files as PyTorch extensions.

## How The Files Connect

```text
baseline/bench_*.py
  -> torch.utils.cpp_extension.load(...)
  -> tq4_cuda_vN_bind.cpp       PyTorch/Python exports
  -> tq4_cuda_vN.cu             CUDA entry point and kernel
  -> tq4_cuda_stage1_template.cuh  shared V4-V7 implementation
```

For each version, the `.cu` file owns the CUDA implementation and the matching
`_bind.cpp` file only declares and exports its callable functions. Directories
named `build_v*` are generated extension build products, are ignored by Git,
and may be removed with `./baseline/run.sh clean` from the repository root.

## Version Map

| Version | Main change | Fixed-workload Stage1 |
| --- | --- | ---: |
| V1 | One CTA per `(batch, KV head, split)`; shares decoded K/V across four GQA heads | 4.61 ms |
| V2 | One warp per Q head; warp-local online softmax and no CTA barriers, at the cost of repeated K/V decode | 4.04 ms |
| V3 | Single-pass tiled online softmax and WMMA Tensor Core QK/PV; accumulator stays in fragments | 2.29 ms |
| V4 | Fixed 128-token split specialization, register centroid LUT, unrolled tile loop | 2.24 ms |
| V5 | Writes valid WMMA fragment rows directly to `mid_o`, removing the large output scratch | 1.73 ms |
| V6 | Uses aligned `uint32` packed-cache loads and `half2` reconstructed-value stores | 1.41 ms |
| V7 | Fuses adjacent tile synchronization and adds CUDA Stage2 plus the Full Decode launcher | 1.40 ms |

Times are representative RTX 3090 medians for the repository's fixed workload;
the root README contains the authoritative recorded run and comparison table.
V1-V7 are optimization-history labels, not production API versions.

V1 and V2 each have a self-contained `.cu` implementation. V3 is also
self-contained and establishes the Tensor Core execution graph. V4-V7 reuse:

```text
tq4_cuda_stage1_template.cuh
```

Their small `.cu` files select the entry-point names and feature flags:

| Flag | Introduced | Effect |
| --- | --- | --- |
| `TQ4_DIRECT_WRITE` | V5 | Direct WMMA-fragment writeback |
| `TQ4_VECTOR_DECODE` | V6 | Packed vector decode and stores |
| `TQ4_FUSED_TILE_BARRIER` | V7 | Removes the redundant per-tile barrier |

## Stage1, Stage2, And Full Decode

V1-V6 export Stage1 only. V7 is the complete CUDA chain and exports:

| Python symbol | Work performed |
| --- | --- |
| `tq4_cuda_v7_stage1` | Decode compressed K/V, compute split attention, and write partial output plus split LSE to `mid_o` |
| `tq4_cuda_v7_stage2` | Merge 32 split results with log-sum-exp correction into final output and LSE |
| `tq4_cuda_v7_full` | Launch Stage1 followed by Stage2 |
| `tq4_cuda_v7` | Compatibility alias for the Stage1 benchmark |

Stage2 is a separate launch because all split CTAs must finish before their
partial results can be reduced. A normal CUDA kernel has no grid-wide barrier,
so keeping the boundary makes the dependency explicit and avoids a cooperative
launch constraint.

## Diagnostic File

`wmma_fragment_probe.cu` recovers and verifies the `sm_86` WMMA accumulator
lane-to-row mapping used by direct fragment writeback. It is a development
probe, not part of the benchmark or Full Decode path.

## Build And Run

From the repository root:

```bash
./baseline/run.sh smoke       # compile V7 and run a short regression
./baseline/run.sh benchmark   # compile and measure V1-V7
./baseline/run.sh clean       # remove generated build_v* directories
```

See [`../baseline/README.md`](../baseline/README.md) for every test entry point.

## Reading Order

To understand the final implementation first, read:

```text
tq4_cuda_v7.cu
  -> tq4_cuda_stage1_template.cuh
  -> tq4_cuda_v7_bind.cpp
  -> ../baseline/bench_full_decode.py
```

Then compare the feature macros in V4-V6. Read V1-V3 afterward when studying
why the execution design changed.

## Current Constraints

- V3-V7 target the fixed Qwen3-4B-shaped workload documented in the root README.
- V4-V7 require an aligned 128-token split.
- V5-V7 depend on the empirically verified `sm_86` WMMA fragment mapping.
- V6-V7 require the packed cache addresses to be four-byte aligned.
- These kernels are research extensions and are not yet registered as a
  drop-in vLLM attention backend.
