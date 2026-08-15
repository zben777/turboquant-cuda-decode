# TurboQuant CUDA Stage1 Study

[中文版](README_CN.md)

This directory benchmarks TurboQuant `turboquant_4bit_nc` decode Stage1 on a
fixed Qwen3-4B-shaped workload. It is a kernel research harness, not yet a
drop-in vLLM backend.

## Repository Scope

The CUDA kernels, standalone Triton baselines, benchmark harness, and analysis
in this repository are the research implementation developed here. The
`vllm/` directory is different: it is a path-preserving extraction of the
original TurboQuant-related files from a local vLLM source tree, not a complete
vLLM checkout and not this repository's CUDA implementation. See
[`vllm/README.md`](vllm/README.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and licensing.

`reference/` contains flat, byte-identical copies of selected files from the
extracted `vllm/` tree. This intentional duplication lets the standalone
benchmark load the reference kernels without depending on a full vLLM install.

## Requirements

The recorded results were collected with the following environment:

```text
GPU             NVIDIA RTX 4090 (sm_89)
Python          3.11.15
PyTorch         2.5.1+cu121
Triton          3.7.1
Ninja           1.13.0
CUDA compiler   12.2
```

Install a CUDA-enabled PyTorch build suitable for the host, then install the
remaining dependencies. If CUDA is not found automatically, set `CUDA_HOME`
before launching a CUDA benchmark.

```bash
python -m pip install -r requirements.txt
export CUDA_HOME=/path/to/cuda  # only when auto-detection is insufficient
```

## Quick Start

Run these commands from the repository root. The first CUDA run JIT-compiles
the extension and can take a few minutes.

```bash
./run.sh smoke       # short Triton/CUDA V8-V9 correctness and timing
./run.sh store       # raw Q/K/V -> vLLM Store -> both decoders
./run.sh benchmark   # official Triton and CUDA V1-V9 measurements
```

The runner works from any current directory and accepts `PYTHON` and
`CUDA_HOME` overrides. Run `./run.sh help` to list all modes.

## Licensing

The extracted vLLM files under `vllm/` and their flat copies under
`reference/` retain the upstream Apache-2.0 license. No repository-wide
license has been selected yet for the original CUDA research code. Add a
top-level `LICENSE` before public release if you intend to grant reuse rights
for that code.

## Fixed Workload

| Parameter | Value |
| --- | ---: |
| GPU used for current results | NVIDIA RTX 4090 (`sm_89`) |
| Batch | 64 |
| Context length | 4096 |
| Q heads / KV heads | 32 / 8 |
| GQA group | 4 |
| Head dimension | 128 |
| KV block size | 16 |
| KV splits | 32 |
| K / V quantization | 4-bit Lloyd-Max / 4-bit uniform |
| Cache layout | AoS or SoA, depending on baseline |

Timing includes Stage1 only. Query rotation, pair-LUT construction, Stage2
split reduction, input construction, allocation, and JIT compilation are
outside the timed region.

## Baselines

The authoritative baseline command is:

```bash
python -B -m benchmarks.stage1 \
  --include-cuda --include-cuda-v2 --include-cuda-v3 \
  --include-cuda-v4 --include-cuda-v5 --include-cuda-v6 \
  --include-cuda-v7 --include-cuda-v8 --include-cuda-v9
```

The harness builds one logical cache, converts it losslessly between SoA and
AoS, rotates measurement order across five rounds, and reports medians over
100 CUDA-event-timed launches per round.

Current RTX 4090 results, compiled natively for `sm_89` (2026-08-15):

| Implementation | Stage1 median | Role |
| --- | ---: | --- |
| AoS Triton V1 | 1.692314 ms | Production/reference baseline |
| SoA Triton V1 | 1.284270 ms | Layout ablation |
| SoA Triton V2-fixed | 1.075988 ms | Strong baseline and primary target |
| CUDA V1 | 2.074204 ms | First CUDA candidate |
| CUDA V2 | 1.748470 ms | Warp-per-Q experiment |
| CUDA V3 | 1.380587 ms | Single-pass Tensor Core candidate |
| CUDA V4 | 1.121208 ms | Fixed-workload `sm_89` candidate |
| CUDA V5 | 0.845486 ms | Direct WMMA register writeback candidate |
| CUDA V6 | 0.638863 ms | Vectorized INT4 decode candidate |
| CUDA V7 | 0.631122 ms | Fused tile-barrier candidate |
| CUDA V8 | 0.513208 ms | Native `m16n8k16` GQA-4 candidate |
| CUDA V9 | 0.484516 ms | FlashInfer-style fused online-softmax candidate |

Measured improvements:

```text
AoS V1 -> SoA V1       1.318x
SoA V1 -> V2-fixed     1.194x
AoS V1 -> V2-fixed     1.573x
CUDA V1 -> CUDA V2     1.186x
CUDA V1 vs V2-fixed    1.924x slower
CUDA V2 vs V2-fixed    1.625x slower
CUDA V2 -> CUDA V3     1.266x
CUDA V3 vs V2-fixed    1.283x slower
CUDA V3 -> CUDA V4     1.231x
CUDA V4 vs V2-fixed    1.042x slower
CUDA V4 -> CUDA V5     1.326x
CUDA V5 vs V2-fixed    1.272x faster
CUDA V5 -> CUDA V6     1.323x
CUDA V6 vs V2-fixed    1.684x faster
CUDA V6 -> CUDA V7     1.012x
CUDA V7 vs V2-fixed    1.705x faster
CUDA V7 -> CUDA V8     1.230x
CUDA V8 vs V2-fixed    2.097x faster
CUDA V8 -> CUDA V9     1.059x
CUDA V9 vs V2-fixed    2.221x faster
```

Correctness against SoA Triton V1 on the full Stage1 output:

```text
AoS V1 output max abs       1.28e-05
V2-fixed output max abs     9.68e-05
CUDA V1 output max abs      6.85e-07
CUDA V2 output max abs      6.85e-07
CUDA V3 output max abs      9.68e-05
CUDA V4 output max abs      9.68e-05
CUDA V5 output max abs      9.68e-05
CUDA V6 output max abs      9.68e-05
CUDA V7 output max abs      9.68e-05
CUDA V8 output max abs      9.68e-05
CUDA V9 output max abs      9.68e-05
```

## Version And Entry-Point Map

The CUDA version number records each successive **Stage1 optimization**. It
does not mean that every historical version has three decode stages:

| CUDA source | Stage1 | Stage2 | Full | Main change |
| --- | :---: | :---: | :---: | --- |
| `tq4_cuda_v1.cu` | yes | - | - | first CUDA candidate |
| `tq4_cuda_v2.cu` | yes | - | - | warp per Q head |
| `tq4_cuda_v3.cu` | yes | - | - | single-pass WMMA Tensor Core |
| `tq4_cuda_v4.cu` | yes | - | - | fixed-workload specialization |
| `tq4_cuda_v5.cu` | yes | - | - | direct fragment writeback |
| `tq4_cuda_v6.cu` | yes | - | - | vectorized packed-cache decode |
| `tq4_cuda_v7.cu` | yes | yes | yes | fused barriers plus complete decode |
| `tq4_cuda_v8.cu` | yes | - | - | native `m16n8k16` GQA-4 Tensor Core path |
| `tq4_cuda_v9.cu` | yes | - | - | register-resident fused online softmax |

V4 through V7 select compile-time optimizations and include the shared Stage1
implementation from `cuda/tq4_cuda_stage1_template.cuh`. The template is not
another benchmark version. V8 and V9 are independent Stage1 implementations
because their transposed `m16n8k16` fragment layout differs from the WMMA template. V7
exports three explicit Python entry points:

```text
tq4_cuda_v7_stage1(...)  -> Stage1 only; produces mid_o
tq4_cuda_v7_stage2(...)  -> Stage2 only; consumes mid_o
tq4_cuda_v7_full(...)    -> calls V7 Stage1, then V7 Stage2
```

The older `tq4_cuda_v7(...)` name remains as a compatibility alias for
`tq4_cuda_v7_stage1(...)`, so existing benchmark commands keep working.

## Complete Decode

The complete-decode benchmark adds the Stage2 log-sum-exp reduction across all
32 KV splits:

```bash
python -B -m benchmarks.full_decode
```

Here, "complete decode" means pre-rotated `q_rot` plus compressed KV through
Stage1 and Stage2 to the final `[B,Hq,D]` attention output and `[B,Hq]` LSE.
Query rotation and cache store remain outside the benchmark.

Five-round RTX 4090 `sm_89` medians:

| Implementation | Stage1 | Stage2 | Full decode |
| --- | ---: | ---: | ---: |
| Triton V2-fixed | 1.066916 ms | 0.024852 ms | 1.104148 ms |
| CUDA V7 | 0.631255 ms | 0.008868 ms | 0.664842 ms |

```text
CUDA V7 full vs Triton full  1.661x faster
CUDA Stage2 share             1.33% of full decode
```

Final-output correctness against the Triton complete decode:

```text
CUDA Stage2 output max abs    2.38e-07
CUDA Stage2 LSE max abs       9.54e-07
CUDA V7 full output max abs   5.66e-07
CUDA V7 full LSE max abs      9.54e-07
```

The CUDA Stage2 kernel uses 40 registers/thread and 136 B shared/CTA.

`tools/cuda_v1_diagnostic.py` additionally diagnoses one worst-case split
against canonical FP32 and a PyTorch mimic of the Triton FP16 tensor-core
path.

## Real Store Compatibility

The standalone compatibility test runs the unmodified vLLM SoA Triton Store
before decode. It starts from raw FP16 Q/K/V, performs the real K rotation,
Lloyd-Max bucketization, K/V 4-bit packing, norm/scale/zero metadata writes,
and Q rotation, then passes the same cache tensor directly to Triton V2-fixed
and CUDA V7:

```bash
python -B -m validation.store_decode
```

No cache conversion or byte rearrangement occurs between Store and Decode.
RTX 4090 `sm_89` correctness results:

```text
CUDA V7 vs Triton output max/mean  5.0617382e-06 / 1.3898631e-07
CUDA V7 vs Triton LSE    max/mean  9.5367432e-07 / 2.4447218e-07
```

For sequence 0, the same test also compares quantized Triton decode with
attention computed from the original, unquantized FP32 Q/K/V:

```text
Quantization output max/mean  0.018739756 / 0.0028449418
Quantization LSE    max/mean  0.0064659119 / 0.0024851561
```

The much smaller CUDA-vs-Triton difference shows that CUDA V7 correctly
consumes the production Store layout; the remaining larger difference from
FP32 is the expected 4-bit quantization error rather than a cache-layout error.

## V2 Correctness Fix

The copied upstream V2 reconstructs V with `tl.interleave(v_lo, v_hi)` and
feeds that value directly to `tl.dot`. On CUDA this produced a silent V-column
permutation: LSE remained correct, but output max error reached about `0.325`.

`baseline/triton_v2.py` is the fixed research baseline. It constructs the
final `[TILE_SIZE, BLOCK_D]` V layout directly with `d // 2` byte indices and
per-dimension nibble shifts. Its max output error against canonical FP32 is
about `8.5e-05`.

The old erroneous V2 timing near `2.29 ms` is invalid and must not be used.

## CUDA Candidates

CUDA V1 maps one CTA to `(batch, KV head, split)`. Each thread owns one D
coordinate for all four GQA heads. This reuses decoded K/V, but performs four
warp reductions in each of four warps per token and synchronizes through
shared memory.

CUDA V2 maps one warp to each Q head and gives every lane four D coordinates.
It reduces QK once per warp, keeps online softmax warp-local, and uses no
shared memory or CTA barriers. Static cubin resources changed from:

```text
CUDA V1: 40 registers/thread, 1328 B shared/CTA, 20 SHFL.DOWN sites
CUDA V2: 40 registers/thread,    0 B shared/CTA,  5 SHFL.DOWN sites
```

The tradeoff is four-way repeated K/V loads and dequantization. The experiment
improves Stage1 by 1.186x, but remains 1.623x slower than V2-fixed.

CUDA V3 now follows the Triton execution graph in one 16-token tile loop:
dequantize K/V, compute grouped QK, update online softmax, rescale the PV
accumulator by row, and compute PV. The accumulator remains in WMMA registers
across all eight tiles. This removes the original two-pass score/weight staging
and improves over CUDA V2 by 1.265x. It remains 1.283x slower than V2-fixed on
RTX 4090, with the same expected FP16 Tensor Core error scale.

Static cubin inspection confirms actual Tensor Core code generation:

```text
CUDA V3: 59 registers/thread, 21408 B shared/CTA
CUDA V3: 20 static HMMA.16816.F32 instruction sites
```

CUDA V4 keeps the V3 online-softmax design and specializes it further for the
fixed aligned workload. It performs one block-table load per 16-token tile,
initializes only the four real Q rows, uses a warp-register centroid LUT, and
fully unrolls the eight tile iterations. The result is a further 1.231x over
V3 and remains 1.043x slower than V2-fixed.

```text
CUDA V4: 58 registers/thread, 21408 B shared/CTA
CUDA V4: 160 static HMMA sites and 42 static BAR.SYNC sites
```

The V4 static counts include all eight fully unrolled tile iterations; they do
not represent additional runtime MMA or barriers relative to the loop body.

CUDA V5 removes V4's full `16x128` shared-memory output scratch. The verified
`sm_89` WMMA row/column mapping lets the first four valid accumulator rows be
written directly from fragment registers to their final `mid_o` addresses.
The Tensor Core work is unchanged, but shared memory drops by about 7 KB and
the full accumulator shared store/reload path disappears.

```text
CUDA V5: 50 registers/thread, 14224 B shared/CTA
CUDA V5: 1.326x faster than V4, 1.272x faster than V2-fixed
```

CUDA V6 keeps V5's direct register writeback and vectorizes the packed-cache
decode. Each aligned `uint32` load supplies four packed K or V bytes, and
`half2` stores write the corresponding eight reconstructed dimensions. This
reduces load, address-generation, and shared-store instruction counts without
changing occupancy or Tensor Core work.

```text
CUDA V6: 50 registers/thread, 14224 B shared/CTA
CUDA V6: 1.324x faster than V5, 1.684x faster than V2-fixed
```

CUDA V7 removes the barrier at the end of each tile. The following tile's
metadata writes use independent shared storage, and its existing opening
barrier simultaneously waits for the preceding PV MMA and publishes the new
metadata. This reduces executed CTA barriers from 42 to 34 without changing
the cache, WMMA, or occupancy configuration.

```text
CUDA V7: 49 registers/thread, 14224 B shared/CTA
CUDA V7: 34 static BAR.SYNC sites, 1.705x faster than V2-fixed
```

CUDA V8 replaces the C++ WMMA `m16n16k16` path with native
`mma.sync.m16n8k16`. Four adjacent KV heads cannot be stacked as 16 Q rows in
one dense MMA because each GQA group has a different K/V matrix. V8 instead
transposes both products within each valid group: QK computes
`K(16x16) * Q^T(16x8)`, and PV computes `V^T(16x16) * P(16x8)`. The four real
Q heads occupy four of eight N columns instead of four of sixteen M rows.

This halves the static HMMA sites, stores only four Q rows in shared memory,
halves QK scratch, and halves each PV accumulator fragment. Direct PTX fragment
loads cost two extra registers per thread, but the net Stage1 result is 1.230x
faster than V7.

```text
CUDA V8: 51 registers/thread, 10336 B shared/CTA
CUDA V8: 80 static HMMA sites, 34 static BAR.SYNC sites
CUDA V8: 1.230x faster than V7, 2.097x faster than V2-fixed
```

CUDA V9 applies FlashInfer's register-resident attention-state pattern to the
quantized V8 path. Warp 0 reduces the four GQA-head score columns directly
from MMA accumulators, updates their online-softmax `(m, l)` states, and writes
only FP16 probabilities for PV. This removes the `qk_s` round trip and one
CTA barrier per 16-token tile.

```text
CUDA V9: 50 registers/thread, 9824 B shared/CTA
CUDA V9: 80 static HMMA sites, 26 static BAR.SYNC sites
CUDA V9: 1.059x faster than V8, 2.221x faster than V2-fixed
```

## Directory Map

| Path | Role | Start here |
| --- | --- | --- |
| `baseline/` | Four reusable Triton baseline modules: common, V1, V2-fixed, and Stage2 | [Baseline guide](baseline/README.md) |
| `benchmarks/` | Authoritative Stage1 and Full Decode performance entry points | [Benchmark guide](benchmarks/README.md) |
| `validation/` | Real vLLM SoA Store-to-Decode compatibility test | [Validation guide](validation/README.md) |
| `tools/` | Historical CUDA V1 and focused profiler diagnostics | [Tools guide](tools/README.md) |
| `cuda/` | Versioned CUDA entry points, shared Stage1 template, and bindings | [CUDA guide](cuda/README.md) |
| `reference/` | Unmodified flat snapshots copied from the source vLLM tree | [Provenance](reference/README.md) |
| `results/` | Pre-fix V2 Nsight Compute reports and SASS export | [Results notes](results/README.md) |
| `docs/` | Historical profiling and design notes | [V2 Stage1 notes](docs/v2_stage1.md) |
| `vllm/` | Path-preserving extraction of upstream vLLM TurboQuant sources | [Extraction guide](vllm/README.md) |

## Known Limits

- Existing Nsight Compute reports predate the V2 correctness fix. Re-profile
  V2-fixed before using their exact metrics for optimization decisions.
- A CUDA V1 NCU attempt failed with `ERR_NVGPUCTRPERM`; no dynamic counter
  claims are made for CUDA V1/V2. Cubin resource and SASS instruction counts
  above come from `cuobjdump`.
- CUDA V1 assumes the fixed aligned workload. Its page mapping is not correct
  for arbitrary sequence lengths or split boundaries.
- CUDA V3 currently supports at most 128 tokens per split, matching the fixed
  `context=4096, splits=32` workload. It is an optimization candidate rather
  than a general production kernel.
- CUDA V4 requires exactly 128 aligned tokens per split. It deliberately trades
  general sequence/split support for the fixed-workload fast path.
- CUDA V5 inherits V4's fixed-workload and `sm_89` fragment-mapping constraints.
  Its direct writeback must be re-derived and tested before targeting another
  GPU architecture.
- CUDA V6 additionally assumes four-byte alignment for packed K/V vector loads;
  the fixed 128-byte slot layout satisfies that contract.
- CUDA V7 relies on the next tile's metadata barrier to protect the preceding
  PV inputs before K/V shared storage is overwritten.
- CUDA V8/V9 are Stage1-only, `sm_89`-specific inline-PTX experiments. They keep
  V7's fixed 128-token split and four-byte packed-cache alignment contracts.
- CUDA V3 scales WMMA accumulator fragments in registers using the empirically
  verified `sm_89` lane-to-row mapping. `cuda/wmma_fragment_probe.cu` reproduces
  that mapping. Treat this code path as architecture-specific until it is
  replaced by an explicit inline `mma.sync` register contract.
- Synthetic pages are contiguous and all sequences have length 4096. The fixed
  workload has also been tested with real Store-generated caches; random block
  tables and variable sequence lengths remain to be tested.
- `reference/soa_decode_v2.py` intentionally preserves the copied upstream
  implementation and therefore still contains the `tl.interleave` issue.
- The copied backend's logical cache shape declaration is head-major, while
  its store/decode launchers treat dimensions as position-major. Resolve that
  production integration contract before upstreaming a CUDA kernel.
- The repository does not yet integrate CUDA V7/V8/V9 into the production vLLM
  backend; the standalone fixed-workload harness is the current integration
  boundary.
