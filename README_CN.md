# TurboQuant CUDA Stage1 研究

[English](README.md)

本目录在一组固定的、形状与 Qwen3-4B 相符的 workload 上，对 TurboQuant
`turboquant_4bit_nc` decode Stage1 进行基准测试。这是一个 CUDA kernel
研究与实验框架，目前还不是可以直接接入 vLLM 的完整 backend。

## 仓库范围

本仓库中的 CUDA kernel、独立 Triton baseline、benchmark 框架和分析文档，
是本项目开发的研究实现。`vllm/` 目录的性质不同：它按照原始路径保存了从
本地 vLLM 源码树中摘取的 TurboQuant 相关文件，不是完整的 vLLM checkout，
也不是本项目实现的 CUDA 代码。来源与许可证信息见
[`vllm/README.md`](vllm/README.md) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

`reference/` 保存了从 `vllm/` 摘取的部分文件的扁平、逐字节一致副本。
这部分重复是有意保留的，使独立 benchmark 无需安装完整 vLLM 就能加载
reference kernel。

## 环境要求

README 中记录的结果使用以下环境测得：

```text
GPU             NVIDIA RTX 3090 (sm_86)
Python          3.11
PyTorch         2.9.1+cu128
Triton          3.5.1
Ninja           1.13.0
CUDA compiler   12.2
```

请先安装适合本机驱动的 CUDA 版 PyTorch，再安装其余依赖。如果程序无法
自动找到 CUDA toolkit，请在运行 CUDA benchmark 前设置 `CUDA_HOME`。

```bash
python -m pip install -r requirements.txt
export CUDA_HOME=/path/to/cuda  # 仅在自动检测失败时需要
```

## 快速开始

请在仓库根目录执行以下命令。第一次运行 CUDA 时会 JIT 编译 extension，
可能需要几分钟。

```bash
./baseline/run.sh smoke       # 短版 Triton/CUDA V7 correctness 与计时
./baseline/run.sh store       # 原始 Q/K/V -> vLLM Store -> 两种 Decode
./baseline/run.sh benchmark   # 正式 Triton 与 CUDA V1-V7 测量
```

该 runner 可以从任意当前目录调用，并支持通过 `PYTHON` 和 `CUDA_HOME`
覆盖环境。其他模式及逐文件说明见 [baseline 指南](baseline/README_CN.md)。

## 许可证

`vllm/` 中摘取的 vLLM 文件以及 `reference/` 中对应的扁平副本继续使用上游
Apache-2.0 许可证。本项目原创的 CUDA 研究代码目前还没有选择仓库级许可证。
如果公开发布时希望允许他人复用这部分代码，需要在仓库根目录补充 `LICENSE`。

## 固定 Workload

| 参数 | 数值 |
| --- | ---: |
| 当前测试使用的 GPU | NVIDIA RTX 3090 (`sm_86`) |
| Batch | 64 |
| Context length | 4096 |
| Q heads / KV heads | 32 / 8 |
| GQA group | 4 |
| Head dimension | 128 |
| KV block size | 16 |
| KV splits | 32 |
| K / V 量化方式 | 4-bit Lloyd-Max / 4-bit uniform |
| Cache layout | 根据 baseline 分别使用 AoS 或 SoA |

这里的 Stage1 时间只包含 Stage1 kernel。本项计时不包含 Query rotation、
pair-LUT 构造、Stage2 split reduction、输入构造、内存分配和 JIT 编译。

## Baseline

正式 baseline 测试命令为：

```bash
cd baseline
python -B bench_triton_baselines.py \
  --include-cuda --include-cuda-v2 --include-cuda-v3 \
  --include-cuda-v4 --include-cuda-v5 --include-cuda-v6 \
  --include-cuda-v7
```

测试脚本首先构造一份逻辑 cache，然后在 SoA 和 AoS 之间进行无损转换。
测试共运行五轮，每轮轮换各实现的测量顺序，并对每个实现执行 100 次基于
CUDA event 的计时，最终报告五轮的中位数。

当前 RTX 3090 测试结果（2026-08-14）：

| 实现 | Stage1 中位时间 | 定位 |
| --- | ---: | --- |
| AoS Triton V1 | 3.407176 ms | Production/reference baseline |
| SoA Triton V1 | 3.207219 ms | Layout ablation |
| SoA Triton V2-fixed | 2.795724 ms | 强 baseline 和首要比较目标 |
| CUDA V1 | 4.602296 ms | 第一个 CUDA candidate |
| CUDA V2 | 4.024474 ms | Warp-per-Q 实验 |
| CUDA V3 | 2.303662 ms | 单遍 Tensor Core candidate |
| CUDA V4 | 2.251817 ms | 固定 workload 的 `sm_86` candidate |
| CUDA V5 | 1.724498 ms | WMMA fragment 直接写回 candidate |
| CUDA V6 | 1.409382 ms | 向量化 INT4 decode candidate |
| CUDA V7 | 1.392732 ms | 融合 tile barrier candidate |

实测提升如下：

```text
AoS V1 -> SoA V1       1.062x
SoA V1 -> V2-fixed     1.147x
AoS V1 -> V2-fixed     1.219x
CUDA V1 -> CUDA V2     1.144x
CUDA V1 vs V2-fixed    慢 1.646x
CUDA V2 vs V2-fixed    慢 1.440x
CUDA V2 -> CUDA V3     1.747x
CUDA V3 vs V2-fixed    快 1.214x
CUDA V3 -> CUDA V4     1.023x
CUDA V4 vs V2-fixed    快 1.242x
CUDA V4 -> CUDA V5     1.306x
CUDA V5 vs V2-fixed    快 1.621x
CUDA V5 -> CUDA V6     1.224x
CUDA V6 vs V2-fixed    快 1.984x
CUDA V6 -> CUDA V7     1.012x
CUDA V7 vs V2-fixed    快 2.007x
```

以 SoA Triton V1 的完整 Stage1 输出为参照，correctness 结果为：

```text
AoS V1 output max abs       1.21e-05
V2-fixed output max abs     8.46e-05
CUDA V1 output max abs      6.56e-07
CUDA V2 output max abs      7.15e-07
CUDA V3 output max abs      8.46e-05
CUDA V4 output max abs      8.46e-05
CUDA V5 output max abs      8.46e-05
CUDA V6 output max abs      8.46e-05
CUDA V7 output max abs      8.46e-05
```

## 版本与执行入口对应关系

CUDA 的版本号记录的是连续进行的每一轮 **Stage1 优化**。它并不表示每个
历史版本都同时拥有 Stage1、Stage2 和 Full 三个入口：

| CUDA 源文件 | Stage1 | Stage2 | Full | 主要变化 |
| --- | :---: | :---: | :---: | --- |
| `tq4_cuda_v1.cu` | 有 | - | - | 第一个 CUDA candidate |
| `tq4_cuda_v2.cu` | 有 | - | - | 每个 Q head 使用一个 warp |
| `tq4_cuda_v3.cu` | 有 | - | - | 单遍 WMMA Tensor Core |
| `tq4_cuda_v4.cu` | 有 | - | - | 针对固定 workload 特化 |
| `tq4_cuda_v5.cu` | 有 | - | - | fragment 直接写回 |
| `tq4_cuda_v6.cu` | 有 | - | - | 向量化 packed-cache decode |
| `tq4_cuda_v7.cu` | 有 | 有 | 有 | 融合 barrier，并加入完整 decode |

V4 到 V7 通过编译期选项启用不同优化，并包含共享 Stage1 实现
`cuda/tq4_cuda_stage1_template.cuh`。这个模板文件不是一个新的 benchmark
版本。V7 明确导出以下三个 Python 入口：

```text
tq4_cuda_v7_stage1(...)  -> 只运行 Stage1，生成 mid_o
tq4_cuda_v7_stage2(...)  -> 只运行 Stage2，读取 mid_o
tq4_cuda_v7_full(...)    -> 先调用 V7 Stage1，再调用 V7 Stage2
```

旧入口 `tq4_cuda_v7(...)` 继续作为 `tq4_cuda_v7_stage1(...)` 的兼容别名，
因此已有 benchmark 命令仍可正常运行。

## 完整 Decode

完整 decode benchmark 在 Stage1 之后加入 Stage2，对全部 32 个 KV split
执行 log-sum-exp reduction：

```bash
cd baseline
python -B bench_full_decode.py
```

这里的“完整 decode”指：从预先旋转好的 `q_rot` 和压缩 KV cache 开始，
依次经过 Stage1 和 Stage2，得到最终 `[B,Hq,D]` attention 输出与 `[B,Hq]`
LSE。Query rotation 和 cache store 仍不包含在 benchmark 中。

RTX 3090 五轮测试中位数：

| 实现 | Stage1 | Stage2 | 完整 decode |
| --- | ---: | ---: | ---: |
| Triton V2-fixed | 2.773299 ms | 0.051436 ms | 2.825226 ms |
| CUDA V7 | 1.392681 ms | 0.044339 ms | 1.436283 ms |

```text
CUDA V7 full vs Triton full  快 1.967x
CUDA Stage2 占完整 decode     3.09%
```

相对于 Triton 完整 decode 的最终输出 correctness：

```text
CUDA Stage2 output max abs    2.38e-07
CUDA Stage2 LSE max abs       9.54e-07
CUDA V7 full output max abs   5.96e-07
CUDA V7 full LSE max abs      9.54e-07
```

CUDA Stage2 kernel 使用 40 registers/thread 和 136 B shared/CTA。

`baseline/bench_cuda_v1.py` 还会选择一个误差最坏的 split，分别与标准 FP32
实现和模拟 Triton FP16 Tensor Core 路径的 PyTorch 实现进行诊断比较。

## 真实 Store 兼容性

独立兼容性测试会在 decode 前运行未经修改的 vLLM SoA Triton Store。测试从
原始 FP16 Q/K/V 开始，真实执行 K rotation、Lloyd-Max bucketize、K/V 4-bit
packing、norm/scale/zero metadata 写入和 Q rotation，然后将同一个 cache
tensor 直接交给 Triton V2-fixed 与 CUDA V7：

```bash
python -B baseline/bench_store_decode.py
```

Store 与 Decode 之间没有 cache conversion 或 byte rearrangement。RTX 3090
correctness 结果如下：

```text
CUDA V7 vs Triton output max/mean  4.4517219e-06 / 1.3544668e-07
CUDA V7 vs Triton LSE    max/mean  9.5367432e-07 / 2.5099143e-07
```

对于 sequence 0，该测试还会将量化后的 Triton decode 与使用原始、未量化
FP32 Q/K/V 计算的 attention 进行比较：

```text
Quantization output max/mean  0.013347317 / 0.0027882606
Quantization LSE    max/mean  0.0039367676 / 0.0016208589
```

CUDA 与 Triton 之间的差异远小于量化结果与 FP32 之间的差异。这说明 CUDA V7
能够正确读取 production Store layout；相对 FP32 的较大差异来自预期的 4-bit
量化误差，而不是 cache layout 错误。

## V2 Correctness 修复

从上游复制的 V2 使用 `tl.interleave(v_lo, v_hi)` 重建 V，然后把结果直接
传给 `tl.dot`。该写法在 CUDA 上悄悄产生了 V 列置换：LSE 仍然正确，但
输出最大误差达到了约 `0.325`。

`baseline/tq4_v2_stage1.py` 是修复后的研究 baseline。它使用 `d // 2`
字节索引和每个维度对应的 nibble shift，直接构造最终的
`[TILE_SIZE, BLOCK_D]` V layout。相对于标准 FP32 实现，其最大输出误差约为
`8.5e-05`。

旧的错误 V2 时间约为 `2.29 ms`，该数据已经作废，不能作为 baseline 使用。

## CUDA Candidate 演进

CUDA V1 将一个 CTA 映射到 `(batch, KV head, split)`。每个 thread 负责一个
D 坐标，并为四个 GQA head 处理该坐标。这种设计可以复用解码后的 K/V，
但每处理一个 token，四个 warp 都需要执行四次 warp reduction，并通过
shared memory 同步。

CUDA V2 为每个 Q head 分配一个 warp，并让每个 lane 负责四个 D 坐标。
每个 warp 只对 QK 做一次 reduction，online softmax 完全保留在 warp 内，
不使用 shared memory 或 CTA barrier。静态 cubin 资源从以下配置发生变化：

```text
CUDA V1: 40 registers/thread, 1328 B shared/CTA, 20 SHFL.DOWN sites
CUDA V2: 40 registers/thread,    0 B shared/CTA,  5 SHFL.DOWN sites
```

代价是 K/V load 和反量化被重复执行四次。该实验让 Stage1 提升约 15%，
但仍比 V2-fixed 慢 1.440x。

CUDA V3 在一个 16-token tile 循环中遵循 Triton 的执行图：反量化 K/V、
计算 grouped QK、更新 online softmax、按行重新缩放 PV accumulator，最后
计算 PV。全部八个 tile 的 accumulator 始终保留在 WMMA registers 中。
这消除了原来的两遍 score/weight staging，相比 CUDA V2 提升 1.747x，
相比 V2-fixed 快 1.214x，同时保持预期的 FP16 Tensor Core 误差量级。

静态 cubin 检查确认实际生成了 Tensor Core 指令：

```text
CUDA V3: 66 registers/thread, 21408 B shared/CTA
CUDA V3: 20 static HMMA.16816.F32 instruction sites
```

CUDA V4 保留 V3 的 online-softmax 设计，并进一步针对固定、对齐的 workload
进行特化。它让每个 16-token tile 只读取一次 block table，只初始化四个
实际使用的 Q 行，使用 warp-register centroid LUT，并完全展开八次 tile
迭代。最终相比 V3 继续提升 1.023x，相比 V2-fixed 快 1.242x。

```text
CUDA V4: 58 registers/thread, 21408 B shared/CTA
CUDA V4: 160 static HMMA sites and 42 static BAR.SYNC sites
```

V4 的静态计数包含完全展开后的全部八次 tile 迭代；与循环体相比，这不
代表运行时执行了额外的 MMA 或 barrier。

CUDA V5 移除了 V4 完整的 `16x128` shared-memory output scratch。利用已验证
的 `sm_86` WMMA row/column 映射，前四个有效 accumulator 行可以直接从
fragment registers 写入最终 `mid_o` 地址。Tensor Core 工作量没有变化，
但 shared memory 减少约 7 KB，并移除了完整 accumulator 的 shared-memory
写入与重新读取路径。

```text
CUDA V5: 58 registers/thread, 14224 B shared/CTA
CUDA V5: 比 V4 快 1.306x，比 V2-fixed 快 1.621x
```

CUDA V6 保留 V5 的 register 直接写回，并对 packed-cache decode 进行向量化。
每次对齐的 `uint32` load 提供四个 packed K 或 V 字节，`half2` store 写入
对应的八个重建维度。这减少了 load、地址生成和 shared store 指令数量，
同时不改变 occupancy 或 Tensor Core 工作量。

```text
CUDA V6: 58 registers/thread, 14224 B shared/CTA
CUDA V6: 比 V5 快 1.224x，比 V2-fixed 快 1.984x
```

CUDA V7 移除了每个 tile 末尾的 barrier。下一个 tile 的 metadata 写入使用
独立的 shared storage，而下一个 tile 已有的起始 barrier 会同时等待上一个
PV MMA 完成并发布新的 metadata。在不改变 cache、WMMA 或 occupancy 配置
的情况下，实际执行的 CTA barrier 数量从 42 降到了 34。

```text
CUDA V7: 58 registers/thread, 14224 B shared/CTA
CUDA V7: 34 static BAR.SYNC sites，比 V2-fixed 快 2.007x
```

## 目录结构

| 路径 | 作用 | 从这里开始 |
| --- | --- | --- |
| `baseline/` | 固定输入、独立 launcher、correctness 与计时 | [Baseline 指南](baseline/README_CN.md) |
| `cuda/` | 分版本 CUDA 入口、共享 Stage1 模板与 binding | [CUDA 指南](cuda/README_CN.md) |
| `reference/` | 从 vLLM 源码树复制且未经修改的扁平快照 | [来源说明](reference/README.md) |
| `results/` | V2 修复前的 Nsight Compute 报告和 SASS 导出 | [结果说明](results/README.md) |
| `docs/` | 历史 profiling 与设计记录 | [V2 Stage1 记录](docs/v2_stage1.md) |
| `vllm/` | 按原始路径保存的上游 vLLM TurboQuant 摘取源码 | [摘取说明](vllm/README.md) |

## 已知限制

- 现有 Nsight Compute 报告生成于 V2 correctness 修复之前。在依据精确指标
  作出优化判断之前，需要重新 profile V2-fixed。
- CUDA V1 的一次 NCU 测试因 `ERR_NVGPUCTRPERM` 失败，因此这里没有对 CUDA
  V1/V2 的动态计数器结果作出结论。上面的 cubin 资源与 SASS 指令数量来自
  `cuobjdump`。
- CUDA V1 假定使用固定且对齐的 workload，其 page mapping 对任意 sequence
  length 或 split boundary 并不正确。
- CUDA V3 当前每个 split 最多支持 128 个 token，与固定的
  `context=4096, splits=32` workload 相符。它是优化 candidate，不是通用的
  production kernel。
- CUDA V4 要求每个 split 恰好包含 128 个对齐 token。它有意放弃通用的
  sequence/split 支持，以换取固定 workload fast path。
- CUDA V5 继承了 V4 的固定 workload 和 `sm_86` fragment-mapping 限制。
  在面向其他 GPU 架构之前，需要重新推导并测试 direct writeback 映射。
- CUDA V6 还假定 packed K/V vector load 满足四字节对齐；固定的 128-byte
  slot layout 满足这一约束。
- CUDA V7 依赖下一个 tile 的 metadata barrier，在 K/V shared storage 被
  覆盖之前保护上一个 tile 的 PV 输入。
- CUDA V3 使用在 `sm_86` 上通过实验验证的 lane-to-row 映射，在 registers
  中缩放 WMMA accumulator fragment。`cuda/wmma_fragment_probe.cu` 可以复现
  该映射。在将其替换为具有明确 register contract 的 inline `mma.sync`
  之前，应将此路径视为特定架构实现。
- 合成页面是连续的，并且所有 sequence length 都是 4096。固定 workload
  已经使用真实 Store 生成的 cache 完成测试；随机 block table 和可变
  sequence length 仍有待测试。
- `reference/soa_decode_v2.py` 有意保留复制来的上游实现，因此其中仍然存在
  `tl.interleave` 问题。
- 复制来的 backend 在逻辑 cache shape 声明中采用 head-major，但其
  store/decode launcher 把维度当作 position-major。将 CUDA kernel 合入上游
  之前，需要解决该 production integration contract。
- CUDA V7 尚未集成进 production vLLM backend；当前的集成边界是独立、固定
  workload 的 benchmark 框架。
