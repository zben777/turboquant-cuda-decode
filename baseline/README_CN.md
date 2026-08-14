# 独立 Baseline 与验证框架

[English](README.md)

`baseline/` 是整个仓库最主要的运行入口，负责构造固定的 Qwen3-4B workload、
启动 Triton baseline 与 CUDA 版本、验证数值正确性并测量延迟。

文件命名规则为：

```text
bench_*.py    可直接运行的 benchmark 或 correctness 入口
tq4_*.py      被入口脚本复用的数据布局、Triton kernel、Store 或 Stage2 模块
profile_*.py  面向 NCU 等外部 profiler 的单次 launch 入口
```

## 快速开始

在仓库根目录执行：

```bash
./baseline/run.sh smoke
```

第一次运行 CUDA 会通过 PyTorch JIT 编译 extension，可能需要几分钟。后续运行
会复用 `cuda/build_v*/`；执行 `./baseline/run.sh clean` 可以删除这些生成目录。
它们已经被 Git 忽略。

`run.sh` 支持以下模式：

| 模式 | 作用 | 开销 |
| --- | --- | --- |
| `check` | 解析全部 Python 源码 | 只用 CPU，数秒 |
| `layout` | 验证 synthetic SoA layout 和输入 tensor | 短 GPU 检查 |
| `smoke` | 短版 Triton/CUDA V7 Stage1 与 Full 回归 | 编译一个 V7 |
| `store` | 原始 Q/K/V 经过真实 SoA Store 后交给两种 Decode | correctness 测试 |
| `benchmark` | 正式五轮 Triton 与 CUDA V1-V7 测试 | 编译全部 CUDA 版本 |
| `all` | 完整发布前回归 | 时间最长 |
| `clean` | 删除 CUDA extension 编译产物 | 不运行测试 |

需要指定环境时：

```bash
PYTHON=/path/to/python CUDA_HOME=/path/to/cuda ./baseline/run.sh smoke
```

## 可执行入口

| 文件 | 作用 |
| --- | --- |
| `bench_triton_baselines.py` | 正式 Stage1 对比：AoS V1、SoA V1、V2-fixed，以及可选 CUDA V1-V7 |
| `bench_full_decode.py` | 分别测量 Triton V2-fixed 与 CUDA V7 的 Stage1、Stage2 和 Full Decode |
| `bench_store_decode.py` | 从原始 FP16 Q/K/V 开始调用未修改的 vLLM SoA Store，再比较 Triton/CUDA 最终结果与 FP32 attention |
| `bench_v2_stage1.py` | 单独测量或 profile 修复后的 Triton V2 Stage1 |
| `bench_cuda_v1.py` | 历史 CUDA V1 诊断，包括与 FP32 和 Tensor Core mimic 的 split 级比较 |
| `profile_cuda_v1.py` | 为 NCU 等外部 profiler 准备的最小 CUDA V1 warmed launch |

## 可复用模块

| 文件 | 作用 |
| --- | --- |
| `tq4_common.py` | 固定参数、134-byte SoA slot 格式、synthetic cache、block table、centroid 和 pair LUT |
| `tq4_v1_stage1.py` | 通过轻量 import stub 从 `reference/` 加载未修改的 AoS/SoA Triton V1 Stage1 |
| `tq4_v2_stage1.py` | 修复 V-column interleave 错误后的强 SoA Triton V2 Stage1 baseline |
| `tq4_stage2.py` | 对 32 个 KV split 执行 log-sum-exp 归并的 Triton Stage2 |
| `tq4_soa_store.py` | 未修改 vLLM SoA TurboQuant Store 的独立运行适配器 |

## 不同测试分别证明什么

`bench_triton_baselines.py` 让所有实现读取同一份逻辑 synthetic cache，用于隔离
Decode correctness 与性能，但不覆盖量化 Store。

`bench_store_decode.py` 补上生产者一侧：

```text
原始 Q/K/V
  -> Hadamard rotation 与 4-bit SoA Store
  -> 同一个压缩 cache tensor
     -> Triton V2-fixed Decode
     -> CUDA V7 Decode
  -> sequence 0 的 FP32 attention 对照
```

Store 和 Decode 之间没有 AoS conversion 或重新 packing。

## 推荐阅读顺序

第一次阅读代码时按下面顺序：

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

固定 workload 需要一张显存足以容纳 batch-64、context-4096 cache 的 NVIDIA
CUDA GPU。CUDA V3-V7 当前针对 `sm_86` 特化；各版本限制见
`../cuda/README_CN.md`。
