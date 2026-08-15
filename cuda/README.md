# CUDA Kernel 实现

本目录保存按版本演进的 CUDA 实验代码。它本身不是独立可执行程序；
`benchmarks/` 中的 Python 入口会把这些文件 JIT 编译成 PyTorch extension。

## 文件如何连接

```text
benchmarks/*.py
  -> torch.utils.cpp_extension.load(...)
  -> tq4_cuda_vN_bind.cpp       PyTorch/Python 导出层
  -> tq4_cuda_vN.cu             CUDA 入口和 kernel
  -> tq4_cuda_stage1_template.cuh  V4-V7 共享实现
  -> tq4_cuda_v8.cu / tq4_cuda_v9.cu  独立 m16n8k16 实现
```

每个版本的 `.cu` 文件包含 CUDA 实现，对应的 `_bind.cpp` 只负责声明并导出
可由 Python 调用的函数。`build_v*` 是 extension 编译时生成的目录，已经被
Git 忽略；在仓库根目录执行 `./run.sh clean` 即可删除。

## 版本地图

| 版本 | 核心变化 | 固定 workload Stage1 |
| --- | --- | ---: |
| V1 | 一个 CTA 负责 `(batch, KV head, split)`，四个 GQA head 复用解码后的 K/V | 2.074204 ms |
| V2 | 每个 Q head 使用一个 warp，softmax 保留在 warp 内且无 CTA barrier，但重复解码 K/V | 1.748470 ms |
| V3 | 单遍 tiled online softmax，使用 WMMA Tensor Core 完成 QK/PV，累加器保留在 fragment 中 | 1.380587 ms |
| V4 | 特化固定 128-token split，register centroid LUT，展开 tile 循环 | 1.121208 ms |
| V5 | 从 WMMA fragment 把有效行直接写入 `mid_o`，移除大型 output scratch | 0.845486 ms |
| V6 | 使用对齐 `uint32` packed-cache load 和 `half2` 重建值 store | 0.638863 ms |
| V7 | 合并相邻 tile 的同步，并加入 CUDA Stage2 和 Full Decode launcher | 0.631122 ms |
| V8 | 转置 QK/PV，使用原生 `mma.sync.m16n8k16` 提高 GQA-4 Tensor Core 利用率 | 0.513208 ms |
| V9 | 在 MMA accumulator 上融合 FlashInfer 式 online-softmax state update | 0.484516 ms |

这些是 RTX 4090 原生 `sm_89` 固定 workload 下的五轮中位数；根 README 的实测表是正式
记录。V1-V9 表示优化历史，不是 production API 版本号。

V1、V2 各自拥有完整的 `.cu` 实现。V3 也是独立实现，并首次建立 Tensor
Core 执行图。V4-V7 共同复用：

```text
tq4_cuda_stage1_template.cuh
```

对应的小型 `.cu` 文件只选择入口名称并逐步打开以下特性：

| 宏 | 首次启用 | 作用 |
| --- | --- | --- |
| `TQ4_DIRECT_WRITE` | V5 | WMMA fragment 直接写回 |
| `TQ4_VECTOR_DECODE` | V6 | packed 数据向量化解码与写入 |
| `TQ4_FUSED_TILE_BARRIER` | V7 | 移除每个 tile 中冗余的 barrier |

V8 不使用该模板。它通过 inline PTX 直接调用 `m16n8k16`，将 QK 和 PV
转置为以 8 列承载 4 个 GQA head，从而把静态 HMMA 指令点从 160 降到 80，
并将 shared memory 从 14224 B 降到 10336 B。

V9 继承 V8 的 MMA 布局，但将 QK score 保留在 warp 0 的 registers
中直接更新四个 online-softmax state。它不再实体化 `qk_s`，将每个
tile 的 CTA barrier 从 4 次降到 3 次。

## Stage1、Stage2 与 Full Decode

V1-V6、V8 和 V9 只导出 Stage1。V7 是完整 CUDA 链路，导出以下函数：

| Python symbol | 执行内容 |
| --- | --- |
| `tq4_cuda_v7_stage1` | 解码压缩 K/V、计算每个 split 的 attention，并把 partial output 和 split LSE 写入 `mid_o` |
| `tq4_cuda_v7_stage2` | 使用 log-sum-exp 修正归并 32 个 split，生成最终 output 和 LSE |
| `tq4_cuda_v7_full` | 依次 launch Stage1 和 Stage2 |
| `tq4_cuda_v7` | Stage1 benchmark 使用的兼容别名 |

Stage2 单独 launch，是因为归并前必须确保所有 split CTA 都已经完成。普通
CUDA kernel 不提供 grid-wide barrier；保留 kernel 边界可以明确表达该依赖，
也不需要 cooperative launch 的额外限制。

## 诊断文件

`wmma_fragment_probe.cu` 用于恢复并验证 `sm_89` WMMA accumulator 的
lane-to-row 映射，V5 的 fragment 直接写回依赖该结论。它是开发诊断程序，
不参与 benchmark 或 Full Decode 链路。

## 编译与运行

在仓库根目录执行：

```bash
./run.sh smoke       # 编译 V8/V9 Stage1，并执行 V7 Full Decode 短回归
./run.sh benchmark   # 编译和测量 V1-V9
./run.sh clean       # 删除生成的 build_v* 目录
```

性能测量入口见 [`../benchmarks/README.md`](../benchmarks/README.md)。

## 推荐阅读顺序

希望先理解最终实现时，按下面顺序阅读：

```text
tq4_cuda_v9.cu
  -> tq4_cuda_v9_bind.cpp
  -> ../benchmarks/stage1.py

tq4_cuda_v7.cu
  -> tq4_cuda_stage1_template.cuh
  -> tq4_cuda_v7_bind.cpp
  -> ../benchmarks/full_decode.py
```

前三个文件是最新 Stage1 路径，后四个文件是当前 Full Decode 路径。
然后比较 V4-V6 文件中逐步增加的 feature macro。研究设计为什么变化时，再
回头阅读 V1-V3。

## 当前限制

- V3-V9 面向根 README 记录的固定 Qwen3-4B-shaped workload。
- V4-V7 要求每个 split 是对齐的 128 个 token。
- V5-V7 依赖实验验证的 `sm_89` WMMA fragment mapping。
- V6-V7 要求 packed cache 地址满足四字节对齐。
- V8/V9 仅支持 `sm_89`、固定 128-token split 和四字节对齐 packed cache。
- 这些 kernel 目前是研究 extension，尚未注册成可直接替换的 vLLM
  attention backend。
