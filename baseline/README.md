# Triton Baseline

本目录只保存可以复用的 Triton baseline 实现：

| 文件 | 作用 |
| --- | --- |
| `common.py` | 固定 Qwen3-4B workload、134-byte cache layout、测试 tensor、centroid 和 pair LUT |
| `triton_v1.py` | 从 `reference/` 加载未经修改的 AoS V1 与 SoA V1 Stage1 kernel |
| `triton_v2.py` | 修复 correctness 的 SoA V2-fixed Stage1，也是主要性能 baseline |
| `stage2.py` | 对 Stage1 的 32 个 KV split 执行 log-sum-exp 归并 |

Baseline 实现与测量它的程序有意分开。日常使用统一运行根目录的
`../run.sh`；其他职责分别位于：

- [`../benchmarks/`](../benchmarks/README.md)：正式性能测量
- [`../validation/`](../validation/README.md)：Store 到 Decode 的完整验证
- [`../tools/`](../tools/README.md)：历史诊断与 profiler 入口

推荐阅读顺序：

```text
common.py -> triton_v1.py -> triton_v2.py -> stage2.py
```
