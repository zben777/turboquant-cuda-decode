# 性能测试

| 入口 | 测量内容 |
| --- | --- |
| `python -m benchmarks.stage1` | AoS V1、SoA V1、V2-fixed 和可选 CUDA V1-V9 Stage1 |
| `python -m benchmarks.full_decode` | Triton V2-fixed 与 CUDA V7 的 Stage1、Stage2 和 Full Decode |

推荐从仓库根目录运行 `./run.sh smoke` 或 `./run.sh benchmark`，无需手写参数。
