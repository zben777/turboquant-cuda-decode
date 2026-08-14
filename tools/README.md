# 开发工具

这些历史诊断工具不属于日常 benchmark 链路：

| 文件 | 作用 |
| --- | --- |
| `cuda_v1_diagnostic.py` | CUDA V1 correctness 与 Tensor Core mimic 的详细分析 |
| `profile_cuda_v1.py` | 供外部 profiler 使用的最小 CUDA V1 warmed launch |
| `benchmark_triton_v2.py` | 单独测量或 profile V2-fixed 的入口 |
