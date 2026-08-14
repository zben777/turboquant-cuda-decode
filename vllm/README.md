# 摘取的 vLLM TurboQuant 源码

本目录 **不是完整的 vLLM checkout**。这里只保存从本地 vLLM 源码树中摘取
的 TurboQuant 相关文件，用于将 CUDA 实验与原始 backend 组织和实现对照。

摘取的文件保留原始 package 路径，且没有针对本研究框架进行修改。每个源码
文件都保留 vLLM 的 Apache-2.0 SPDX header，许可证见本目录的 `LICENSE`。

上游项目：<https://github.com/vllm-project/vllm>

快照准备日期：2026-08-14。由于本地源代码目录没有 Git metadata，无法追溯
并记录准确的上游 commit。

## 包含的路径

```text
vllm/model_executor/layers/quantization/turboquant/
vllm/v1/attention/backends/turboquant_attn.py
vllm/v1/attention/ops/triton_turboquant_store.py
vllm/v1/attention/ops/triton_turboquant_decode.py
vllm/v1/attention/ops/turboquant_soa/
vllm/v1/attention/ops/flydsl_turboquant_decode.py
```

这些文件仅用于来源追踪和源码对照。由于有意省略了完整 vLLM 中的其他依赖，
它们不能作为独立的 `vllm` package 直接导入。
