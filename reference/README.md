# Reference 快照

本目录保存从 vLLM 摘取的部分 TurboQuant 文件的扁平、未经修改的快照。
独立 benchmark 可以直接加载这些文件，无需安装完整 vLLM。

相同的上游文件还按照原始 package 路径保存在 `../vllm/` 中。这里有意保留
重复副本；准备本仓库时，两处对应文件逐字节一致。

| 扁平快照 | `vllm/` 下的原始路径 |
| --- | --- |
| `config.py` | `model_executor/layers/quantization/turboquant/config.py` |
| `centroids.py` | `model_executor/layers/quantization/turboquant/centroids.py` |
| `turboquant_attn.py` | `v1/attention/backends/turboquant_attn.py` |
| `aos_store.py` | `v1/attention/ops/triton_turboquant_store.py` |
| `aos_decode.py` | `v1/attention/ops/triton_turboquant_decode.py` |
| `soa_store.py` | `v1/attention/ops/turboquant_soa/triton_turboquant_store.py` |
| `soa_decode_v1.py` | `v1/attention/ops/turboquant_soa/triton_turboquant_decode.py` |
| `soa_decode_v2.py` | `v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.py` |
| `soa_unified.py` | `v1/attention/ops/turboquant_soa/triton_turboquant_unified_attention.py` |
| `flydsl_decode.py` | `v1/attention/ops/flydsl_turboquant_decode.py` |

这些文件保留原始 Apache-2.0 SPDX header。许可证与第三方来源说明见
`../vllm/LICENSE` 和 `../THIRD_PARTY_NOTICES.md`。
