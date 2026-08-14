# Reference Snapshots

This directory contains unmodified, flat snapshots of selected TurboQuant
files extracted from vLLM. The standalone benchmark imports these files
without requiring the rest of vLLM.

The same upstream files also appear under `../vllm/` in their original package
paths. The duplicate copies are intentional and were byte-identical when this
repository was prepared.

| Flat snapshot | Original path under `vllm/` |
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

These files retain their original Apache-2.0 SPDX headers. See
`../vllm/LICENSE` and `../THIRD_PARTY_NOTICES.md`.
