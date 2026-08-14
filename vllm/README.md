# Extracted vLLM TurboQuant Sources

This directory is **not a complete vLLM checkout**. It preserves only the
TurboQuant-related files extracted from a local vLLM source tree so that the
CUDA experiments can be compared with the original backend organization.

The extracted files are kept in their original package paths and have not
been modified for this research harness. Each source file retains the vLLM
Apache-2.0 SPDX header. See `LICENSE` in this directory.

Upstream project: <https://github.com/vllm-project/vllm>

Snapshot prepared: 2026-08-14. The local source tree did not contain Git
metadata, so the exact upstream commit cannot be recorded retroactively.

## Included paths

```text
vllm/model_executor/layers/quantization/turboquant/
vllm/v1/attention/backends/turboquant_attn.py
vllm/v1/attention/ops/triton_turboquant_store.py
vllm/v1/attention/ops/triton_turboquant_decode.py
vllm/v1/attention/ops/turboquant_soa/
vllm/v1/attention/ops/flydsl_turboquant_decode.py
```

These files are provided for provenance and source comparison. They cannot be
imported as a standalone `vllm` package because their normal dependencies from
the rest of vLLM are intentionally absent.
