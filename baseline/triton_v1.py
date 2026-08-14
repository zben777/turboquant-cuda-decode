import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import triton

from .common import (
    BLOCK_SIZE,
    GQA_GROUP_SIZE,
    HEAD_DIM,
    KEY_DATA_BYTES,
    META_REGION_OFFSET,
    MSE_BITS,
    MSE_BYTES,
    NUM_KV_HEADS,
    NUM_KV_SPLITS,
    NUM_SOA_FIELDS,
    SOA_K_NORM,
    SOA_V_SCALE,
    SOA_V_ZERO,
    VALUE_BITS,
    VAL_DATA_BYTES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference"


def _install_reference_import_stubs() -> None:
    """Provide only the vLLM symbols needed to define Stage1 kernels."""
    module_names = (
        "vllm",
        "vllm.platforms",
        "vllm.triton_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.ops",
        "vllm.v1.attention.ops.triton_decode_attention",
    )
    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.platforms"].current_platform = types.SimpleNamespace(
        is_cuda_alike=lambda: True,
    )
    sys.modules["vllm.triton_utils"].tl = __import__(
        "triton.language",
        fromlist=["language"],
    )
    sys.modules["vllm.triton_utils"].triton = triton
    # The reference files import Stage2, but this harness calls Stage1 only.
    sys.modules["vllm.v1.attention.ops.triton_decode_attention"]._fwd_kernel_stage2 = None


def _load_stage1(filename: str, module_name: str):
    _install_reference_import_stubs()
    spec = importlib.util.spec_from_file_location(
        module_name,
        REFERENCE_DIR / filename,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference kernel {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module._tq_decode_stage1


_aos_stage1 = _load_stage1(
    "aos_decode.py",
    "_tq_reference_aos_decode",
)

_soa_stage1 = _load_stage1(
    "soa_decode_v1.py",
    "_tq_reference_soa_decode_v1",
)


BLOCK_D = triton.next_power_of_2(HEAD_DIM)
BLOCK_KV = 4
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)
AOS_KEY_PACKED_SIZE = MSE_BYTES + 2


def launch_aos_v1_stage1(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    mid_o: torch.Tensor,
) -> None:
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    assert D == HEAD_DIM
    assert Hk == NUM_KV_HEADS
    assert Hq // Hk == GQA_GROUP_SIZE
    grid = (
        B,
        Hq,
        NUM_KV_SPLITS,
    )
    _aos_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=GQA_GROUP_SIZE,
        MSE_BITS=MSE_BITS,
        MSE_BYTES=MSE_BYTES,
        KPS=AOS_KEY_PACKED_SIZE,
        VQB=VALUE_BITS,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        ATTN_SCALE=ATTN_SCALE,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=0,
        NORM_CORRECTION=1,
        FP8_E4B15=0,
        num_warps=1,
        num_stages=1,
    )


def launch_soa_v1_stage1(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    mid_o: torch.Tensor,
) -> None:
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    assert D == HEAD_DIM
    assert Hk == NUM_KV_HEADS
    assert Hq // Hk == GQA_GROUP_SIZE
    grid = (
        B,
        Hq,
        NUM_KV_SPLITS,
    )
    _soa_stage1[grid](
        q_rot,
        kv_cache,
        kv_cache.view(torch.uint16),
        block_table,
        seq_lens,
        centroids,
        mid_o,
        None,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=GQA_GROUP_SIZE,
        MSE_BITS=MSE_BITS,
        MSE_BYTES=MSE_BYTES,
        VQB=VALUE_BITS,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        KEY_DATA_BYTES=KEY_DATA_BYTES,
        META_REGION_OFFSET=META_REGION_OFFSET,
        NUM_SOA_FIELDS=NUM_SOA_FIELDS,
        SOA_K_NORM=SOA_K_NORM,
        SOA_V_SCALE=SOA_V_SCALE,
        SOA_V_ZERO=SOA_V_ZERO,
        ATTN_SCALE=ATTN_SCALE,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=0,
        NORM_CORRECTION=1,
        FP8_E4B15=0,
        USE_SINKS=0,
        num_warps=1,
        num_stages=1,
    )
