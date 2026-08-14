import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton

from baseline.common import MSE_BITS, MSE_BYTES, NORM_CORRECTION, VALUE_BITS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference"
REFERENCE_PACKAGE = "_tq_reference_soa"


def _install_reference_import_stubs() -> None:
    """Provide the small vLLM import surface needed by the upstream store."""
    for name in ("vllm", "vllm.triton_utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.triton_utils"].tl = __import__(
        "triton.language",
        fromlist=["language"],
    )
    sys.modules["vllm.triton_utils"].triton = triton
    if REFERENCE_PACKAGE not in sys.modules:
        package = types.ModuleType(REFERENCE_PACKAGE)
        package.__path__ = []
        sys.modules[REFERENCE_PACKAGE] = package
    decode_name = f"{REFERENCE_PACKAGE}.triton_turboquant_decode"
    if decode_name not in sys.modules:
        decode_stub = types.ModuleType(decode_name)
        decode_stub._use_fp8_e4b15 = lambda device=0: 0
        sys.modules[decode_name] = decode_stub


def _load_reference_store():
    _install_reference_import_stubs()
    module_name = f"{REFERENCE_PACKAGE}.triton_turboquant_store"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REFERENCE_DIR / "soa_store.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load reference/soa_store.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.triton_turboquant_store


_reference_soa_store = _load_reference_store()


def launch_soa_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation: torch.Tensor,
    midpoints: torch.Tensor,
    centroids: torch.Tensor,
) -> None:
    """Run the unmodified vLLM SoA TurboQuant store on standalone tensors."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must have the same [N,Hk,D] shape")
    if key.device != kv_cache.device or value.device != kv_cache.device:
        raise ValueError("key, value, and kv_cache must share a CUDA device")
    if slot_mapping.shape != (key.shape[0],):
        raise ValueError("slot_mapping must have shape [N]")
    _reference_soa_store(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        PiT=rotation,
        midpoints=midpoints,
        mse_bits=MSE_BITS,
        key_packed_size=MSE_BYTES + 2,
        value_quant_bits=VALUE_BITS,
        key_fp8=False,
        centroids=centroids,
        norm_correction=NORM_CORRECTION,
    )
