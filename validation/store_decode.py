import argparse
import math

import torch

from benchmarks.stage1 import build_cuda_v7_extension, diff_stats
from baseline.common import (
    BATCH_SIZE,
    BLOCK_BYTES,
    BLOCK_SIZE,
    CONTEXT_LEN,
    GQA_GROUP_SIZE,
    HEAD_DIM,
    META_REGION_OFFSET,
    MSE_BITS,
    NUM_KV_HEADS,
    NUM_KV_SPLITS,
    NUM_Q_HEADS,
    NUM_SOA_FIELDS,
    SLOT_SIZE_ALIGNED,
    SOA_K_NORM,
    SOA_V_SCALE,
    TOTAL_PHYSICAL_BLOCKS,
    build_block_table,
    build_centroids,
    build_pair_lut,
    build_seq_lens,
)
from validation.soa_store import launch_soa_store
from baseline.stage2 import launch_stage2
from baseline.triton_v2 import launch_tq4_v2_stage1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store-chunk-tokens",
        type=int,
        default=8192,
        help="Number of raw KV tokens materialized for each Store launch.",
    )
    return parser.parse_args()


def build_hadamard(device: torch.device) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < HEAD_DIM:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(HEAD_DIM)).contiguous()


def fp32_attention_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical GQA attention for one sequence before KV quantization."""
    q = query.float()
    kv_head = torch.arange(NUM_Q_HEADS, device=query.device) // GQA_GROUP_SIZE
    k = key[:, kv_head, :].float().permute(1, 0, 2)
    v = value[:, kv_head, :].float().permute(1, 0, 2)
    scores = torch.einsum("hd,htd->ht", q, k) / math.sqrt(HEAD_DIM)
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum("ht,htd->hd", torch.softmax(scores, dim=-1), v)
    return output, lse


def check_store_metadata(kv_cache: torch.Tensor) -> None:
    metadata = (
        kv_cache.view(torch.float16)
        .view(TOTAL_PHYSICAL_BLOCKS, BLOCK_BYTES // 2)[:, META_REGION_OFFSET // 2 :]
        .view(
            TOTAL_PHYSICAL_BLOCKS,
            NUM_KV_HEADS,
            NUM_SOA_FIELDS,
            BLOCK_SIZE,
        )
    )
    if not bool(torch.isfinite(metadata).all()):
        raise RuntimeError("SoA Store produced NaN/Inf metadata")
    if not bool((metadata[:, :, SOA_K_NORM, :] > 0).all()):
        raise RuntimeError("SoA Store produced a non-positive K norm")
    if not bool((metadata[:, :, SOA_V_SCALE, :] > 0).all()):
        raise RuntimeError("SoA Store produced a non-positive V scale")


@torch.inference_mode()
def main():
    args = parse_args()
    if args.store_chunk_tokens <= 0:
        raise ValueError("--store-chunk-tokens must be positive")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Building raw Q/K/V and writing cache with vLLM SoA Store...")
    torch.manual_seed(20260814)
    rotation = build_hadamard(device)
    centroids = build_centroids(device)
    midpoints = ((centroids[:-1] + centroids[1:]) * 0.5).contiguous()
    pair_lut = build_pair_lut(centroids)
    kv_cache = torch.empty(
        TOTAL_PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        SLOT_SIZE_ALIGNED,
        dtype=torch.uint8,
        device=device,
    )
    total_tokens = BATCH_SIZE * CONTEXT_LEN
    reference_key = None
    reference_value = None
    for start in range(0, total_tokens, args.store_chunk_tokens):
        end = min(start + args.store_chunk_tokens, total_tokens)
        count = end - start
        key = torch.randn(
            count,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.float16,
            device=device,
        )
        value = torch.randn_like(key)
        if start == 0:
            reference_key = key[:CONTEXT_LEN].clone()
            reference_value = value[:CONTEXT_LEN].clone()
        slot_mapping = torch.arange(
            start,
            end,
            dtype=torch.int32,
            device=device,
        )
        launch_soa_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            rotation,
            midpoints,
            centroids,
        )
    torch.cuda.synchronize()
    if reference_key is None or reference_value is None:
        raise RuntimeError("Failed to retain the first sequence")
    check_store_metadata(kv_cache)
    query = torch.randn(
        BATCH_SIZE,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
    )
    q_rot = torch.matmul(query.float(), rotation).contiguous()
    block_table = build_block_table(device)
    seq_lens = build_seq_lens(device)
    mid_shape = (
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
    )
    mid_triton = torch.empty(mid_shape, dtype=torch.float32, device=device)
    mid_cuda = torch.empty_like(mid_triton)
    out_triton = torch.empty(
        BATCH_SIZE,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    out_cuda = torch.empty_like(out_triton)
    lse_triton = torch.empty(
        BATCH_SIZE,
        NUM_Q_HEADS,
        dtype=torch.float32,
        device=device,
    )
    lse_cuda = torch.empty_like(lse_triton)
    cuda_ext = build_cuda_v7_extension()
    print("Running Triton V2-fixed and CUDA V7 on the Store-generated cache...")
    launch_tq4_v2_stage1(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        pair_lut,
        mid_o=mid_triton,
    )
    launch_stage2(mid_triton, seq_lens, out_triton, lse_triton)
    cuda_ext.tq4_cuda_v7_full(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_cuda,
        out_cuda,
        lse_cuda,
    )
    fp32_output, fp32_lse = fp32_attention_sequence(
        query[0],
        reference_key,
        reference_value,
    )
    torch.cuda.synchronize()
    for name, tensor in (
        ("Triton output", out_triton),
        ("Triton LSE", lse_triton),
        ("CUDA output", out_cuda),
        ("CUDA LSE", lse_cuda),
        ("FP32 output", fp32_output),
        ("FP32 LSE", fp32_lse),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} produced NaN/Inf")
    print()
    print("Store -> Decode correctness")
    out_max, out_mean = diff_stats(out_triton, out_cuda)
    lse_max, lse_mean = diff_stats(lse_triton, lse_cuda)
    print("CUDA V7 vs Triton output max/mean: " f"{out_max:.8g} / {out_mean:.8g}")
    print("CUDA V7 vs Triton LSE    max/mean: " f"{lse_max:.8g} / {lse_mean:.8g}")
    if out_max > 1e-4 or lse_max > 1e-4:
        raise RuntimeError("CUDA V7 does not match Triton on the Store-generated cache")
    print()
    print("Quantized decode vs original FP32 attention (sequence 0)")
    out_max, out_mean = diff_stats(fp32_output, out_triton[0])
    lse_max, lse_mean = diff_stats(fp32_lse, lse_triton[0])
    print("Triton quantization output max/mean: " f"{out_max:.8g} / {out_mean:.8g}")
    print("Triton quantization LSE    max/mean: " f"{lse_max:.8g} / {lse_mean:.8g}")


if __name__ == "__main__":
    main()
