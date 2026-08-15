import os
from pathlib import Path


# RTX 4090 = sm_89
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch

from torch.utils.cpp_extension import load

from baseline.common import (
    BATCH_SIZE,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    NUM_KV_SPLITS,
    HEAD_DIM,
    build_inputs,
)

from baseline.triton_v2 import (
    launch_tq4_v2_stage1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUDA_DIR = PROJECT_ROOT / "cuda"

BUILD_DIR = CUDA_DIR / "build_v1"

BUILD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def build_extension():
    print("Building CUDA V1 extension...")
    print("CUDA source :", CUDA_DIR / "tq4_cuda_v1.cu")
    print("Bind source :", CUDA_DIR / "tq4_cuda_v1_bind.cpp")
    print()
    ext = load(
        name="tq4_cuda_v1_ext",
        sources=[
            str(CUDA_DIR / "tq4_cuda_v1_bind.cpp"),
            str(CUDA_DIR / "tq4_cuda_v1.cu"),
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--use_fast_math",
        ],
        build_directory=str(BUILD_DIR),
        verbose=True,
        with_cuda=True,
    )
    return ext


def event_time_ms(
    fn,
    warmup=20,
    iters=100,
):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / iters


def print_diff(
    name,
    ref,
    test,
):
    abs_diff = (test - ref).abs()
    denom = ref.abs() + 1.0e-6
    rel_diff = abs_diff / denom
    print(name)
    print(
        "  max abs  :",
        float(abs_diff.max().item()),
    )
    print(
        "  mean abs :",
        float(abs_diff.mean().item()),
    )
    print(
        "  max rel  :",
        float(rel_diff.max().item()),
    )


def fp32_reference_one_split(
    q_rot,
    kv_cache,
    block_table,
    seq_lens,
    centroids,
    b,
    kvh,
    sid,
):
    """
    Canonical FP32 reference for one (batch, KV head, split).
    This directly decodes the SoA TurboQuant cache without Triton
    tl.dot and without CUDA V1.
    """
    gqa = NUM_Q_HEADS // NUM_KV_HEADS
    block_size = 16
    k_bytes = HEAD_DIM // 2
    v_bytes = HEAD_DIM // 2
    data_bytes_per_slot = k_bytes + v_bytes
    num_soa_fields = 3
    soa_k_norm = 0
    soa_v_scale = 1
    soa_v_zero = 2
    meta_region_offset = block_size * NUM_KV_HEADS * data_bytes_per_slot
    attn_scale = 1.0 / (HEAD_DIM**0.5)
    seq_len = int(seq_lens[b].item())
    split_len = (seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    split_start = sid * split_len
    split_end = min(
        split_start + split_len,
        seq_len,
    )
    tokens = torch.arange(
        split_start,
        split_end,
        device=q_rot.device,
        dtype=torch.long,
    )
    page_idx = tokens // block_size
    page_off = tokens % block_size
    block_nums = block_table[b, page_idx].to(torch.long)
    # The cache is byte-addressed.
    flat_u8 = kv_cache.contiguous().view(-1)
    block_stride = int(kv_cache.stride(0))
    data_base = (
        block_nums * block_stride
        + page_off * NUM_KV_HEADS * data_bytes_per_slot
        + kvh * data_bytes_per_slot
    )
    byte_offs = torch.arange(
        k_bytes,
        device=q_rot.device,
        dtype=torch.long,
    )
    k_packed = flat_u8[data_base[:, None] + byte_offs[None, :]]
    v_packed = flat_u8[data_base[:, None] + k_bytes + byte_offs[None, :]]
    k_lo = (k_packed & 0xF).to(torch.long)
    k_hi = ((k_packed >> 4) & 0xF).to(torch.long)
    k_idx = torch.stack(
        [k_lo, k_hi],
        dim=-1,
    ).reshape(
        -1,
        HEAD_DIM,
    )
    v_lo = (v_packed & 0xF).to(torch.float32)
    v_hi = ((v_packed >> 4) & 0xF).to(torch.float32)
    v_idx = torch.stack(
        [v_lo, v_hi],
        dim=-1,
    ).reshape(
        -1,
        HEAD_DIM,
    )
    # Reinterpret the same byte storage as FP16 metadata.
    flat_half = kv_cache.contiguous().view(torch.float16).view(-1)

    def load_meta(field):
        meta_byte_addr = (
            block_nums * block_stride
            + meta_region_offset
            + ((kvh * num_soa_fields + field) * block_size + page_off) * 2
        )
        half_idx = meta_byte_addr // 2
        return flat_half[half_idx].float()

    k_norm = load_meta(soa_k_norm)
    v_scale = load_meta(soa_v_scale)
    v_zero = load_meta(soa_v_zero)
    K = (centroids[k_idx] * k_norm[:, None]).float()
    V = (v_idx * v_scale[:, None] + v_zero[:, None]).float()
    qh0 = kvh * gqa
    Q = q_rot[
        b,
        qh0 : qh0 + gqa,
        :,
    ].float()
    scores = (
        torch.matmul(
            Q,
            K.transpose(0, 1),
        )
        * attn_scale
    )
    probs = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    )
    out = torch.matmul(
        probs,
        V,
    )
    lse = torch.logsumexp(
        scores,
        dim=-1,
    )
    return out, lse


def triton_mimic_reference_one_split(
    q_rot,
    kv_cache,
    block_table,
    seq_lens,
    centroids,
    b,
    kvh,
    sid,
):
    """PyTorch-level semantic mimic of Triton V2 Stage1."""
    gqa = NUM_Q_HEADS // NUM_KV_HEADS
    block_size = 16
    tile_size = 16
    k_bytes = HEAD_DIM // 2
    data_bytes_per_slot = HEAD_DIM
    num_soa_fields = 3
    soa_k_norm = 0
    soa_v_scale = 1
    soa_v_zero = 2
    meta_region_offset = block_size * NUM_KV_HEADS * data_bytes_per_slot
    attn_scale = 1.0 / (HEAD_DIM**0.5)
    rcp_ln2 = 1.4426950408889634
    ln2 = 0.6931471805599453
    seq_len = int(seq_lens[b].item())
    split_len = (seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    split_start = sid * split_len
    split_end = min(
        split_start + split_len,
        seq_len,
    )
    qh0 = kvh * gqa
    Q = q_rot[
        b,
        qh0 : qh0 + gqa,
        :,
    ].float()
    flat_u8 = kv_cache.contiguous().view(-1)
    flat_half = kv_cache.contiguous().view(torch.float16).view(-1)
    block_stride = int(kv_cache.stride(0))
    # Triton online-softmax states.
    M = torch.full(
        (gqa,),
        -float("inf"),
        dtype=torch.float32,
        device=q_rot.device,
    )
    L = torch.zeros(
        (gqa,),
        dtype=torch.float32,
        device=q_rot.device,
    )
    acc = torch.zeros(
        (gqa, HEAD_DIM),
        dtype=torch.float32,
        device=q_rot.device,
    )
    for tile_base in range(
        split_start,
        split_end,
        tile_size,
    ):
        tile_end = min(
            tile_base + tile_size,
            split_end,
        )
        tile_tokens = tile_end - tile_base
        tokens = torch.arange(
            tile_base,
            tile_end,
            device=q_rot.device,
            dtype=torch.long,
        )
        page_idx = tokens // block_size
        page_off = tokens % block_size
        block_nums = block_table[b, page_idx].to(torch.long)
        data_base = (
            block_nums * block_stride
            + page_off * NUM_KV_HEADS * data_bytes_per_slot
            + kvh * data_bytes_per_slot
        )
        byte_offs = torch.arange(
            k_bytes,
            device=q_rot.device,
            dtype=torch.long,
        )
        k_packed = flat_u8[data_base[:, None] + byte_offs[None, :]]
        v_packed = flat_u8[data_base[:, None] + k_bytes + byte_offs[None, :]]
        k_lo = (k_packed & 0xF).long()
        k_hi = ((k_packed >> 4) & 0xF).long()
        k_idx = torch.stack(
            [k_lo, k_hi],
            dim=-1,
        ).reshape(
            tile_tokens,
            HEAD_DIM,
        )
        v_lo = (v_packed & 0xF).float()
        v_hi = ((v_packed >> 4) & 0xF).float()
        v_idx = torch.stack(
            [v_lo, v_hi],
            dim=-1,
        ).reshape(
            tile_tokens,
            HEAD_DIM,
        )

        def load_meta(field):
            meta_byte_addr = (
                block_nums * block_stride
                + meta_region_offset
                + ((kvh * num_soa_fields + field) * block_size + page_off) * 2
            )
            return flat_half[meta_byte_addr // 2].float()

        k_norm = load_meta(soa_k_norm)
        v_scale = load_meta(soa_v_scale)
        v_zero = load_meta(soa_v_zero)
        K = (centroids[k_idx] * k_norm[:, None]).float()
        V = (v_idx * v_scale[:, None] + v_zero[:, None]).float()
        # Match tl.dot(Q.fp16, K.fp16).
        S = torch.matmul(
            Q.to(torch.float16),
            K.transpose(0, 1).to(torch.float16),
        ).float()
        S = S * attn_scale * rcp_ln2
        m_j = torch.maximum(
            M,
            S.max(dim=1).values,
        )
        m_j = torch.where(
            torch.isfinite(m_j),
            m_j,
            torch.zeros_like(m_j),
        )
        P = torch.exp2(S - m_j[:, None])
        l_j = P.sum(dim=1)
        alpha = torch.exp2(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        # Match tl.dot(P.fp16, V.fp16).
        pv = torch.matmul(
            P.to(torch.float16),
            V.to(torch.float16),
        ).float()
        acc += pv
    safe_L = torch.where(
        L > 0.0,
        L,
        torch.ones_like(L),
    )
    out = acc / safe_L[:, None]
    lse = M * ln2 + torch.log(safe_L)
    return out, lse


def main():
    print(
        "GPU       :",
        torch.cuda.get_device_name(0),
    )
    print(
        "CC        :",
        torch.cuda.get_device_capability(0),
    )
    print(
        "PyTorch   :",
        torch.__version__,
    )
    print(
        "CUDA      :",
        torch.version.cuda,
    )
    print()
    # Build extension outside timing.
    cuda_ext = build_extension()
    print()
    print("Extension build: OK")
    print()
    device = torch.device("cuda")
    print("Building common synthetic input...")
    inputs = build_inputs(device)
    q_rot = inputs["q_rot"]
    kv_cache = inputs["kv_cache"]
    block_table = inputs["block_table"]
    seq_lens = inputs["seq_lens"]
    centroids = inputs["centroids"]
    pair_lut = inputs["pair_lut"]
    # Same output shape for both implementations.
    mid_triton = torch.empty(
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
        dtype=torch.float32,
        device=device,
    )
    mid_cuda = torch.empty_like(mid_triton)
    print("Running Triton reference...")
    launch_tq4_v2_stage1(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        pair_lut,
        mid_o=mid_triton,
    )
    print("Running CUDA V1...")
    cuda_ext.tq4_cuda_v1(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_cuda,
    )
    torch.cuda.synchronize()
    # --------------------------------------------------
    # Correctness
    # --------------------------------------------------
    print()
    print("Correctness")
    print()
    finite_triton = bool(torch.isfinite(mid_triton).all().item())
    finite_cuda = bool(torch.isfinite(mid_cuda).all().item())
    print(
        "Triton finite :",
        finite_triton,
    )
    print(
        "CUDA finite   :",
        finite_cuda,
    )
    print()
    # Compare partial attention output separately from LSE.
    triton_out = mid_triton[..., :HEAD_DIM]
    cuda_out = mid_cuda[..., :HEAD_DIM]
    triton_lse = mid_triton[..., HEAD_DIM]
    cuda_lse = mid_cuda[..., HEAD_DIM]
    print_diff(
        "Partial output:",
        triton_out,
        cuda_out,
    )
    print()
    print_diff(
        "LSE:",
        triton_lse,
        cuda_lse,
    )
    # Diagnose the largest Triton-vs-CUDA mismatch.
    diff = (cuda_out - triton_out).abs()
    flat_idx = int(diff.argmax().item())
    d = flat_idx % HEAD_DIM
    tmp = flat_idx // HEAD_DIM
    sid = tmp % NUM_KV_SPLITS
    tmp //= NUM_KV_SPLITS
    qh = tmp % NUM_Q_HEADS
    b = tmp // NUM_Q_HEADS
    gqa = NUM_Q_HEADS // NUM_KV_HEADS
    kvh = qh // gqa
    qh0 = kvh * gqa
    print()
    print("Worst mismatch location")
    print(
        "  batch   :",
        b,
    )
    print(
        "  q_head  :",
        qh,
    )
    print(
        "  kv_head :",
        kvh,
    )
    print(
        "  split   :",
        sid,
    )
    print(
        "  dim     :",
        d,
    )
    ref_out, ref_lse = fp32_reference_one_split(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        b,
        kvh,
        sid,
    )
    mimic_out, mimic_lse = triton_mimic_reference_one_split(
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        b,
        kvh,
        sid,
    )
    cuda_slice = cuda_out[
        b,
        qh0 : qh0 + gqa,
        sid,
        :,
    ]
    triton_slice = triton_out[
        b,
        qh0 : qh0 + gqa,
        sid,
        :,
    ]
    cuda_lse_slice = cuda_lse[
        b,
        qh0 : qh0 + gqa,
        sid,
    ]
    triton_lse_slice = triton_lse[
        b,
        qh0 : qh0 + gqa,
        sid,
    ]
    print()
    print("Against canonical FP32 reference")
    print()
    print_diff(
        "CUDA V1 output vs FP32:",
        ref_out,
        cuda_slice,
    )
    print()
    print_diff(
        "Triton output vs FP32:",
        ref_out,
        triton_slice,
    )
    print()
    print_diff(
        "CUDA V1 LSE vs FP32:",
        ref_lse,
        cuda_lse_slice,
    )
    print()
    print_diff(
        "Triton LSE vs FP32:",
        ref_lse,
        triton_lse_slice,
    )
    print()
    print("Triton numerical-path mimic")
    print()
    print_diff(
        "Mimic output vs FP32:",
        ref_out,
        mimic_out,
    )
    print()
    print_diff(
        "Triton output vs Mimic:",
        mimic_out,
        triton_slice,
    )
    print()
    print_diff(
        "CUDA output vs Mimic:",
        mimic_out,
        cuda_slice,
    )
    print()
    print_diff(
        "Mimic LSE vs FP32:",
        ref_lse,
        mimic_lse,
    )
    print()
    print_diff(
        "Triton LSE vs Mimic:",
        mimic_lse,
        triton_lse_slice,
    )
    # CUDA V1 intentionally uses FP32 CUDA-core arithmetic
    # instead of Triton's FP16 Tensor-Core tl.dot.
    #
    # Therefore exact bit equality is NOT expected.
    output_ok = torch.allclose(
        cuda_out,
        triton_out,
        atol=5.0e-2,
        rtol=5.0e-2,
    )
    lse_ok = torch.allclose(
        cuda_lse,
        triton_lse,
        atol=5.0e-2,
        rtol=5.0e-2,
    )
    print()
    print(
        "Output allclose :",
        bool(output_ok),
    )
    print(
        "LSE allclose    :",
        bool(lse_ok),
    )
    if not (finite_cuda and output_ok and lse_ok):
        print()
        print("CUDA V1 correctness check FAILED.")
        print("Do not benchmark performance yet.")
        print("Send the diff values to me first.")
        return
    print()
    print("CUDA V1 correctness: PASS")
    # --------------------------------------------------
    # Performance
    # --------------------------------------------------
    print()
    print("Benchmark")
    print()

    def run_triton():
        launch_tq4_v2_stage1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            pair_lut,
            mid_o=mid_triton,
        )

    def run_cuda():
        cuda_ext.tq4_cuda_v1(
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_cuda,
        )

    triton_ms = event_time_ms(
        run_triton,
        warmup=20,
        iters=100,
    )
    cuda_ms = event_time_ms(
        run_cuda,
        warmup=20,
        iters=100,
    )
    speedup = triton_ms / cuda_ms
    print(f"Triton V2 Stage1 : " f"{triton_ms:.6f} ms")
    print(f"CUDA V1 Stage1   : " f"{cuda_ms:.6f} ms")
    print(f"CUDA V1 speedup  : " f"{speedup:.3f}x")
    print()
    print(
        "Grid              :",
        (
            BATCH_SIZE,
            NUM_KV_HEADS,
            NUM_KV_SPLITS,
        ),
    )
    print("CUDA block        : 128 threads")
    print("CTA mapping       : " "1 batch x 1 KV head x 1 split")
    print("Q mapping         : " "one thread owns one D coordinate " "for all 4 GQA heads")


if __name__ == "__main__":
    main()
