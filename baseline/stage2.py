import torch
import triton
import triton.language as tl

from .common import HEAD_DIM, NUM_KV_SPLITS


@triton.jit
def tq4_stage2_kernel(
    mid_o,
    output,
    lse,
    seq_lens,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_out_b,
    stride_out_h,
    stride_lse_b,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(seq_lens + b)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    base = b * stride_mid_b + h * stride_mid_h
    for split in range(NUM_SPLITS):
        split_len = tl.cdiv(seq_len, NUM_SPLITS)
        split_start = split_len * split
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            value = tl.load(
                mid_o + base + split * stride_mid_s + d,
                mask=d_mask,
                other=0.0,
            )
            split_lse = tl.load(mid_o + base + split * stride_mid_s + D)
            new_max = tl.maximum(running_max, split_lse)
            old_scale = tl.exp(running_max - new_max)
            weight = tl.exp(split_lse - new_max)
            acc = acc * old_scale + weight * value
            running_sum = running_sum * old_scale + weight
            running_max = new_max
    tl.store(
        output + b * stride_out_b + h * stride_out_h + d,
        acc / running_sum,
        mask=d_mask,
    )
    tl.store(lse + b * stride_lse_b + h, running_max + tl.log(running_sum))


def launch_stage2(
    mid_o: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    grid = (mid_o.shape[0], mid_o.shape[1])
    tq4_stage2_kernel[grid](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_SPLITS=NUM_KV_SPLITS,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        D=HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )
