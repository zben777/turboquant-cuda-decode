import torch

from .cuda_v1_diagnostic import build_extension
from baseline.common import (
    BATCH_SIZE,
    HEAD_DIM,
    NUM_KV_SPLITS,
    NUM_Q_HEADS,
    build_inputs,
)


def main():
    device = torch.device("cuda")
    extension = build_extension()
    inputs = build_inputs(device)

    mid_o = torch.empty(
        BATCH_SIZE,
        NUM_Q_HEADS,
        NUM_KV_SPLITS,
        HEAD_DIM + 1,
        dtype=torch.float32,
        device=device,
    )

    def launch():
        extension.tq4_cuda_v1(
            inputs["q_rot"],
            inputs["kv_cache"],
            inputs["block_table"],
            inputs["seq_lens"],
            inputs["centroids"],
            mid_o,
        )

    for _ in range(5):
        launch()

    torch.cuda.synchronize()
    launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
