#include <torch/extension.h>

torch::Tensor tq4_cuda_v7_stage1_cuda(
    torch::Tensor q_rot,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor centroids,
    torch::Tensor mid_o
);

torch::Tensor tq4_cuda_v7_full_cuda(
    torch::Tensor q_rot,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor centroids,
    torch::Tensor mid_o,
    torch::Tensor output,
    torch::Tensor lse
);

torch::Tensor tq4_cuda_v7_stage2_cuda(
    torch::Tensor mid_o,
    torch::Tensor output,
    torch::Tensor lse
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tq4_cuda_v7_stage1",
        &tq4_cuda_v7_stage1_cuda,
        "TurboQuant CUDA V7 Stage1"
    );
    // Compatibility alias used by the existing Stage1 benchmark.
    m.def(
        "tq4_cuda_v7",
        &tq4_cuda_v7_stage1_cuda,
        "TurboQuant CUDA V7 Stage1 (compatibility alias)"
    );
    m.def(
        "tq4_cuda_v7_stage2",
        &tq4_cuda_v7_stage2_cuda,
        "TurboQuant CUDA V7 Stage2"
    );
    m.def(
        "tq4_cuda_v7_full",
        &tq4_cuda_v7_full_cuda,
        "TurboQuant CUDA V7 full decode"
    );
}
