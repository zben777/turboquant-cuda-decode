#include <torch/extension.h>

torch::Tensor tq4_cuda_v4_cuda(
    torch::Tensor q_rot,
    torch::Tensor kv_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor centroids,
    torch::Tensor mid_o
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq4_cuda_v4", &tq4_cuda_v4_cuda, "TurboQuant CUDA V4 Stage1");
}
