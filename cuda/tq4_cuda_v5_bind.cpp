#include <torch/extension.h>

torch::Tensor tq4_cuda_v5_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
                               torch::Tensor block_table, torch::Tensor seq_lens,
                               torch::Tensor centroids, torch::Tensor mid_o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq4_cuda_v5", &tq4_cuda_v5_cuda, "TurboQuant CUDA V5 Stage1");
}
