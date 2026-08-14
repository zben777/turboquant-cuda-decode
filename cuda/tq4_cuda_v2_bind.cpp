#include <torch/extension.h>

torch::Tensor tq4_cuda_v2_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
                               torch::Tensor block_table, torch::Tensor seq_lens,
                               torch::Tensor centroids, torch::Tensor mid_o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq4_cuda_v2", &tq4_cuda_v2_cuda, "TurboQuant CUDA V2 Stage1");
}
