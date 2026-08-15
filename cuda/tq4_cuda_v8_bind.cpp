#include <torch/extension.h>

torch::Tensor tq4_cuda_v8_cuda(torch::Tensor q_rot, torch::Tensor kv_cache,
                               torch::Tensor block_table, torch::Tensor seq_lens,
                               torch::Tensor centroids, torch::Tensor mid_o);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq4_cuda_v8", &tq4_cuda_v8_cuda, "TurboQuant CUDA V8 Stage1 m16n8k16");
}
