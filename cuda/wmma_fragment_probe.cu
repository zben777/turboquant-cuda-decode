#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

__global__ void probe(float *output) {
    __shared__ float matrix[16][16];
    for (int i = threadIdx.x; i < 256; i += 32) {
        matrix[i / 16][i % 16] = static_cast<float>((i / 16) * 100 + i % 16);
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
    wmma::load_matrix_sync(frag, &matrix[0][0], 16, wmma::mem_row_major);
    for (int i = 0; i < frag.num_elements; ++i) {
        output[threadIdx.x * frag.num_elements + i] = frag.x[i];
    }
}

int main() {
    float *device, host[32 * 8];
    cudaMalloc(&device, sizeof(host));
    probe<<<1, 32>>>(device);
    cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
    for (int lane = 0; lane < 32; ++lane) {
        std::printf("lane %2d:", lane);
        for (int i = 0; i < 8; ++i)
            std::printf(" %.0f", host[lane * 8 + i]);
        std::printf("\n");
    }
    cudaFree(device);
}
