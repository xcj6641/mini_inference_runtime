// app/runtime/cuda/paged_attention_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


torch::Tensor paged_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t num_tokens,
    int64_t layer_idx
) {
    // Day 20 Step 2:
    // We are not computing attention yet.
    //
    // Just prove:
    //
    // Python -> C++ -> CUDA
    //
    // works.

    auto output = torch::zeros_like(query);

    return output;
}