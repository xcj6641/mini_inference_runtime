// app/runtime/cuda/paged_attention.cpp

#include <torch/extension.h>


// Implemented in paged_attention_kernel.cu
torch::Tensor paged_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t num_tokens,
    int64_t layer_idx
);


torch::Tensor paged_attention_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t num_tokens,
    int64_t layer_idx
) {
    TORCH_CHECK(
        query.is_cuda(),
        "query must be a CUDA tensor"
    );

    TORCH_CHECK(
        key_cache.is_cuda(),
        "key_cache must be a CUDA tensor"
    );

    TORCH_CHECK(
        value_cache.is_cuda(),
        "value_cache must be a CUDA tensor"
    );

    TORCH_CHECK(
        block_table.is_cuda(),
        "block_table must be a CUDA tensor"
    );

    TORCH_CHECK(
        query.scalar_type() == torch::kFloat16,
        "query must be float16"
    );

    TORCH_CHECK(
        key_cache.scalar_type() == torch::kFloat16,
        "key_cache must be float16"
    );

    TORCH_CHECK(
        value_cache.scalar_type() == torch::kFloat16,
        "value_cache must be float16"
    );

    TORCH_CHECK(
        block_table.scalar_type() == torch::kInt32,
        "block_table must be int32"
    );

    TORCH_CHECK(
        query.dim() == 3,
        "query must have shape [batch, num_heads, head_dim]"
    );

    TORCH_CHECK(
        key_cache.dim() == 5,
        "key_cache must have shape "
        "[num_layers, num_blocks, num_kv_heads, block_size, head_dim]"
    );

    TORCH_CHECK(
        value_cache.dim() == 5,
        "value_cache must have shape "
        "[num_layers, num_blocks, num_kv_heads, block_size, head_dim]"
    );

    TORCH_CHECK(
        query.size(0) == 1,
        "Day 20 only supports batch_size=1"
    );

    TORCH_CHECK(
        query.size(1) == 2,
        "Day 20 only supports num_heads=2"
    );

    TORCH_CHECK(
        query.size(2) == 64,
        "Day 20 only supports head_dim=64"
    );

    TORCH_CHECK(
        key_cache.sizes() == value_cache.sizes(),
        "key_cache and value_cache must have the same shape"
    );

    TORCH_CHECK(
        key_cache.size(2) == 2,
        "Day 20 only supports num_kv_heads=2"
    );

    TORCH_CHECK(
        key_cache.size(3) == 16,
        "Day 20 only supports block_size=16"
    );

    TORCH_CHECK(
        key_cache.size(4) == 64,
        "Day 20 only supports head_dim=64"
    );

    TORCH_CHECK(
        layer_idx >= 0 && layer_idx < key_cache.size(0),
        "layer_idx is out of range"
    );

    TORCH_CHECK(
        num_tokens > 0,
        "num_tokens must be positive"
    );

    return paged_attention_cuda_forward(
        query,
        key_cache,
        value_cache,
        block_table,
        num_tokens,
        layer_idx
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &paged_attention_forward,
        "Minimal CUDA PagedAttention forward"
    );
}