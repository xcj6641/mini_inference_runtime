#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


__global__ void read_paged_kv_kernel(
    const __half* key_cache,
    const int32_t* block_table,
    __half* output,
    int64_t layer_idx,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim
) {
    int token_idx = blockIdx.x;
    int kv_head = blockIdx.y;
    int dim = threadIdx.x;

    if (token_idx >= num_tokens || dim >= head_dim) {
        return;
    }

    int logical_block = token_idx / block_size;
    int slot = token_idx % block_size;

    int physical_block = block_table[logical_block];

    int64_t cache_offset =
        ((((layer_idx * num_blocks + physical_block)
            * num_kv_heads + kv_head)
            * block_size + slot)
            * head_dim + dim);

    int64_t output_offset =
        ((token_idx * num_kv_heads + kv_head)
            * head_dim + dim);

    output[output_offset] = key_cache[cache_offset];
}


torch::Tensor paged_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t num_tokens,
    int64_t layer_idx
) {
    const auto num_blocks = key_cache.size(1);
    const auto num_kv_heads = key_cache.size(2);
    const auto block_size = key_cache.size(3);
    const auto head_dim = key_cache.size(4);

    auto output = torch::empty(
        {num_tokens, num_kv_heads, head_dim},
        key_cache.options()
    );

    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(head_dim);

    read_paged_kv_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        layer_idx,
        num_tokens,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim
    );

    return output;
}