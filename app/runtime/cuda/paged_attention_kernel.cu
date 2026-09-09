// app/runtime/cuda/paged_attention_kernel.cu

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


__global__ void paged_qk_kernel(
    const __half* query,
    const __half* key_cache,
    const int32_t* block_table,
    float* output,
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

    if (
        token_idx >= num_tokens ||
        kv_head >= num_kv_heads ||
        dim >= head_dim
    ) {
        return;
    }

    // ------------------------------------------------------------
    // Step 1:
    // Find where this logical token lives in the paged KV cache.
    // ------------------------------------------------------------

    int logical_block = token_idx / block_size;
    int slot = token_idx % block_size;

    int physical_block = block_table[logical_block];

    // ------------------------------------------------------------
    // Step 2:
    // Calculate the flattened key_cache address:
    //
    // key_cache[
    //     layer_idx,
    //     physical_block,
    //     kv_head,
    //     slot,
    //     dim,
    // ]
    // ------------------------------------------------------------

    int64_t key_offset =
        ((((layer_idx * num_blocks + physical_block)
            * num_kv_heads + kv_head)
            * block_size + slot)
            * head_dim + dim);

    // ------------------------------------------------------------
    // Step 3:
    // query shape:
    //
    // [1, num_kv_heads, head_dim]
    //
    // batch_size is frozen to 1.
    //
    // query[0, kv_head, dim]
    // ------------------------------------------------------------

    int64_t query_offset =
        kv_head * head_dim + dim;

    // ------------------------------------------------------------
    // Step 4:
    // Every thread computes ONE element of:
    //
    // Q[dim] * K[dim]
    //
    // Accumulate in FP32.
    // ------------------------------------------------------------

    float q = __half2float(query[query_offset]);
    float k = __half2float(key_cache[key_offset]);

    float partial = q * k;

    // ------------------------------------------------------------
    // Step 5:
    // All threads in this CUDA block need to add their partial
    // products together.
    //
    // head_dim = 64, so we have 64 partial values.
    // ------------------------------------------------------------

    __shared__ float partial_sums[64];

    partial_sums[dim] = partial;

    __syncthreads();

    // ------------------------------------------------------------
    // Parallel reduction:
    //
    // 64 values
    //   ↓
    // 32 values
    //   ↓
    // 16
    //   ↓
    // 8
    //   ↓
    // 4
    //   ↓
    // 2
    //   ↓
    // 1
    // ------------------------------------------------------------

    for (int stride = head_dim / 2; stride > 0; stride /= 2) {
        if (dim < stride) {
            partial_sums[dim] += partial_sums[dim + stride];
        }

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Thread 0 now owns the complete dot product.
    //
    // output shape:
    //
    // [num_tokens, num_kv_heads]
    // ------------------------------------------------------------

    if (dim == 0) {
        int64_t output_offset =
            token_idx * num_kv_heads + kv_head;

        output[output_offset] = partial_sums[0];
    }
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

    // Q·K score:
    //
    // one score for each:
    //
    // (logical token, KV head)

    auto output = torch::empty(
        {num_tokens, num_kv_heads},
        key_cache.options().dtype(torch::kFloat32)
    );

    dim3 grid(
        num_tokens,
        num_kv_heads
    );

    dim3 block(
        head_dim
    );

    paged_qk_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(
            query.data_ptr<at::Half>()
        ),
        reinterpret_cast<const __half*>(
            key_cache.data_ptr<at::Half>()
        ),
        block_table.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        layer_idx,
        num_tokens,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim
    );

    return output;
}