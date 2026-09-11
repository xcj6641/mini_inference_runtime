#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>


__global__ void paged_qk_kernel(
    const __half* query,
    const __half* key_cache,
    const int32_t* block_table,
    float* scores,
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

    int logical_block = token_idx / block_size;
    int slot = token_idx % block_size;

    int physical_block =
        block_table[logical_block];

    int64_t key_offset =
        ((((layer_idx * num_blocks + physical_block)
            * num_kv_heads + kv_head)
            * block_size + slot)
            * head_dim + dim);

    int64_t query_offset =
        kv_head * head_dim + dim;

    float q = __half2float(
        query[query_offset]
    );

    float k = __half2float(
        key_cache[key_offset]
    );

    float partial = q * k;

    __shared__ float partial_sums[64];

    partial_sums[dim] = partial;

    __syncthreads();

    for (
        int stride = head_dim / 2;
        stride > 0;
        stride /= 2
    ) {
        if (dim < stride) {
            partial_sums[dim] +=
                partial_sums[dim + stride];
        }

        __syncthreads();
    }

    if (dim == 0) {
        int64_t score_offset =
            token_idx * num_kv_heads
            + kv_head;

        scores[score_offset] =
            partial_sums[0];
    }
}

__global__ void scaled_softmax_kernel(
    const float* scores,
    float* weights,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    int kv_head = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (kv_head >= num_kv_heads) {
        return;
    }

    extern __shared__ float shared_memory[];

    float* shared_scores = shared_memory;
    float* shared_reduction =
        shared_memory + num_tokens;

    float scale =
        1.0f / sqrtf(
            static_cast<float>(head_dim)
        );

    // A fixed-size CUDA block can process more
    // tokens than it has threads.
    for (
        int token_idx = thread_idx;
        token_idx < num_tokens;
        token_idx += blockDim.x
    ) {
        int64_t score_offset =
            token_idx * num_kv_heads
            + kv_head;

        float scaled_score =
            scores[score_offset] * scale;

        shared_scores[token_idx] =
            scaled_score;

        shared_reduction[token_idx] =
            scaled_score;
    }

    __syncthreads();

    // Stable softmax: serial maximum scan.
    if (thread_idx == 0) {
        float max_score =
            shared_reduction[0];

        for (
            int i = 1;
            i < num_tokens;
            ++i
        ) {
            max_score = fmaxf(
                max_score,
                shared_reduction[i]
            );
        }

        shared_reduction[0] =
            max_score;
    }

    __syncthreads();

    float max_score =
        shared_reduction[0];

    for (
        int token_idx = thread_idx;
        token_idx < num_tokens;
        token_idx += blockDim.x
    ) {
        shared_scores[token_idx] =
            expf(
                shared_scores[token_idx]
                - max_score
            );
    }

    __syncthreads();

    // Intentionally serial for the naive
    // Day 21 implementation.
    if (thread_idx == 0) {
        float sum_exp = 0.0f;

        for (
            int i = 0;
            i < num_tokens;
            ++i
        ) {
            sum_exp +=
                shared_scores[i];
        }

        shared_reduction[0] =
            sum_exp;
    }

    __syncthreads();

    float sum_exp =
        shared_reduction[0];

    for (
        int token_idx = thread_idx;
        token_idx < num_tokens;
        token_idx += blockDim.x
    ) {
        int64_t score_offset =
            token_idx * num_kv_heads
            + kv_head;

        weights[score_offset] =
            shared_scores[token_idx]
            / sum_exp;
    }
}



__global__ void paged_weighted_value_kernel(
    const __half* value_cache,
    const int32_t* block_table,
    const float* weights,
    float* output,
    int64_t layer_idx,
    int64_t num_tokens,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim
) {
    // One CUDA block = one KV head.
    int kv_head = blockIdx.x;

    // One thread = one output dimension.
    int dim = threadIdx.x;

    if (
        kv_head >= num_kv_heads ||
        dim >= head_dim
    ) {
        return;
    }

    float accumulator = 0.0f;

    // Each thread handles one output dimension and
    // loops over all logical tokens.
    for (
        int token_idx = 0;
        token_idx < num_tokens;
        ++token_idx
    ) {
        int logical_block =
            token_idx / block_size;

        int slot =
            token_idx % block_size;

        int physical_block =
            block_table[logical_block];

        int64_t value_offset =
            ((((layer_idx * num_blocks + physical_block)
                * num_kv_heads + kv_head)
                * block_size + slot)
                * head_dim + dim);

        int64_t weight_offset =
            token_idx * num_kv_heads
            + kv_head;

        float weight =
            weights[weight_offset];

        float value =
            __half2float(
                value_cache[value_offset]
            );

        accumulator +=
            weight * value;
    }

    // output shape:
    // [1, num_kv_heads, head_dim]
    //
    // batch size is frozen to 1.
    int64_t output_offset =
        kv_head * head_dim
        + dim;

    output[output_offset] =
        accumulator;
}


torch::Tensor paged_attention_cuda_forward(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t num_tokens,
    int64_t layer_idx
) {
    const auto num_blocks =
        key_cache.size(1);

    const auto num_kv_heads =
        key_cache.size(2);

    const auto block_size =
        key_cache.size(3);

    const auto head_dim =
        key_cache.size(4);

    // ------------------------------------------------------------
    // Kernel 1: Q · K
    // ------------------------------------------------------------

    auto scores = torch::empty(
        {
            num_tokens,
            num_kv_heads,
        },
        key_cache.options().dtype(
            torch::kFloat32
        )
    );

    dim3 qk_grid(
        num_tokens,
        num_kv_heads
    );

    dim3 qk_block(
        head_dim
    );

    paged_qk_kernel<<<
        qk_grid,
        qk_block
    >>>(
        reinterpret_cast<const __half*>(
            query.data_ptr<at::Half>()
        ),
        reinterpret_cast<const __half*>(
            key_cache.data_ptr<at::Half>()
        ),
        block_table.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        layer_idx,
        num_tokens,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim
    );

    // ------------------------------------------------------------
    // Kernel 2: scaled softmax
    // ------------------------------------------------------------

    auto weights =
        torch::empty_like(scores);

    dim3 softmax_grid(
        num_kv_heads
    );

    constexpr int SOFTMAX_THREADS = 256;

    dim3 softmax_block(
        SOFTMAX_THREADS
    );

    size_t shared_memory_bytes =
        2 * num_tokens * sizeof(float);

    scaled_softmax_kernel<<<
        softmax_grid,
        softmax_block,
        shared_memory_bytes
    >>>(
        scores.data_ptr<float>(),
        weights.data_ptr<float>(),
        num_tokens,
        num_kv_heads,
        head_dim
    );

    // ------------------------------------------------------------
    // Kernel 3:
    //
    // attention_output =
    // sum_t weights[t, h] * V[t, h, d]
    // ------------------------------------------------------------

    auto output = torch::empty(
        {
            1,
            num_kv_heads,
            head_dim,
        },
        key_cache.options().dtype(
            torch::kFloat32
        )
    );

    dim3 value_grid(
        num_kv_heads
    );

    dim3 value_block(
        head_dim
    );

    paged_weighted_value_kernel<<<
        value_grid,
        value_block
    >>>(
        reinterpret_cast<const __half*>(
            value_cache.data_ptr<at::Half>()
        ),
        block_table.data_ptr<int32_t>(),
        weights.data_ptr<float>(),
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