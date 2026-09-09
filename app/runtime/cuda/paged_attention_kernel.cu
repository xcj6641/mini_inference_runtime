// app/runtime/cuda/paged_attention_kernel.cu

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

    // ------------------------------------------------------------
    // Logical token -> logical KV block + slot
    // ------------------------------------------------------------

    int logical_block = token_idx / block_size;
    int slot = token_idx % block_size;

    // ------------------------------------------------------------
    // Logical KV block -> physical KV block
    // ------------------------------------------------------------

    int physical_block =
        block_table[logical_block];

    // ------------------------------------------------------------
    // key_cache shape:
    //
    // [
    //     num_layers,
    //     num_blocks,
    //     num_kv_heads,
    //     block_size,
    //     head_dim
    // ]
    // ------------------------------------------------------------

    int64_t key_offset =
        ((((layer_idx * num_blocks + physical_block)
            * num_kv_heads + kv_head)
            * block_size + slot)
            * head_dim + dim);

    // query shape:
    //
    // [1, num_kv_heads, head_dim]

    int64_t query_offset =
        kv_head * head_dim + dim;

    float q = __half2float(
        query[query_offset]
    );

    float k = __half2float(
        key_cache[key_offset]
    );

    float partial = q * k;

    // Day 20 frozen head_dim = 64.
    __shared__ float partial_sums[64];

    partial_sums[dim] = partial;

    __syncthreads();

    // ------------------------------------------------------------
    // Reduce 64 partial products into one Q·K score.
    //
    // 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // ------------------------------------------------------------

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

    // One score per:
    //
    // [token_idx, kv_head]

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
    // ------------------------------------------------------------
    // One CUDA thread block handles ONE head.
    // ------------------------------------------------------------

    int kv_head = blockIdx.x;

    // ------------------------------------------------------------
    // One thread handles ONE token.
    //
    // thread 0 -> token 0
    // thread 1 -> token 1
    // ...
    // ------------------------------------------------------------

    int token_idx = threadIdx.x;

    if (
        kv_head >= num_kv_heads ||
        token_idx >= num_tokens
    ) {
        return;
    }

    // ------------------------------------------------------------
    // Dynamic shared memory.
    //
    // We need:
    //
    // shared_scores[num_tokens]
    // shared_reduction[num_tokens]
    //
    // Both live inside one CUDA thread block.
    // ------------------------------------------------------------

    extern __shared__ float shared_memory[];

    float* shared_scores =
        shared_memory;

    float* shared_reduction =
        shared_memory + num_tokens;

    int64_t score_offset =
        token_idx * num_kv_heads
        + kv_head;

    // ------------------------------------------------------------
    // Attention scaling:
    //
    // Q·K / sqrt(head_dim)
    // ------------------------------------------------------------

    float scale =
        1.0f / sqrtf(
            static_cast<float>(head_dim)
        );

    float scaled_score =
        scores[score_offset] * scale;

    shared_scores[token_idx] =
        scaled_score;

    __syncthreads();

    // ============================================================
    // Step 1: find maximum score
    //
    // Stable softmax uses:
    //
    // exp(score - max_score)
    //
    // instead of:
    //
    // exp(score)
    //
    // to avoid numerical overflow.
    // ============================================================

    shared_reduction[token_idx] =
        scaled_score;

    __syncthreads();

    // ------------------------------------------------------------
    // This deliberately simple reduction works for our Day 20
    // small token counts.
    //
    // Thread 0 finds the maximum serially.
    // This is NOT optimized.
    // ------------------------------------------------------------

    if (token_idx == 0) {
        float max_score =
            shared_reduction[0];

        for (
            int i = 1;
            i < num_tokens;
            ++i
        ) {
            max_score =
                fmaxf(
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

    // ============================================================
    // Step 2: exponentiate
    //
    // exp(score - max_score)
    // ============================================================

    float exp_score =
        expf(
            shared_scores[token_idx]
            - max_score
        );

    shared_scores[token_idx] =
        exp_score;

    __syncthreads();

    // ============================================================
    // Step 3: sum exponentials
    //
    // Again, thread 0 does this serially for simplicity.
    // ============================================================

    if (token_idx == 0) {
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

    // ============================================================
    // Step 4: normalize
    //
    // weight_i =
    //
    // exp(score_i - max)
    // ------------------
    // sum(exp(score_j - max))
    // ============================================================

    weights[score_offset] =
        exp_score / sum_exp;
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
    // Temporary Q·K scores:
    //
    // [num_tokens, num_kv_heads]
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

    // ------------------------------------------------------------
    // Kernel 1:
    //
    // one CUDA thread block per (token, head)
    //
    // one CUDA thread per head dimension
    // ------------------------------------------------------------

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
    // Attention weights:
    //
    // [num_tokens, num_kv_heads]
    // ------------------------------------------------------------

    auto weights = torch::empty_like(
        scores
    );

    // ------------------------------------------------------------
    // Kernel 2:
    //
    // one CUDA thread block per KV head
    //
    // one thread per logical token
    //
    // Example:
    //
    // num_tokens = 20
    // num_heads  = 2
    //
    // grid  = 2 blocks
    // block = 20 threads
    // ------------------------------------------------------------

    dim3 softmax_grid(
        num_kv_heads
    );

    dim3 softmax_block(
        num_tokens
    );

    // Two float arrays of num_tokens elements.
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

    return weights;
}