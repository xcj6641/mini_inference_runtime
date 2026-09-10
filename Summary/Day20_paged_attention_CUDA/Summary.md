Day 20 — Minimal CUDA PagedAttention
1. Goal

Implement the smallest CUDA PagedAttention path that proves we can perform decode attention by accessing paged KV storage directly through block_table, without:

materialize_request_kv()

Frozen scope:

one request
one decode query token
one layer per call
FP16 Q/K/V storage
2 KV heads
head_dim = 64
block_size = 16
direct paged K/V access
no GQA
no optimization/fusion

The KV cache layout is:

[num_layers,
 num_blocks,
 num_kv_heads,
 block_size,
 head_dim]
2. Final architecture

We implemented the attention calculation as three CUDA kernels:

                    Query
                      │
                      ▼
Paged K ───────→ Kernel 1: Q·K
   ↑                  │
block_table           ▼
                  raw scores
                      │
                      ▼
              Kernel 2: scale
                  + softmax
                      │
                      ▼
              attention weights
                      │
Paged V ──────────────┤
   ↑                  ▼
block_table    Kernel 3: weighted V
                      │
                      ▼
             attention output
                [1, 2, 64]

The important property is:

K and V are never reconstructed into contiguous KV tensors by the CUDA implementation.

Instead, every required K/V value is found through paged addressing.

3. Paged addressing

For logical token t:

logical_block = token_idx / block_size;
slot = token_idx % block_size;

physical_block =
    block_table[logical_block];

Then K/V is accessed as:

cache[
    layer_idx,
    physical_block,
    kv_head,
    slot,
    dim
]

This separated two concepts that were initially easy to confuse:

CUDA thread block
    ≠
KV cache block

A KV block is a memory-management concept.

A CUDA thread block is a GPU execution/scheduling concept.

block_table maps logical KV blocks to physical KV blocks; the CUDA grid determines which GPU threads perform the work.

4. Kernel 1 — Paged Q·K

Launch configuration:

grid  = (num_tokens, num_kv_heads)
block = (head_dim)

Therefore:

One CUDA thread block handles one (token, head) pair.

For example:

CUDA block (token=15, head=0)

thread 0  → Q[0]  × K[token15, head0, 0]
thread 1  → Q[1]  × K[token15, head0, 1]
...
thread 63 → Q[63] × K[token15, head0, 63]

The 64 threads then perform a shared-memory reduction:

64 partial products
       ↓
64 → 32 → 16 → 8 → 4 → 2 → 1
       ↓
Q · K

Output:

scores[num_tokens, num_kv_heads]

This was our first real CUDA attention computation.

5. Kernel 2 — Scaling + softmax

Scaled dot-product attention requires:

$$ S_t=\frac{QK_t^T}{\sqrt{d}} $$

followed by:

$$ P_t=\operatorname{softmax}(S)_t $$

The important realization was that softmax has a different parallelization requirement from Q·K.

QK needs communication across:

head_dim

while softmax needs communication across:

tokens

Therefore Kernel 2 uses:

grid  = num_kv_heads
block = num_tokens

One CUDA block handles one head:

head 0:

token0  ─┐
token1   │
token2   ├── softmax
...      │
token19 ─┘

We use numerically stable softmax:

$$ \frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}} $$

The implementation is intentionally naive; max and sum reduction aren't optimized.

6. Kernel 3 — Weighted paged V

Finally:

$$ O_{h,d} = \sum_t P_{t,h}V_{t,h,d} $$

We use:

grid  = num_kv_heads
block = head_dim

Therefore:

One thread owns one final (head, dim) output value.

Example:

CUDA block 0 → head 0

thread 7:

output[0,0,7]
=
weight[0,0]  × V[token0, head0, 7]
+
weight[1,0]  × V[token1, head0, 7]
+
...
+
weight[19,0] × V[token19, head0, 7]

Each thread loops through the tokens and follows block_table to read V directly.

This avoids races because only one thread owns each output element.

We discussed an alternative:

grid = (num_tokens, num_heads)

which provides more token-level parallelism, but then many CUDA blocks contribute to the same output element. That requires atomics or an additional cross-token reduction.

For Day 20, we intentionally chose the simpler correctness-first design.

7. FP16 vs FP32

Q/K/V are stored as FP16:

PagedKVCache → FP16
Query        → FP16

but our calculations use FP32:

FP16 Q/K/V
     ↓
convert to float
     ↓
FP32 QK accumulation
     ↓
FP32 softmax
     ↓
FP32 weighted-V accumulation
     ↓
FP32 output

Key lesson:

Storage dtype and computation/accumulation dtype do not have to be the same.

FP16 reduces memory usage, while FP32 accumulation improves numerical accuracy for our correctness-first implementation.

We also used:

key_cache.options().dtype(torch::kFloat32)

to create outputs on the same device/configuration as the cache while overriding the dtype.

8. Correctness tests

The final parametrized test covers:

16 tokens → one complete block

20 tokens → complete block + partial block

32 tokens → two complete blocks

scrambled block_table
    e.g. [3, 1]

CUDA output is compared against a PyTorch reference implementing:

materialize logical K/V in test only
        ↓
Q·K / sqrt(d)
        ↓
softmax(tokens)
        ↓
weights × V
        ↓
reference output

All cases passed.

The scrambled-block tests are particularly important because they demonstrate that the CUDA implementation actually follows:

logical token
     ↓
logical block
     ↓
block_table
     ↓
physical block

rather than accidentally assuming contiguous physical storage.

9. Day 20 result

Success criterion achieved:

Built a one-layer FP16-input CUDA PagedAttention implementation that directly consumes paged K/V through block_table and produces numerically correct decode-attention output without calling materialize_request_kv().

Current limitations are intentional:

batch/request = 1
query length = 1
2 KV heads
head_dim = 64
one layer per invocation
num_tokens limited by naive softmax layout
no GQA
no kernel fusion
naive reductions
naive weighted-V token loop
no performance optimization

Those are now optimization/generalization problems, rather than gaps in understanding the basic PagedAttention mechanism.