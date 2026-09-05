Your all-layer benchmark is executing 24 calls to contiguous_attention() sequentially inside one timed operation, so the ~2 ms represents the total synthetic attention work across 24 layers.

1. Compare single-layer vs 24-layer

Earlier, your single-layer results were approximately:

seq_len	1 layer	24 layers	24-layer / 1-layer
128	0.106 ms	2.003 ms	~18.9×
512	0.104 ms	1.997 ms	~19.1×
1024	0.114 ms	2.167 ms	~19.0×
2048	0.112 ms	2.148 ms	~19.2×
4096	0.111 ms	2.043 ms	~18.4×

This is quite reasonable.

You might initially expect:

$$ 0.11\text{ ms} \times 24 \approx 2.64\text{ ms} $$

but you're getting ~2.0–2.17 ms.

That's not concerning. Tiny CUDA operations are particularly affected by launch overhead, GPU scheduling, caching, and measurement granularity. Measuring a group of 24 operations together amortizes some of those effects.

So the important observation is:

The 24-layer result scales roughly with the amount of attention work, and its magnitude is consistent with the single-layer benchmark.

That gives us considerably more confidence that the benchmark itself is behaving normally.

2. But the sequence-length result is interesting

Look at your medians:

128      2.003 ms
512      1.997 ms
1024     2.167 ms
2048     2.148 ms
4096     2.043 ms

Despite increasing sequence length 32×:

$$ 128 \rightarrow 4096 $$

latency basically stays around:

$$ 2.0\text{–}2.2\text{ ms} $$

This is the same phenomenon we saw in your single-layer experiment.

It does not mean attention complexity is independent of sequence length.

For decode attention, Q has only one query token:

$$ Q:[1,H,1,D] $$

while:

$$ K:[1,H,S,D] $$

so the main matmul is approximately:

$$ QK^T: (1\times D)(D\times S) $$

and computational work grows approximately linearly with \(S\).

But your workload is extremely small:

num_kv_heads = 2
head_dim     = 64
batch        = 1

Even at seq_len=4096, this isn't enough work to make the A10G's compute capacity the dominant factor. Kernel-launch/framework overhead and low GPU utilization can dominate.

That's actually an important systems lesson from this benchmark.

3. Compare this with materialization

Now we have a much more striking comparison.

Your latest materialization results were approximately:

seq_len    materialize      24-layer attention

128        171 ms            2.00 ms
512        659 ms            2.00 ms
1024       1315 ms           2.17 ms
2048       2631 ms           2.15 ms
4096       5250 ms           2.04 ms

At 4096 tokens:

$$ \frac{5250}{2.04}\approx2570 $$

Your Python/PyTorch materialization implementation is more than 2,000× slower than this synthetic 24-layer contiguous-attention computation.

And materialization scales almost perfectly linearly:

128       ~171 ms
512       ~659 ms
1024     ~1315 ms
2048     ~2631 ms
4096     ~5250 ms

while attention stays near 2 ms.

This strengthens the conclusion we reached earlier:

The problem is not GPU memory bandwidth itself. The reference materialize_request_kv() implementation is dominated by many small Python/PyTorch indexing/copy operations and kernel-launch overhead.

Your materialization benchmark really is repeatedly calling the cache's materialization operation and calculating effective bandwidth from the full KV size.

That is a result worth saving for the project documentation/interview discussion.

4. This connects directly to why real PagedAttention exists

Our reference architecture currently effectively does:

Paged KV
   │
   │ materialize
   ▼
Contiguous KV
   │
   ▼
Attention

Your benchmark says approximately:

Paged KV
   │
   │ ~5250 ms @ 4096
   ▼
Contiguous KV
   │
   │ ~2 ms attention
   ▼
Output

That's terrible if used as a real inference path.

Real PagedAttention instead wants:

               block_table
                    │
                    ▼
Q ────────► PagedAttention
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       KV block 7          KV block 21
          │                   │
          └─────────┬─────────┘
                    ▼
                  output

In other words:

Don't reconstruct a contiguous KV tensor at all. Let the attention operation consume paged KV directly.

And that's exactly the architectural motivation your benchmark is exposing experimentally.