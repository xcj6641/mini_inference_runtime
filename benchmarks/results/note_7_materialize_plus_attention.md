Your measured materialize + attention latency is:

Seq len	KV size	Materialize only	24-layer attention only	Materialize + attention
128	1.5 MiB	~171.0 ms	~2.00 ms	175.7 ms
512	6 MiB	~659.0 ms	~2.00 ms	664.3 ms
1024	12 MiB	~1315.5 ms	~2.17 ms	1364.8 ms
2048	24 MiB	~2631.0 ms	~2.15 ms	2642.0 ms
4096	48 MiB	~5250.4 ms	~2.04 ms	5314.7 ms

There is some run-to-run noise—especially 1024 and 4096—so we should not interpret combined - materialize as the true attention latency. These experiments were run separately, whereas your direct attention benchmark already tells us attention itself is only ~2 ms.

The important result is much clearer:

The end-to-end path is overwhelmingly dominated by KV materialization, not attention computation.

For example, at 4096:

materialization       ~5250 ms
attention                ~2 ms
combined              ~5315 ms

That is exactly the architectural problem we wanted Day 19 to expose.

Why this result matters

Your materialize_request_kv() performs token-by-token gathering from physical blocks. In your current implementation, the work scales with layers × sequence length. The benchmark is therefore measuring the cost of converting paged KV back into the contiguous representation expected by ordinary attention.

And your results scale almost perfectly with KV size:

1.5 MiB  → ~0.17 s
6 MiB    → ~0.66 s
12 MiB   → ~1.32 s
24 MiB   → ~2.63 s
48 MiB   → ~5.25 s

That's an especially useful result because it explains why merely having a paged allocator is not enough:

PagedKVCache
     │
     │  Great: flexible physical allocation
     ↓
paged blocks
     │
     │  BAD: gather everything back
     ↓
contiguous KV
     │
     ↓
normal attention

The memory-management advantage exists, but your current attention interface forces you to pay a huge reconstruction cost before computation.

That's precisely what direct PagedAttention is supposed to remove:

PagedKVCache
     │
     │ block_table
     ↓
PagedAttention
     │
     ↓
output

NO intermediate contiguous KV