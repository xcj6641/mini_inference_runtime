# Day 18 Summary — Reference PagedAttention

Day 18 was important because we crossed from **“I have paged KV storage”** to **“I understand how attention can consume paged KV storage.”**

The main goal was:

> Implement a correctness-first PagedAttention reference that understands `block_table` addressing, and prove that paged physical storage produces the same attention result as logically contiguous KV.

## 1. Where we started

Before Day 18, our runtime already had:

```text
Request
   ↓
block_table
   ↓
PagedKVCache
   ↓
physical KV blocks
```

For example:

```text
block_size = 2
block_table = [2, 0, 1]

Physical memory:

block 0: K2 K3
block 1: K4 XX
block 2: K0 K1
```

But our normal model attention cannot directly understand this layout.

So the runtime has to do:

```text
PagedKVCache
      ↓
materialize()
      ↓
[K0 K1 K2 K3 K4]
      ↓
normal attention
```

This means we had implemented **paged storage**, but not **paged computation**.

---

## 2. First, we built the contiguous attention oracle

We deliberately started with the simplest attention:

```text
Q: [head_dim]

K: [seq_len, head_dim]
V: [seq_len, head_dim]
```

and implemented:

```text
scores = QKᵀ / sqrt(head_dim)

weights = softmax(scores)

output = weights V
```

This became our correctness oracle:

```python
contiguous_attention_reference(...)
```

That separation was useful because later tests don't need to manually calculate complicated attention outputs.

Instead:

```text
PagedAttention output
        ≈
contiguous_attention_reference output
```

---

## 3. We implemented paged address translation

The central operation was:

```text
token_index
     ↓
logical_block = token_index // block_size

block_offset = token_index % block_size
     ↓
physical_block = block_table[logical_block]
     ↓
PagedKVCache[physical_block][block_offset]
```

For:

```text
block_size = 2
block_table = [2, 0, 1]
```

we get:

```text
token 0 → logical block 0 → physical block 2 → slot 0
token 1 → logical block 0 → physical block 2 → slot 1

token 2 → logical block 1 → physical block 0 → slot 0
token 3 → logical block 1 → physical block 0 → slot 1

token 4 → logical block 2 → physical block 1 → slot 0
```

This is probably the single most important concept from Day 18.

### Core invariant

> Logical KV order does not have to equal physical KV order.

Attention correctness depends on the **logical mapping through `block_table`**, not physical contiguity.

---

## 4. We deliberately scrambled physical blocks

We didn't test with:

```text
block_table = [0, 1, 2]
```

because that would be a weak test.

Instead we used things like:

```text
block_table = [2, 0, 1]
```

That matters because otherwise buggy code could accidentally assume:

```text
logical block == physical block
```

and still pass.

This test design made our correctness claim substantially stronger.

---

## 5. We handled partial final blocks

For:

```text
seq_len = 5
block_size = 2
```

we need three blocks:

```text
block 0: token token
block 1: token token
block 2: token UNUSED
```

So:

```text
allocated capacity = 6
valid KV tokens    = 5
```

We deliberately filled unused storage with values such as:

```text
-999
```

and verified that it did not affect attention.

This reinforced an important runtime concept we've seen before:

> Physical KV capacity and logical KV length are different things.

Attention must use `kv_tokens` / `seq_len`, not allocated capacity.

---

# 6. We extended from single-head to multi-head attention

The reference representation evolved to:

```text
query:
[num_attention_heads, head_dim]

KV:
[seq_len, num_kv_heads, head_dim]
```

This allowed us to model actual transformer attention more closely.

For ordinary MHA:

```text
num_attention_heads = num_kv_heads
```

so:

```text
Q0 → KV0
Q1 → KV1
Q2 → KV2
Q3 → KV3
```

---

# 7. Then we implemented GQA

This was especially important because the model we're using has grouped-query attention.

For example:

```text
num_attention_heads = 4
num_kv_heads = 2
```

then:

```text
Q0 ─┐
    ├→ KV0
Q1 ─┘

Q2 ─┐
    ├→ KV1
Q3 ─┘
```

The mapping is:

```python
num_queries_per_kv_head = (
    num_attention_heads // num_kv_heads
)

kv_head_index = (
    attention_head_index
    // num_queries_per_kv_head
)
```

And we verified this against contiguous attention.

This also helped clarify an important memory optimization:

```text
Query heads > KV heads
```

means we don't need a separate K/V cache for every query head.

---

# 8. We integrated the actual `PagedKVCache`

Initially, the reference implementation used simplified standalone tensors.

Then we crossed the important runtime boundary and used the actual cache layout:

```text
[num_layers,
 num_blocks,
 num_kv_heads,
 block_size,
 head_dim]
```

We implemented the real-cache lookup:

```text
layer
  ↓
physical_block
  ↓
kv_head
  ↓
block_offset
  ↓
head_dim
```

and verified that logical tokens correctly retrieve:

```text
[num_kv_heads, head_dim]
```

from the real cache.

This meant Day 18 stopped being just an isolated attention exercise and became compatible with the runtime architecture.

---

# 9. We built `paged_attention_reference_from_cache`

The final conceptual path became:

```text
Q
+
block_table
+
real PagedKVCache
        ↓
logical-token lookup
        ↓
GQA head mapping
        ↓
scaled dot-product attention
        ↓
output
```

And we compared:

```text
Real PagedKVCache
      ↓
reference PagedAttention
      ↓
actual

       ≈

logical contiguous KV
      ↓
contiguous attention
      ↓
expected
```

That was our strongest deterministic integration test.

---

# 10. Finally, we added randomized regression tests

We varied:

```text
seq_len
block_size
num_attention_heads
num_kv_heads
head_dim
physical block placement
```

including:

```text
MHA:
Q heads = KV heads

GQA:
Q heads > KV heads

MQA-like:
many Q heads → one KV head
```

We also randomized the physical block permutation.

That gave us confidence that the implementation wasn't only correct for one carefully constructed example.

---

# 11. The most important limitation we discovered

This came from your question near the end of Day 18.

Our Python reference currently does:

```python
for token_index in range(seq_len):
    key, value = read_paged_kv_token_from_cache(...)
    keys.append(key)
    values.append(value)

key = torch.stack(keys)
value = torch.stack(values)
```

So it still effectively does:

```text
PagedKVCache
     ↓
page-table lookup
     ↓
collect all logical K/V
     ↓
torch.stack()
     ↓
temporary contiguous K/V
     ↓
attention
```

You correctly noticed:

> “Isn't that basically still fetching the KV and reconstructing it?”

**Yes.**

That's the crucial distinction we arrived at.

Our implementation is:

> **Reference PagedAttention — a correctness model, not a performance implementation.**

Its job is to prove the addressing semantics.

---

# 12. Reference vs runtime vs real PagedAttention

This is probably the most important interview takeaway from Day 18.

### Current runtime

```text
PagedKVCache
     ↓
materialize()
     ↓
full contiguous KV
     ↓
existing attention
```

Attention itself knows nothing about paging.

### Python reference

```text
PagedKVCache
     ↓
attention code understands block_table
     ↓
token-by-token page lookup
     ↓
torch.stack()
     ↓
attention
```

The attention implementation understands paged addressing, but still reconstructs temporary K/V.

### Real CUDA PagedAttention

```text
PagedKVCache
      ↓
CUDA kernel
      │
      ├─ block_table lookup
      │
      ├─ load K/V tile
      │
      ├─ Q · K
      │
      ├─ update softmax
      │
      ├─ accumulate V
      │
      └─ next block
      ↓
output
```

There should be no:

```python
torch.stack(all_KV)
```

and no:

```python
materialize()
```

of the complete logical sequence.

That's the optimization we're going after next.

---

# 13. The deeper architectural lesson

Before Day 18, these two concepts were easy to conflate:

```text
Paged KV Cache
      ≠
PagedAttention
```

Now we can distinguish them clearly:

```text
Paged KV Cache
=
how KV is STORED


PagedAttention
=
how attention CONSUMES
that storage
```

Our runtime currently has:

```text
paged storage
+
contiguous computation
```

with `materialize()` acting as the bridge.

Our target architecture is:

```text
paged storage
+
paged computation
```

where attention consumes the page table directly.

---

# 14. What Day 18 accomplished

I would summarize Day 18 in one sentence as:

> **Implemented a PyTorch correctness reference for PagedAttention that resolves logical KV tokens through a block table over non-contiguous physical KV blocks, supports partial blocks and GQA, integrates with the runtime's real `PagedKVCache`, and matches contiguous attention across deterministic and randomized tests.**

But equally important, we discovered the limitation:

> **The Python reference still uses `torch.stack()` to reconstruct temporary K/V, so it validates PagedAttention addressing semantics but does not model the optimized execution strategy of a real CUDA PagedAttention kernel.**


---

## Day 18 → Day 19

Day 18 answered:

> **Can attention be made logically correct when KV is stored in non-contiguous physical blocks?**

**Yes.**

Day 19 now asks:

> **How much are we paying because our current runtime still has to materialize those blocks into contiguous KV?**

Then the CUDA work asks:

> **Can we eliminate that materialization by consuming paged KV directly inside the kernel?**

So the remaining progression is very coherent:

```text
Day 18
Correctness
"Can it work?"

       ↓

Day 19
Measurement
"What does materialization cost?"

       ↓

Day 20+
CUDA
"Can we remove that cost?"

       ↓

Benchmark + profiling
"Did it actually help, and why?"
```
