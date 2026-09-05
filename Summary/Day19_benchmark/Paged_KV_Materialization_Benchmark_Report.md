# Day 19 — Paged KV Materialization Benchmark Report

## 1. Research Question

The runtime stores KV cache using paged physical blocks. However, the current PyTorch attention path expects contiguous K/V tensors, so each request's paged KV must first be reconstructed using `PagedKVCache.materialize_request_kv()`.

This benchmark investigates:

> **How much overhead does KV materialization introduce, and how does that overhead scale with context length?**

The experiment also establishes a baseline for later comparison with direct CUDA PagedAttention, which will access paged KV without first reconstructing contiguous K/V.

---

## 2. Experimental Setup

The benchmark uses the following KV-cache configuration:

| Parameter           |                      Value |
| ------------------- | -------------------------: |
| Number of layers    |                         24 |
| KV heads            |                          2 |
| Head dimension      |                         64 |
| Block size          |                  16 tokens |
| Data type           |                       FP16 |
| Warmup iterations   |                          5 |
| Measured iterations |                         20 |
| Sequence lengths    | 128, 512, 1024, 2048, 4096 |

CUDA events are used for GPU timing, with warmup iterations executed before measurement and `torch.cuda.synchronize()` used to ensure GPU work has completed before timing results are consumed.

Materialization correctness is checked by comparing reconstructed K/V tensors against the original contiguous K/V tensors written into the paged cache.

---

# 3. Materialization Results

| Sequence Length | KV Size | Median Materialization | Effective Bandwidth |
| --------------: | ------: | ---------------------: | ------------------: |
|             128 | 1.5 MiB |               171.0 ms |         0.0092 GB/s |
|             512 |   6 MiB |               659.0 ms |         0.0095 GB/s |
|            1024 |  12 MiB |              1315.5 ms |         0.0096 GB/s |
|            2048 |  24 MiB |              2631.0 ms |         0.0096 GB/s |
|            4096 |  48 MiB |              5250.4 ms |         0.0096 GB/s |

## Finding 1 — Materialization latency scales almost linearly with context length

The strongest result is the relationship between sequence length and materialization latency.

Increasing the context from 512 to 1024 tokens approximately doubles latency:

```text
512 tokens   →   ~659 ms
1024 tokens  →  ~1315 ms
```

Doubling again:

```text
1024 tokens  →  ~1315 ms
2048 tokens  →  ~2631 ms
```

and again:

```text
2048 tokens  →  ~2631 ms
4096 tokens  →  ~5250 ms
```

Therefore:

> **The current materialization implementation has approximately O(sequence length) runtime behavior.**

This is expected because every cached token must be copied from its paged physical location into the newly allocated contiguous K/V tensors.

---

## Finding 2 — Effective bandwidth remains approximately constant

Across larger sequence lengths, effective bandwidth stays close to:

```text
~0.0095 GB/s
```

For example:

```text
512 tokens    0.00955 GB/s
1024 tokens   0.00957 GB/s
2048 tokens   0.00957 GB/s
4096 tokens   0.00959 GB/s
```

This explains the near-linear latency scaling.

As KV size increases:

```text
bytes ↑  →  latency ↑
```

while effective throughput remains nearly unchanged.

The benchmark is therefore behaving consistently rather than producing arbitrary latency spikes.

However, this number should **not** be interpreted as GPU memory bandwidth.

The implementation performs token-wise tensor assignments from Python:

```text
for layer:
    for token:
        locate physical block
        copy K[token]
        copy V[token]
```

Consequently, the benchmark measures the effective throughput of this entire PyTorch/Python materialization mechanism, including many small tensor operations, rather than the hardware's raw device-to-device copy bandwidth.

---

# 4. Contiguous Attention Baseline

A decode-style contiguous-attention benchmark was also measured across all 24 layers.

| Sequence Length | Median Attention Latency |
| --------------: | -----------------------: |
|             128 |                 2.003 ms |
|             512 |                 1.997 ms |
|            1024 |                 2.167 ms |
|            2048 |                 2.148 ms |
|            4096 |                 2.043 ms |

## Finding 3 — Materialization dominates the current decode-attention path

The difference in scale is extremely large.

At 4096 tokens:

```text
materialization      ≈ 5250 ms
attention             ≈    2 ms
```

Even at 128 tokens:

```text
materialization      ≈ 171 ms
attention             ≈   2 ms
```

Therefore, in this reference implementation:

> **Reconstructing contiguous KV is substantially more expensive than the synthetic contiguous-attention computation that consumes it.**

This means optimizing only the attention computation while retaining the current materialization path would not address the dominant measured overhead.

The architectural boundary itself needs to change.

---

# 5. Materialize + Attention Results

The combined path was also benchmarked:

```text
PagedKVCache
      ↓
materialize_request_kv()
      ↓
contiguous KV
      ↓
24-layer attention
```

Representative results:

| Sequence Length | KV Size | Combined Median |
| --------------: | ------: | --------------: |
|             128 | 1.5 MiB |        175.7 ms |
|             512 |   6 MiB |        664.3 ms |
|            1024 |  12 MiB |       1364.8 ms |
|            2048 |  24 MiB |       2642.0 ms |
|            4096 |  48 MiB |       5314.7 ms |

## Finding 4 — End-to-end latency inherits materialization's linear scaling

The combined benchmark follows almost the same scaling pattern as materialization alone:

```text
128       ~176 ms
512       ~664 ms
1024     ~1365 ms
2048     ~2642 ms
4096     ~5315 ms
```

This is consistent with the fact that materialization is orders of magnitude larger than the approximately 2 ms attention baseline.

The individual benchmark medians were obtained from separate runs, so their values should not be expected to add exactly. Nevertheless, their relative scale clearly shows which component dominates.

---

# 6. Why the Materialization Cost Is High

The benchmark does **not** demonstrate that paged KV storage itself is inherently slow.

The problem is the implementation used to convert paged storage back into contiguous storage.

Current materialization effectively performs:

```text
for each layer:
    allocate contiguous K
    allocate contiguous V

    for each token:
        logical token
             ↓
        block_table lookup
             ↓
        physical block + slot
             ↓
        PyTorch tensor assignment for K
        PyTorch tensor assignment for V
```

For 4096 tokens and 24 layers, this produces a very large number of fine-grained operations.

The measured ~0.0095 GB/s therefore indicates:

> **The Python/PyTorch token-wise gather implementation is inefficient as a GPU memory movement mechanism.**

It does not indicate that the GPU itself is capable of only ~0.0095 GB/s.

---

# 7. Reference PagedAttention Finding

A reference PagedAttention implementation was also created that follows the `block_table` directly instead of first calling `materialize_request_kv()`.

Conceptually:

```text
                block_table
                    ↓
Query ──────→ Paged KV Cache
                    ↓
             read required K/V
                    ↓
                attention
```

The initial PyTorch reference implementation was not faster. For example, at sequence length 128 it required roughly 295 ms.

This is expected because it still performs fine-grained paged accesses through Python/PyTorch operations.

## Finding 5 — Removing materialization architecturally is not sufficient for performance

This is an important result.

The experiment demonstrates two separate problems:

```text
Problem 1: Architecture

Paged KV
   ↓
materialize
   ↓
attention
```

requires unnecessary contiguous reconstruction.

But simply changing it to:

```text
Paged KV
   ↓
Python token-wise reads
   ↓
attention
```

does not solve:

```text
Problem 2: Implementation efficiency
```

Direct paged access must itself be implemented efficiently.

This motivates moving the direct paged access and attention computation into a GPU kernel.

---

# 8. Main Experimental Conclusion

Day 19 produced three main conclusions.

### Conclusion A — The current materialization path is a major bottleneck

Materialization latency scales approximately linearly with KV size and reaches approximately 5.25 seconds for a 4096-token, 48-MiB KV cache in this reference implementation.

### Conclusion B — The bottleneck is primarily an implementation artifact, not evidence that paged memory is inherently slow

The current implementation gathers KV one token at a time using Python-driven PyTorch tensor operations.

Its approximately 0.0095 GB/s effective throughput should therefore be interpreted as the throughput of the reference gather implementation, not GPU DRAM bandwidth.

### Conclusion C — Direct PagedAttention is the correct next architectural experiment

The current path is:

```text
Paged KV
   ↓
materialize contiguous KV
   ↓
optimized attention
```

The desired path is:

```text
Paged KV
   ↓
CUDA kernel directly follows block_table
   ↓
attention
```

The benchmark therefore provides a concrete motivation for the CUDA PagedAttention work:

> **Avoid reconstructing the entire contiguous KV cache before every attention operation and instead consume the paged representation directly inside the attention kernel.**

---

# 9. What the Results Do NOT Prove

The benchmark should not be used to claim that:

* paged KV storage is inherently slow;
* GPU memory bandwidth is approximately 0.0095 GB/s;
* production systems spend seconds materializing a 4096-token KV cache;
* CUDA PagedAttention will necessarily be faster than PyTorch attention;
* the future CUDA kernel will eliminate all context-length-dependent cost.

In particular, attention itself must still process the cached context. Direct PagedAttention removes the **separate KV reconstruction step**; it does not eliminate the computational and memory cost of attending over previous tokens.

---

# 10. Day 20 Hypothesis

Day 19 gives us a concrete hypothesis to test next:

> **If block-table translation and paged K/V access are performed directly inside a CUDA attention kernel, we can eliminate the separate materialization stage while preserving attention correctness.**

Day 20 therefore does not initially ask:

> "Can our CUDA kernel beat PyTorch?"

The first question is:

> **Can CUDA directly consume the paged KV layout and produce the same output without `materialize_request_kv()`?**

Once correctness is established, later benchmarks can compare:

```text
Baseline:

Paged KV
   ↓
materialize
   ↓
PyTorch attention


vs.


Direct:

Paged KV
   ↓
CUDA PagedAttention
```

That comparison will determine whether eliminating materialization produces a useful end-to-end performance improvement and where the remaining bottlenecks lie.
