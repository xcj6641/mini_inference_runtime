# Day 21 Summary — Direct CUDA PagedAttention Benchmark

## 1. Goal

Day 21 answered one architectural question:

> What changes when we avoid `materialize_request_kv()` and let CUDA read K/V directly from paged storage using `block_table`?

We compared two paths using the same logical Q/K/V data.

### Path A: Materialized baseline

```text
PagedKVCache
    ↓
materialize_request_kv()
    ↓
contiguous K/V
    ↓
PyTorch contiguous attention
```

### Path B: Direct paged path

```text
PagedKVCache
    ↓
block_table
    ↓
custom CUDA PagedAttention
    ↓
attention output
```

The goal was not to make the educational CUDA kernel beat optimized PyTorch attention. It was to isolate the effect of removing KV materialization.

---

## 2. Fair benchmark configuration

We used:

```text
dtype            = FP16
batch_size       = 1
num_layers       = 1
num_kv_heads     = 2
head_dim          = 64
block_size        = 16
layer_idx         = 0
warmup_iters      = 5
benchmark_iters   = 20
sequence lengths  = 128, 512, 1024, 2048, 4096
```

We deliberately used `num_layers=1` because the current CUDA implementation computes only one layer. Materializing 24 layers and comparing it against one CUDA layer would not be fair.

The benchmark used shuffled physical block IDs rather than sequential placement:

```text
logical block
    ↓ block_table
shuffled physical block
```

This ensured the CUDA kernel genuinely exercised paged addressing.

---

## 3. Correctness before performance

Before timing each sequence length, we compared:

```text
direct CUDA PagedAttention output
≈
materialized contiguous attention output
```

The two implementations used:

```text
QKᵀ / sqrt(head_dim)
    ↓
softmax
    ↓
attention weights × V
```

The CUDA output was FP32 because QK accumulation and weighted-value accumulation used `float`.

The reference output was converted to FP32 for comparison:

```python
torch.testing.assert_close(
    cuda_output,
    reference_output.float(),
    rtol=2e-2,
    atol=2e-2,
)
```

Only configurations that passed correctness were benchmarked.

---

## 4. Supporting 4096 tokens

The original softmax launch used:

```cpp
dim3 softmax_block(
    num_tokens
);
```

This required one CUDA thread per token. It could not support 2048 or 4096 tokens because an A10G CUDA block supports at most 1024 threads.

We changed it to a fixed-size block:

```cpp
constexpr int SOFTMAX_THREADS = 256;

dim3 softmax_block(
    SOFTMAX_THREADS
);
```

The softmax kernel now uses grid-stride loops:

```cpp
for (
    int token_idx = thread_idx;
    token_idx < num_tokens;
    token_idx += blockDim.x
) {
    // Process token_idx
}
```

At 4096 tokens:

```text
256 threads × 16 tokens per thread
= 4096 tokens
```

The mathematical result did not change. We only changed how tokens were assigned to threads.

The maximum and exponential-sum scans remained serial intentionally. Day 21 was a correctness-and-architecture benchmark, not a production softmax optimization task.

---

## 5. Final CUDA-event benchmark results

| Sequence length |    KV size | Materialize | Contiguous attention | Materialized path | Direct CUDA | Speedup | Materialization share |
| --------------: | ---------: | ----------: | -------------------: | ----------------: | ----------: | ------: | --------------------: |
|             128 | 0.0625 MiB |    7.192 ms |             0.116 ms |          7.395 ms |    0.073 ms |  101.0× |                98.42% |
|             512 |   0.25 MiB |   28.617 ms |             0.120 ms |         28.602 ms |    0.190 ms |  150.2× |                99.58% |
|            1024 |    0.5 MiB |   54.298 ms |             0.111 ms |         54.590 ms |    0.350 ms |  155.9× |                99.80% |
|            2048 |    1.0 MiB |  108.929 ms |             0.113 ms |        115.502 ms |    0.675 ms |  171.0× |                99.90% |
|            4096 |    2.0 MiB |  218.332 ms |             0.115 ms |        218.899 ms |    1.319 ms |  166.0× |                99.95% |

The speedup was calculated using the separately measured end-to-end materialized path:

```python
speedup = (
    materialized_path_median_ms
    / direct_paged_median_ms
)
```

The materialization share was calculated using the component sum:

```python
component_sum_ms = (
    materialize_median_ms
    + contiguous_attention_median_ms
)

materialization_percent = (
    materialize_median_ms
    / component_sum_ms
    * 100
)
```

Using the component sum prevents impossible percentages above 100% caused by comparing medians from independent timing runs.

---

## 6. Main performance findings

### Materialization dominated the baseline

Materialization represented:

```text
128 tokens:  98.42%
4096 tokens: 99.95%
```

At 4096:

```text
materialization:       218.332 ms
contiguous attention:    0.115 ms
direct paged CUDA:        1.319 ms
```

Almost all latency in the baseline came from reconstructing contiguous K/V—not from attention itself.

### Both paths scaled approximately linearly

Materialization:

```text
1024:  54.298 ms
2048: 108.929 ms
4096: 218.332 ms
```

Direct CUDA:

```text
1024: 0.350 ms
2048: 0.675 ms
4096: 1.319 ms
```

For direct CUDA:

```text
1024 → 2048: 1.93× latency
2048 → 4096: 1.95× latency
```

This is expected because both paths process every cached token.

### Direct CUDA did not beat contiguous attention alone

At 4096:

```text
direct CUDA:          1.319 ms
contiguous attention: 0.115 ms
```

The naive CUDA implementation was approximately:

```text
1.319 / 0.115 ≈ 11.5×
```

slower than optimized contiguous PyTorch attention alone.

However, contiguous attention requires contiguous K/V. Including materialization made its end-to-end path dramatically slower:

```text
materialized path: 218.899 ms
direct paged path:   1.319 ms
```

Therefore:

> The improvement came from removing materialization, not from making attention computation itself faster.

---

## 7. CUDA-event versus wall-clock validation

We also measured:

* CUDA-event latency
* Caller-visible wall-clock latency

The representative second-run results were:

| Operation         | Sequence | CUDA event | Wall clock | Ratio |
| ----------------- | -------: | ---------: | ---------: | ----: |
| Materialize       |     4096 | 218.460 ms | 219.224 ms | 1.004 |
| Materialized path |     4096 | 219.205 ms | 219.462 ms | 1.001 |
| Direct paged      |     4096 |   1.318 ms |   1.322 ms | 1.003 |

The additional wall-clock overhead was approximately:

```text
Materialized path:
(219.462 - 219.205) / 219.205
≈ 0.12%

Direct paged:
(1.322 - 1.318) / 1.318
≈ 0.33%
```

This showed:

> There was no large hidden CPU-side cost outside the CUDA-event interval.

The materialization bottleneck was already present on the CUDA timeline.

The CUDA-event interval can contain:

* GPU indexing and copy operations
* Kernel execution
* Memory operations
* Stream idle gaps while more work is submitted

Therefore, CUDA-event latency is not necessarily pure kernel-computation time. Exact attribution would require Nsight Systems or another profiler.

The first validation run contained one 4096 materialized-path ratio of `1.089`, but it did not reproduce. The second run produced `1.001`, so the first result was treated as benchmark noise.

---

## 8. Why materialization was so expensive

The cache data remained on the GPU:

```text
paged GPU memory
    ↓
GPU indexing/copy operations
    ↓
contiguous GPU memory
```

There was no GPU-to-CPU-to-GPU KV transfer.

However, the educational `materialize_request_kv()` implementation likely performs many small indexing and copy operations rather than one optimized GPU gather.

That can cause:

* Many small CUDA operations
* Repeated kernel-launch overhead
* Poor operation granularity
* Stream gaps
* Inefficient memory access
* Repeated tensor allocation and copying

Therefore, the measured `101–171×` improvement applies to this runtime’s current materialization implementation. It must not be presented as a universal claim that all PagedAttention implementations are 100× faster.

---

## 9. Current CUDA limitations

The custom implementation launches three kernels:

```text
Kernel 1: paged QK dot products
Kernel 2: scaled softmax
Kernel 3: weighted-value accumulation
```

Known limitations include:

* Three separate kernel launches
* No kernel fusion
* Serial maximum scan in softmax
* Serial exponential-sum scan in softmax
* Each weighted-value thread loops over all tokens
* No warp-level reductions
* No vectorized memory loads
* No shared-memory K/V tiling
* No batching
* Only two KV heads
* Only one layer per call
* No GQA head mapping
* Fixed Day 20 tensor shapes

These are expected limitations of a correctness-first educational implementation.

---

## 10. Day 19 cross-validation

Day 19 measured approximately:

```text
1024 tokens
24 layers
materialization ≈ 1380 ms
```

Normalized per layer:

```text
1380 / 24 ≈ 57.5 ms
```

Day 21 measured:

```text
1024 tokens
1 layer
materialization ≈ 54.3 ms
```

The close agreement provides useful validation between the Day 19 and Day 21 benchmarks.

---

## 11. Final conclusion

> Direct CUDA PagedAttention successfully read K/V directly from paged cache storage using the block table, without reconstructing contiguous K/V tensors. Across sequence lengths from 128 to 4096, materialization accounted for 98.42–99.95% of the component-wise baseline latency. Direct paged attention reduced the current runtime’s measured end-to-end latency by approximately 101–171×. Both paths scaled approximately linearly with sequence length. The improvement came primarily from eliminating the expensive educational materialization implementation, not from outperforming optimized contiguous PyTorch attention. The CUDA kernel still contains serial reductions, unfused stages, and unoptimized memory access, demonstrating why production PagedAttention requires substantially more kernel engineering.

## Interview-ready version

> I implemented a correctness-first CUDA PagedAttention path that directly reads paged KV-cache blocks through a block table. I compared it with the runtime’s existing approach of first materializing contiguous K/V and then running PyTorch attention. From 128 to 4096 tokens, materialization represented 98.4–99.95% of the component-wise baseline and reached about 218 ms at 4096 tokens. Direct paged attention took about 1.32 ms at 4096 tokens and reduced the measured end-to-end latency by approximately 101–171× in this runtime. CUDA-event and wall-clock measurements were nearly identical, confirming that there was no large hidden CPU-side overhead outside the CUDA interval. I would not claim my naive kernel is faster than optimized PyTorch attention itself—the gain came from removing data-layout conversion. The remaining bottlenecks include serial softmax reductions, serial value accumulation, three unfused kernel launches, and no vectorized memory access.

Day 21 is complete. 
