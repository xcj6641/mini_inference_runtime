This is an excellent result because the scaling is extremely clear. Our hypothesis is strongly supported.

I would **definitely preserve both runs**. This is important Day 19 evidence for documentation and interviews.

### 1. Look at the scaling

Using your second run:

| Seq len | KV size |    Median |  Latency/token | Effective BW |
| ------: | ------: | --------: | -------------: | -----------: |
|     128 | 1.5 MiB |  166.9 ms | 1.304 ms/token | 0.00942 GB/s |
|     512 |   6 MiB |  665.8 ms | 1.300 ms/token | 0.00945 GB/s |
|    1024 |  12 MiB | 1329.0 ms | 1.298 ms/token | 0.00947 GB/s |
|    2048 |  24 MiB | 2656.4 ms | 1.297 ms/token | 0.00947 GB/s |
|    4096 |  48 MiB | 5306.2 ms | 1.295 ms/token | 0.00949 GB/s |

That's remarkably close to perfectly linear:

```text
128  →  167 ms
512  →  666 ms       ~4× tokens  → ~4× latency
1024 → 1329 ms       ~8× tokens  → ~8× latency
2048 → 2656 ms       ~16× tokens → ~16× latency
4096 → 5306 ms       ~32× tokens → ~32× latency
```

Or approximately:

$$
T_{\text{materialize}} \approx 1.30 \text{ ms} \times L
$$

where `L` is sequence length.

### 2. Effective bandwidth tells the same story

Look at your second run:

```text
0.00942
0.00945
0.00947
0.00947
0.00949 GB/s
```

It is basically constant at:

> **~0.0095 GB/s ≈ 9.5 MB/s effective payload bandwidth**

That consistency is useful.

The amount of KV data increases linearly with sequence length, and latency increases almost exactly linearly too, so effective throughput stays constant.

But again, don't say:

> "My A10G memory bandwidth is 9.5 MB/s."

Instead:

> **"My token-wise PyTorch materialization implementation achieves only ~9.5 MB/s effective KV payload throughput."**

That's a very different statement.

### 3. Why is it so slow?

Now connect the benchmark back to your code:

```python
for layer_idx in range(self.num_layers):

    for token_idx in range(num_tokens):

        physical_block_id, slot_idx = (
            self.get_physical_location(
                block_table,
                token_idx,
            )
        )

        key[...] = self.key_cache[...]
        value[...] = self.value_cache[...]
```

At `seq_len=4096`:

```text
24 layers × 4096 tokens
= 98,304 iterations
```

And every iteration performs separate K and V indexed tensor operations:

```text
≈ 196,608 fine-grained tensor assignments/accesses
```

to materialize only:

```text
48 MiB
```

So we're not giving the GPU something like:

```text
"Copy this giant 48 MiB contiguous region."
```

We're effectively giving it a huge number of tiny operations.

That is exactly the kind of workload GPUs don't want.

### 4. The two runs also give us useful reproducibility evidence

Run 1 versus Run 2:

|  Seq |  Run 1 |  Run 2 | Difference |
| ---: | -----: | -----: | ---------: |
|  128 |  175.4 |  166.9 |      ~5.1% |
|  512 |  698.3 |  665.8 |      ~4.9% |
| 1024 | 1380.4 | 1329.0 |      ~3.9% |
| 2048 | 2773.4 | 2656.4 |      ~4.4% |
| 4096 | 5352.9 | 5306.2 |      ~0.9% |

The absolute numbers vary somewhat between runs, which is normal enough for this kind of benchmark environment.

But the **scaling behavior does not change**.

Both runs show approximately:

```text
latency ∝ sequence length
```

That's actually more important to our current investigation than whether 128 tokens takes exactly 167 or 175 ms.

---

## 5. What have we actually proved?

Be careful with this distinction.

We have **not** proved:

> Paged KV cache is slow.

And we have **not** proved:

> PagedAttention is slow.

We've demonstrated:

> **Our current runtime pays a large, linearly growing cost when converting paged KV storage back into contiguous K/V using a token-wise PyTorch reference implementation.**

That's much more precise.

And this explains why the architectural boundary matters:

```text
Our current runtime

Paged KV Cache
      ↓
materialize_request_kv()
      ↓
Contiguous KV
      ↓
normal attention
```

The longer the sequence becomes, the more KV we reconstruct.

A real PagedAttention path instead aims for:

```text
Paged KV Cache
       ↓
 block_table
       ↓
PagedAttention
       ↓
attention output
```

The attention implementation understands the paged layout and accesses the required blocks directly.

Therefore there is no separate:

```text
paged → contiguous
```

materialization stage before attention.

That is the architectural motivation we wanted Day 19 to make concrete.

## 6. This is already a strong interview explanation

You can now explain the project progression naturally:

> "I first implemented paged KV storage, but my initial model runner still expected contiguous K/V tensors. So I added a reference materialization path to bridge the two representations. I then benchmarked that path across sequence lengths from 128 to 4096 tokens."

Then give the result:

> "Materialization grew almost perfectly linearly, from roughly 167 ms at 128 tokens to 5.3 seconds at 4096 tokens. Normalized latency stayed around 1.3 ms per token, and effective KV payload throughput was only around 9.5 MB/s."

Then the important interpretation:

> "This wasn't a GPU bandwidth limitation. My reference implementation performed token-wise indexed PyTorch operations across 24 layers, creating enormous fine-grained dispatch overhead. At 4096 tokens it executes roughly 98,000 layer-token iterations."

Finally:

> "That experiment helped me understand why production PagedAttention doesn't materialize paged KV back into a contiguous cache before attention. The attention kernel consumes the paged representation directly using the block table."

That's much stronger than simply saying:

> "I know PagedAttention saves memory."

You now have an experiment behind the explanation.

### One important caveat

Don't put **"5.3 seconds materialization overhead"** on your resume by itself. Someone could reasonably respond, "Why is your implementation catastrophically slow?"

If we ever use the number publicly, label it clearly as a:

> **token-wise PyTorch reference baseline**

The value of this experiment is the architectural insight and before/after comparison we're building toward—not pretending this baseline is production performance.

---

I think **we are done with the materialization scaling baseline**. Don't spend time collecting a third or fourth run.

The next useful step is to move toward the comparison that gives this baseline meaning: **reference PagedAttention/direct paged KV access versus the materialization-based path**. That's where Day 19 becomes a complete engineering story rather than just "I benchmarked some slow Python code."
