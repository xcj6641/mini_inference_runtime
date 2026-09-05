This result is useful, and it makes the materialization problem look even more dramatic. But there is one important caveat before we interpret it too strongly.

### What the result says

| Seq len | Materialize median | Attention median | Materialize / attention |
| ------: | -----------------: | ---------------: | ----------------------: |
|     128 |          171.04 ms |        0.1060 ms |                 ~1,614× |
|     512 |          659.01 ms |        0.1044 ms |                 ~6,310× |
|    1024 |         1315.49 ms |        0.1142 ms |                ~11,521× |
|    2048 |         2631.03 ms |        0.1121 ms |                ~23,465× |
|    4096 |         5250.39 ms |        0.1111 ms |                ~47,257× |

So for your current implementation:

```text
Paged KV
   │
   │ materialize token-by-token
   │ ~5250 ms @ seq=4096
   ▼
Contiguous KV
   │
   │ attention
   │ ~0.11 ms
   ▼
Output
```

This is excellent evidence for the architectural problem we're trying to demonstrate:

> Storing KV in pages is not enough. If normal attention requires reconstructing contiguous KV every decode step, an inefficient materialization path can completely dominate execution time.

And your materializer is especially expensive because it effectively does:

```python
for layer_idx in range(24):
    for token_idx in range(seq_len):
        key[...] = key_cache[...]
        value[...] = value_cache[...]
```

At 4096 tokens that's roughly:

```text
24 × 4096 × 2
≈ 196,608 tensor copy assignments
```

issued from Python.

That's why you're getting seconds to move only 48 MiB.

---

## But there's an important issue with the attention benchmark

Notice something strange:

```text
128     0.106 ms
512     0.104 ms
1024    0.114 ms
2048    0.112 ms
4096    0.111 ms
```

The attention latency is essentially flat.

You might expect attention to become more expensive as sequence length increases because:

```text
Q @ Kᵀ

[1 × D] @ [D × seq_len]
```

and:

```text
softmax(scores) @ V
```

both grow with `seq_len`.

But this isn't necessarily a bug.

Your workload is **tiny**:

```python
batch = 1
num_heads = 2
query_length = 1
head_dim = 64
```

Even at `seq_len=4096`, that's not much GPU compute. Fixed costs such as kernel-launch overhead and multiple PyTorch operations can dominate.

Your function isn't one fused attention kernel either:

```python
scores = torch.matmul(...)
scores = scores * scale
probabilities = torch.softmax(...)
output = torch.matmul(...)
```

so ~0.1 ms can largely reflect launch/framework overhead.

Therefore I would **not claim**:

> Attention cost is independent of sequence length.

Instead:

> For this small single-token decode workload, contiguous attention measured around 0.10–0.11 ms across the tested sequence lengths, suggesting fixed GPU/kernel-launch overhead dominates at this scale.

That's a much safer interpretation.

---

## One more comparison issue

There's an even more important distinction.

Your materialization benchmark materializes:

```text
all 24 layers
```

because:

```python
materialize_request_kv()
```

loops over:

```python
for layer_idx in range(self.num_layers):
```

But your attention benchmark uses:

```python
key, value = materialized[0]
```

So you're benchmarking attention for **only one layer**.

Therefore:

```text
5250 ms materialization
vs
0.111 ms attention
```

is not an apples-to-apples comparison of complete model operations.

A rough 24-layer attention number would not simply be exactly `24 × 0.111`, because execution behavior can differ, but conceptually we're currently comparing:

```text
materialize 24 layers
        vs
attention 1 layer
```

This is important enough that we should fix it before creating the final Day 19 plot.
