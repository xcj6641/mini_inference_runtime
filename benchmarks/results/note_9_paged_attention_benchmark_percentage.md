The corrected `materialization_percent` is now valid: every value is below 100%, and it approaches 100% as sequence length grows.

| Seq. length | Materialize | Contiguous | Component sum | End-to-end baseline | Direct CUDA | Speedup | Materialization share |
| ----------: | ----------: | ---------: | ------------: | ------------------: | ----------: | ------: | --------------------: |
|         128 |    7.192 ms |   0.116 ms |      7.308 ms |            7.395 ms |    0.073 ms |  101.0× |                98.42% |
|         512 |   28.617 ms |   0.120 ms |     28.738 ms |           28.602 ms |    0.190 ms |  150.2× |                99.58% |
|        1024 |   54.298 ms |   0.111 ms |     54.409 ms |           54.590 ms |    0.350 ms |  155.9× |                99.80% |
|        2048 |  108.929 ms |   0.113 ms |    109.041 ms |          115.502 ms |    0.675 ms |  171.0× |                99.90% |
|        4096 |  218.332 ms |   0.115 ms |    218.446 ms |          218.899 ms |    1.319 ms |  166.0× |                99.95% |

## The percentage fix is correct

The new values match:

```python
materialization_percent = (
    materialize_median_ms
    / component_sum_ms
    * 100
)
```

For example, at 4096:

```text
218.3316 / 218.4463 × 100
= 99.9475%
```

That accurately communicates that nearly all separately measured baseline work is materialization.

## Direct CUDA scaling is very consistent

```text
128:   0.073 ms
512:   0.190 ms
1024:  0.350 ms
2048:  0.675 ms
4096:  1.319 ms
```

For longer sequences:

```text
1024 → 2048: 1.93× latency
2048 → 4096: 1.95× latency
```

This is convincing linear scaling. The 4096 direct-CUDA measurement is especially stable:

```text
mean:   1.3189 ms
median: 1.3189 ms
min:    1.3179 ms
max:    1.3230 ms
```

That is an excellent benchmark sample.

## Materialization also scales linearly

```text
512:    28.62 ms
1024:   54.30 ms
2048:  108.93 ms
4096:  218.33 ms
```

From 1024 onward, doubling the sequence length almost exactly doubles materialization latency.

Meanwhile, contiguous attention remains around `0.11–0.12 ms`, making its contribution increasingly negligible:

```text
128:  materialization = 98.42%
4096: materialization = 99.95%
```

## One result deserves another run

The 2048 end-to-end baseline has higher variation:

```text
median: 115.502 ms
mean:   113.726 ms
min:    108.042 ms
max:    116.521 ms
```

Its separately measured component sum is only:

```text
109.041 ms
```

This makes the reported speedup jump to `171×`, higher than both 1024 and 4096. It is not invalid—the speedup correctly uses the measured end-to-end path—but it appears affected by run-to-run timing variation.

I recommend rerunning only 2048 once or twice. You do not need to discard this result. For the final report, avoid focusing on the exact `171×` peak and summarize the range instead:

> Direct paged attention was approximately 100–170× faster than the current materialized execution path.

## Final Day 21 finding

> Direct CUDA PagedAttention eliminated explicit KV materialization while preserving correct paged addressing through the block table. Materialization accounted for 98.4%–99.95% of the component-wise baseline latency and scaled linearly from 7.19 ms at 128 tokens to 218.33 ms at 4096 tokens. Direct paged attention also scaled approximately linearly, from 0.073 ms to 1.319 ms, but with a much smaller constant factor. It reduced end-to-end latency by approximately 100–171× compared with the current educational materialization path. This improvement primarily reflects removal of the runtime’s expensive materialization implementation; it does not mean the naive CUDA attention kernel is faster than optimized contiguous attention alone.

