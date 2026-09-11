The result is clear: except for one 4096 materialized-path measurement, CUDA-event and wall-clock latency are almost identical.

This means there is no large hidden CPU-side latency. The materialization bottleneck is already visible within the CUDA-event interval.

## Results

| Operation            | Seq. length | CUDA event | Wall clock | Ratio |
| -------------------- | ----------: | ---------: | ---------: | ----: |
| Materialize          |         128 |   7.213 ms |   7.169 ms | 0.994 |
| Materialize          |        1024 |  54.171 ms |  54.328 ms | 1.003 |
| Materialize          |        4096 | 215.644 ms | 216.044 ms | 1.002 |
| Contiguous attention |         128 |   0.119 ms |   0.120 ms | 1.005 |
| Contiguous attention |        1024 |   0.109 ms |   0.106 ms | 0.979 |
| Contiguous attention |        4096 |   0.108 ms |   0.104 ms | 0.966 |
| Direct paged         |         128 |   0.071 ms |   0.074 ms | 1.044 |
| Direct paged         |        1024 |   0.352 ms |   0.353 ms | 1.003 |
| Direct paged         |        4096 |   1.377 ms |   1.388 ms | 1.008 |

## 1. Materialization has almost no additional wall-clock overhead

At 4096:

```text
CUDA event: 215.644 ms
Wall clock: 216.044 ms
Difference:   0.400 ms
Ratio:        1.002
```

The caller-visible latency is only approximately 0.2% higher.

At 1024:

```text
CUDA event: 54.171 ms
Wall clock: 54.328 ms
Difference:  0.158 ms
Ratio:       1.003
```

Therefore, we can conclude:

> The high materialization latency is not caused by a large amount of Python work occurring outside the CUDA-event interval. The expensive behavior is already visible on the measured CUDA timeline.

This does not prove that every millisecond is active GPU calculation. The CUDA interval can contain:

* Numerous GPU indexing/copy operations
* Kernel-launch gaps
* GPU idle gaps while Python submits additional work
* CUDA allocator activity
* Synchronization or dependency gaps within the stream

A profiler would be necessary to divide the 216 ms among those categories. But the wall-clock benchmark shows there is no substantial additional caller-visible cost outside that interval.

## 2. Direct CUDA has very little invocation overhead

At 128:

```text
CUDA event: 0.0707 ms
Wall clock: 0.0738 ms
Difference: 0.0031 ms
```

That is only about three microseconds of additional caller-visible latency.

The ratio looks larger:

```text
1.044
```

because the CUDA operation itself is extremely short. An additional `0.0031 ms` becomes approximately 4.4% when the kernel takes only `0.0707 ms`.

At 4096:

```text
CUDA event: 1.3768 ms
Wall clock: 1.3877 ms
Difference: 0.0109 ms
Ratio:      1.008
```

This confirms that the Python wrapper, PyBind call, C++ function, kernel launches, and final synchronization do not introduce significant additional latency.

## 3. Ratios below 1.0 do not mean wall-clock timing is faster

Some results show:

```text
wall clock < CUDA event
```

For example:

```text
contiguous attention, 4096:

CUDA event: 0.1075 ms
Wall clock: 0.1039 ms
Ratio:      0.966
```

Logically, wall-clock measurement includes more work, so why is it smaller?

Because the two measurements come from separate benchmark runs:

```python
cuda_stats = benchmark_cuda_operation(
    operation
)

wall_stats = benchmark_wall_clock_operation(
    operation
)
```

The CUDA median and wall-clock median are collected from different executions. Small variations can come from:

* GPU clock fluctuations
* CUDA scheduling
* Cache state
* Allocator state
* Measurement resolution
* Other activity on the EC2 instance

The absolute difference is only:

```text
0.0037 ms
```

For an operation lasting approximately `0.1 ms`, a few microseconds of normal variation significantly changes the ratio.

Therefore, values such as `0.966`, `0.979`, or `0.994` should be interpreted as:

```text
approximately 1.0
```

They do not mean wall-clock execution is genuinely faster.

## 4. The 4096 materialized-path result is unusual

At 4096:

```text
CUDA event: 216.719 ms
Wall clock: 236.109 ms
Difference: 19.390 ms
Ratio:      1.089
```

This is the only result showing a meaningful discrepancy.

It may be caused by:

* A noisy benchmark run
* GPU frequency changes
* Temporary EC2 activity
* CUDA allocation behavior
* Different allocator/cache state between the two independent runs
* An unusually slow group of wall-clock samples

Because materialization alone showed only a 0.2% difference at 4096, it is unlikely that the materialized path consistently adds 19 ms of CPU overhead.

I recommend rerunning the 4096 validation. If subsequent runs return ratios near `1.0`, treat this as measurement variation. If it repeatedly stays around `1.09`, we should inspect the detailed mean/min/max values or use paired timing.

## 5. Stronger paired measurement, if needed

Currently, CUDA-event and wall-clock measurements are taken in separate runs. We can measure both during the same iteration:

```python
def benchmark_paired_operation(
    operation: Callable[[], Any],
) -> dict[str, float]:
    for _ in range(WARMUP_ITERS):
        operation()

    torch.cuda.synchronize()

    cuda_latencies_ms = []
    wall_latencies_ms = []

    for _ in range(BENCHMARK_ITERS):
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True
        )
        end_event = torch.cuda.Event(
            enable_timing=True
        )

        wall_start = time.perf_counter()

        start_event.record()

        operation()

        end_event.record()
        end_event.synchronize()

        wall_end = time.perf_counter()

        cuda_latencies_ms.append(
            start_event.elapsed_time(
                end_event
            )
        )

        wall_latencies_ms.append(
            (wall_end - wall_start) * 1000
        )

    cuda_stats = calculate_latency_stats(
        cuda_latencies_ms
    )
    wall_stats = calculate_latency_stats(
        wall_latencies_ms
    )

    return {
        "cuda_event_median_ms":
            cuda_stats["median_ms"],
        "wall_clock_median_ms":
            wall_stats["median_ms"],
        "wall_over_cuda_ratio": (
            wall_stats["median_ms"]
            / cuda_stats["median_ms"]
        ),
    }
```

This would ensure that each CUDA-event sample and wall-clock sample describe the same invocation.

However, I would only add this if the 4096 anomaly repeats. The current result is already sufficient for the main Day 21 conclusion.

## Final finding to preserve

> CUDA-event and end-to-end wall-clock measurements were nearly identical for materialization, contiguous attention, and direct CUDA PagedAttention. Materialization at 4096 tokens measured 215.64 ms using CUDA events and 216.04 ms using wall-clock timing, while direct paged attention measured 1.377 ms and 1.388 ms, respectively. This indicates that there is no large hidden CPU-side overhead outside the CUDA-event interval. The materialization bottleneck is already present on the CUDA timeline, likely due to many inefficient GPU indexing/copy operations and possible submission gaps. One 4096-token materialized-path measurement showed an 8.9% difference and should be rerun to determine whether it was benchmark noise.

This timing-validation CSV is important and should be preserved alongside the main Day 21 benchmark.


The second run confirms that the first run’s 4096 materialized-path result was temporary benchmark noise.

In the first run:

```text
CUDA event: 216.719 ms
Wall clock: 236.109 ms
Ratio:      1.089
```

In the second run:

```text
CUDA event: 219.205 ms
Wall clock: 219.462 ms
Ratio:      1.001
```

The second result is consistent with every other operation: CUDA-event and wall-clock latency are almost identical.

## Second-run summary

| Operation            | Seq. length | CUDA event | Wall clock | Ratio |
| -------------------- | ----------: | ---------: | ---------: | ----: |
| Materialize          |         128 |   7.301 ms |   7.521 ms | 1.030 |
| Materialize          |        1024 |  59.673 ms |  59.531 ms | 0.998 |
| Materialize          |        4096 | 218.460 ms | 219.224 ms | 1.004 |
| Contiguous attention |         128 |   0.117 ms |   0.115 ms | 0.982 |
| Contiguous attention |        1024 |   0.119 ms |   0.112 ms | 0.943 |
| Contiguous attention |        4096 |   0.128 ms |   0.134 ms | 1.050 |
| Materialized path    |         128 |   7.869 ms |   7.815 ms | 0.993 |
| Materialized path    |        1024 |  57.056 ms |  56.998 ms | 0.999 |
| Materialized path    |        4096 | 219.205 ms | 219.462 ms | 1.001 |
| Direct paged         |         128 |   0.075 ms |   0.077 ms | 1.025 |
| Direct paged         |        1024 |   0.355 ms |   0.358 ms | 1.008 |
| Direct paged         |        4096 |   1.318 ms |   1.322 ms | 1.003 |

## Main finding

For the two important end-to-end paths at 4096:

```text
Materialized path:
CUDA event = 219.205 ms
Wall clock = 219.462 ms
Difference =   0.257 ms

Direct paged path:
CUDA event = 1.3179 ms
Wall clock = 1.3223 ms
Difference = 0.0044 ms
```

The caller-visible overhead beyond CUDA-event latency is approximately:

```text
Materialized path:
0.257 / 219.205 × 100 ≈ 0.12%

Direct paged path:
0.0044 / 1.3179 × 100 ≈ 0.33%
```

Both are negligible.

Therefore:

> The performance difference between the two paths is not caused by unmeasured Python or CPU overhead outside the CUDA-event interval. The large materialization cost is already visible on the CUDA timeline.

## Why contiguous-attention ratios fluctuate more

At 1024:

```text
ratio = 0.943
```

At 4096:

```text
ratio = 1.050
```

This looks like a larger percentage variation, but the absolute differences are tiny:

```text
1024:
0.1120 - 0.1188 = -0.0068 ms

4096:
0.1343 - 0.1280 = 0.0064 ms
```

That is only about six microseconds. Because contiguous attention takes approximately `0.12 ms`, normal microsecond-level variation changes the ratio noticeably.

These results should simply be interpreted as:

```text
CUDA-event latency ≈ wall-clock latency
```

## Another independent-run effect

At 1024:

```text
materialize only:
CUDA event = 59.673 ms

complete materialized path:
CUDA event = 57.056 ms
```

The complete path cannot fundamentally be cheaper than materialization alone. This occurs because they are separate benchmark runs with different allocator, cache, GPU-frequency, and system states.

It does not invalidate the test. We should not compare those two independent medians as an exact component decomposition. That is why:

* `component_sum` is used to describe component proportions.
* The directly measured materialized-path latency is used for end-to-end comparisons.

## Final timing-validation conclusion to preserve

> CUDA-event and end-to-end wall-clock measurements were nearly identical across materialization, contiguous attention, the complete materialized path, and direct CUDA PagedAttention. In the second run at 4096 tokens, the materialized path measured 219.205 ms with CUDA events and 219.462 ms with wall-clock timing, while direct paged attention measured 1.318 ms and 1.322 ms, respectively. The wall-clock overhead beyond CUDA-event latency was approximately 0.12% for the materialized path and 0.33% for direct paged attention. This confirms that the large materialization bottleneck is already visible on the CUDA timeline and is not caused by substantial hidden CPU-side overhead. The first run’s 8.9% discrepancy at 4096 did not reproduce and was therefore treated as measurement noise.

This validation is complete.
