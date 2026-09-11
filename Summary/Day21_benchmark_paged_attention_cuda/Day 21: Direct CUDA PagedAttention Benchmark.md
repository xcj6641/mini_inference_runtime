Day 21: Direct CUDA PagedAttention Benchmark

Objective

The purpose of Day 21 was to answer one architectural question:

What changes when the runtime avoids materialize_request_kv() and lets a CUDA attention kernel read KV-cache data directly from paged storage through the request's block_table?

Two execution paths were compared using the same logical query, key, and value data:

Path A: Materialized baseline

PagedKVCache
    -> materialize_request_kv()
    -> contiguous K/V tensors
    -> PyTorch contiguous attention

Path B: Direct paged CUDA

PagedKVCache
    -> block_table addressing
    -> CUDA PagedAttention
    -> attention output

The goal was not to prove that the correctness-first CUDA kernel was more optimized than PyTorch. The goal was to measure the effect of eliminating explicit KV materialization.

Benchmark configuration

The two paths used the following common configuration:

device             = CUDA (NVIDIA A10G)
dtype              = FP16 input/KV
num_layers         = 1
num_kv_heads       = 2
head_dim           = 64
block_size         = 16
layer_idx          = 0
batch_size         = 1
warmup_iterations  = 5
benchmark_iters    = 20
sequence_lengths   = 128, 512, 1024, 2048, 4096

One layer was used because the current CUDA PagedAttention kernel computes one layer. This avoids unfairly comparing one-layer CUDA attention against materialization of a 24-layer cache.

The block table mapped logical blocks to shuffled physical blocks, ensuring that the direct CUDA path performed genuine paged addressing rather than relying on sequential physical placement.

Before timing each sequence length, the CUDA PagedAttention output was checked against contiguous reference attention. Only cases that passed the correctness check were benchmarked.

CUDA events were used for timing. Each operation was warmed up, synchronized, and measured repeatedly. Median latency was used as the primary statistic.

Results

Sequence length

KV size

Materialize

Contiguous attention

Component sum

Materialized path

Direct paged CUDA

Speedup vs. materialized

Materialization share

128

0.0625 MiB

7.192 ms

0.116 ms

7.308 ms

7.395 ms

0.073 ms

101.0x

98.42%

512

0.25 MiB

28.617 ms

0.120 ms

28.738 ms

28.602 ms

0.190 ms

150.2x

99.58%

1024

0.5 MiB

54.298 ms

0.111 ms

54.409 ms

54.590 ms

0.350 ms

155.9x

99.80%

2048

1.0 MiB

108.929 ms

0.113 ms

109.041 ms

115.502 ms

0.675 ms

171.0x

99.90%

4096

2.0 MiB

218.332 ms

0.115 ms

218.446 ms

218.899 ms

1.319 ms

166.0x

99.95%

Component sum is the sum of the separately measured materialization and contiguous-attention medians. Materialized path is a separate end-to-end measurement that performs both operations inside one timed interval. The end-to-end measurement is used for the speedup calculation, while the component sum is used to calculate materialization's percentage of the component-wise baseline.

Findings

1. Explicit KV materialization dominated the baseline

Materialization accounted for 98.42% of the component-wise baseline at 128 tokens and increased to 99.95% at 4096 tokens. Contiguous attention itself required only approximately 0.11-0.12 ms across the tested configurations.

At 4096 tokens:

materialization          = 218.332 ms
contiguous attention     =   0.115 ms
direct CUDA attention    =   1.319 ms

Therefore, almost all latency in the current materialized path came from building contiguous K/V tensors, not from the subsequent attention computation.

2. Materialization latency scaled linearly

Materialization increased from 7.192 ms at 128 tokens to 218.332 ms at 4096 tokens. From 1024 tokens onward, doubling the sequence length approximately doubled materialization latency:

1024 tokens:  54.298 ms
2048 tokens: 108.929 ms
4096 tokens: 218.332 ms

This linear behavior is expected because every additional cached token must be located and copied into the contiguous representation.

The absolute latency is extremely high for only 0.0625-2 MiB of logical KV data. This indicates that the current educational materialize_request_kv() implementation behaves like many small indexing/copy operations rather than one optimized GPU gather. Consequently, the measured speedup applies to this runtime implementation and must not be presented as a universal PagedAttention speedup.

3. Direct CUDA PagedAttention also scaled approximately linearly

Direct paged latency increased as follows:

128 tokens:   0.073 ms
512 tokens:   0.190 ms
1024 tokens:  0.350 ms
2048 tokens:  0.675 ms
4096 tokens:  1.319 ms

For the longer sequences, doubling sequence length produced close to a twofold latency increase:

1024 -> 2048: 1.93x latency
2048 -> 4096: 1.95x latency

This is consistent with the kernel's work: it reads every key, computes a score for every token, performs softmax across all tokens, and reads every value for the weighted sum.

The 4096-token direct-CUDA measurements were particularly stable:

mean    = 1.3189 ms
median  = 1.3189 ms
minimum = 1.3179 ms
maximum = 1.3230 ms

4. Direct paged access removed the dominant cost

Compared with the measured end-to-end materialized path, direct PagedAttention reduced latency by approximately 101-171x across the tested sequence lengths.

The important interpretation is not that the custom CUDA kernel is 101-171x more optimized than PyTorch attention. The improvement comes primarily from avoiding the runtime's expensive explicit KV materialization step.

In fact, at 4096 tokens, direct paged CUDA attention was approximately 11.5x slower than contiguous attention alone:

1.319 ms / 0.115 ms = approximately 11.5x

However, the contiguous operation required K/V to have already been materialized. Once that prerequisite was included, the direct paged path was much faster overall.

5. The naive CUDA implementation still contains expected bottlenecks

The direct PagedAttention implementation uses three separate kernels:

Paged QK dot products.

Scaled softmax.

Weighted value accumulation.

It intentionally does not include production optimizations. Its limitations include:

three separate kernel launches;

no fusion of QK, softmax, and weighted-value computation;

a serial maximum scan in softmax;

a serial exponential-sum scan in softmax;

serial token iteration inside each weighted-value output thread;

no warp-level reductions;

no vectorized memory loads;

no shared-memory tiling of K/V;

only two KV heads and a batch size of one.

These limitations explain why direct paged CUDA was slower than optimized contiguous PyTorch attention alone. They also show why production implementations require significantly more sophisticated kernel engineering.

6. The result is consistent with the earlier Day 19 benchmark

Day 19 measured approximately 1380 ms to materialize 1024 tokens across 24 layers. Normalized per layer:

1380 ms / 24 = approximately 57.5 ms per layer

Day 21 measured 54.3 ms for one layer at the same sequence length. The close agreement provides useful cross-validation between the two benchmark suites.

7. Independent timing runs explain small component differences

The separately measured component sum and end-to-end materialized latency are not expected to be identical. They are medians from different benchmark runs, and medians are not additive. GPU clock behavior, allocator reuse, cache state, and system noise can also affect the measurements.

The 2048-token end-to-end baseline showed more variation than the other cases, causing the reported speedup to peak at 171x. Therefore, the final conclusion should emphasize the observed range rather than treating 171x as a precise universal value.

Conclusion

Direct CUDA PagedAttention eliminated explicit KV-cache materialization while preserving correct logical-to-physical addressing through the block table. In the current educational runtime, materialization accounted for 98.42-99.95% of the component-wise baseline latency and scaled linearly from 7.192 ms at 128 tokens to 218.332 ms at 4096 tokens.

Direct paged attention also scaled approximately linearly, increasing from 0.073 ms to 1.319 ms over the same range. By removing the materialization step, it reduced measured end-to-end latency by approximately 101-171x compared with the current materialized execution path.

The benchmark demonstrates the architectural value of operating directly on paged KV storage. It does not establish that the naive CUDA kernel is more computationally optimized than contiguous PyTorch attention. The custom kernel remains slower than contiguous attention alone because of serial reductions, serial token processing, separate kernel launches, and the absence of fusion and vectorization. The result therefore illustrates both why direct paged access matters and why production PagedAttention requires substantial kernel optimization.

8. All three operations run primarily on the GPU, but they use the GPU in very different ways.

Operation	Where it runs	What executes
materialize_request_kv()	GPU operations launched by Python	PyTorch indexing/copy kernels
Contiguous attention	GPU	PyTorch/CUDA matrix multiplication and softmax kernels
Direct PagedAttention	GPU	Your three custom CUDA kernels