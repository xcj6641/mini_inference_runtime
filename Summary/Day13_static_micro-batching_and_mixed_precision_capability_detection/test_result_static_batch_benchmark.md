# decode step 1:
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 19.672
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [1614]
decode_logits_finite: True
decode_steps: 1
total_decode_ms: 28.185
average_decode_ms: 28.185
generated_tokens: 1
tokens_per_second: 35.480
final_kv_length: 65

# decode step 2:
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 19.624
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [44378]
decode_logits_finite: True
decode_steps: 2
total_decode_ms: 46.543
average_decode_ms: 23.272
generated_tokens: 2
tokens_per_second: 42.971
final_kv_length: 66

# decode step 32:
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 19.538
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
decode_logits_finite: True
decode_steps: 32
total_decode_ms: 596.112
average_decode_ms: 18.628
generated_tokens: 32
tokens_per_second: 53.681
final_kv_length: 96

# decode step 32 with updated print form
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 19.694
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 604.629
average_decode_ms: 18.895
generated_tokens: 32
tokens_per_second: 52.925
final_kv_length: 96
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ 

# decode step=32 with memory monitor
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 21.016
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 598.981
average_decode_ms: 18.718
generated_tokens: 32
tokens_per_second: 53.424
final_kv_length: 96
memory_after_model_load_mb: 1885.285
peak_allocated_mb: 1998.730
incremental_peak_mb: 113.445
kv_bytes: 2359296
kv_mb: 2.250

# test batch = 1,2,4 with some issues.
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark
We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 20.632
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 603.568
average_decode_ms: 18.862
generated_tokens: 32
tokens_per_second: 53.018
final_kv_length: 96
memory_after_model_load_mb: 1885.285
peak_allocated_mb: 1998.730
incremental_peak_mb: 113.445
kv_bytes: 2359296
kv_mb: 2.250

-============================================================
Benchmark: fp32, batch_size=1
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 2
prompt_length: 64
prefill_ms: 20.590
prefill_kv_length: 64
first_next_token_ids: [4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 591.116
average_decode_ms: 18.472
generated_tokens: 64
tokens_per_second: 108.270
final_kv_length: 96
memory_after_model_load_mb: 1893.410
peak_allocated_mb: 2100.426
incremental_peak_mb: 207.016
kv_bytes: 4718592
kv_mb: 4.500

-============================================================
Benchmark: fp32, batch_size=2
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 4
prompt_length: 64
prefill_ms: 22.439
prefill_kv_length: 64
first_next_token_ids: [4128, 4128, 4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13, 13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 598.060
average_decode_ms: 18.689
generated_tokens: 128
tokens_per_second: 214.025
final_kv_length: 96
memory_after_model_load_mb: 1893.410
peak_allocated_mb: 2310.160
incremental_peak_mb: 416.750
kv_bytes: 9437184
kv_mb: 9.000

~============================================================
Benchmark: fp32, batch_size=4
~============================================================

## Note
### Another important memory issue: peak includes both prefill and decode states

During your benchmark, these objects remain alive:

batched_prefill_output
final_decode_output
final_past_key_values

The final KV cache is a newly returned cache, while the original prefill cache may still be referenced by:

batched_prefill_output.past_key_values

Depending on how Hugging Face creates the new cache, peak memory may temporarily include old and new cache tensors.

This does not make your benchmark wrong. It measures the memory behavior of your current implementation.

But be precise in your interpretation:

peak_allocated_mb = peak memory of the complete Python/Hugging Face
prefill-and-decode implementation

It is not simply:

model weights + final KV payload

Temporary logits, activations, old cache references, attention masks, and allocator workspaces contribute to the larger incremental peak.

For example, batch 4 has only:

9 MiB final KV payload

but:

416.75 MiB incremental peak

That difference is largely temporary/runtime memory, not KV payload alone.

This is exactly why recording kv_bytes separately is valuable.


# test batch = 1,2,4, dtype:fp32 with issues solved.
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark

-============================================================
Benchmark: torch.float32, batch_size=1
-============================================================

We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 20.590
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 623.980
average_decode_ms: 19.499
generated_tokens: 32
tokens_per_second: 51.284
final_kv_length: 96
memory_after_model_load_mb: 1885.285
peak_allocated_mb: 1998.730
incremental_peak_mb: 113.445
kv_bytes: 2359296
kv_mb: 2.250

-============================================================
Benchmark: torch.float32, batch_size=2
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 2
prompt_length: 64
prefill_ms: 20.646
prefill_kv_length: 64
first_next_token_ids: [4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 597.243
average_decode_ms: 18.664
generated_tokens: 64
tokens_per_second: 107.159
final_kv_length: 96
memory_after_model_load_mb: 1885.285
peak_allocated_mb: 2100.426
incremental_peak_mb: 215.141
kv_bytes: 4718592
kv_mb: 4.500

-============================================================
Benchmark: torch.float32, batch_size=4
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 4
prompt_length: 64
prefill_ms: 22.381
prefill_kv_length: 64
first_next_token_ids: [4128, 4128, 4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13, 13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 607.703
average_decode_ms: 18.991
generated_tokens: 128
tokens_per_second: 210.629
final_kv_length: 96
memory_after_model_load_mb: 1885.285
peak_allocated_mb: 2310.160
incremental_peak_mb: 424.875
kv_bytes: 9437184
kv_mb: 9.000

## note
### Why does prefill latency rise with batch_size increaseing if matrix operations are parallel?
The GPU can parallelize these extra rows, so time does not need to become four times larger. But the amount of arithmetic and memory traffic still increases.

Parallelism means:

Multiple operations can execute concurrently.

It does not mean:

Any amount of additional work takes exactly zero additional time.

# test batch = 1,2,4, dtype:fp16,fp32,bf16
(vllm-env) ubuntu@ip-172-31-27-236:~/CODE/mini_inference_runtime$ python -m app.benchmarks.static_batch_benchmark

-============================================================
Benchmark: fp32, batch_size=1
-============================================================

We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache oruse an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Benchmark result
----------------
dtype: fp32
batch_size: 1
prompt_length: 64
prefill_ms: 20.719
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 601.517
average_decode_ms: 18.797
generated_tokens: 32
tokens_per_second: 53.199
final_kv_length: 96
memory_after_model_load_mb: 1885.285
baseline_allocated_mb: 1893.410
peak_allocated_mb: 1998.730
incremental_peak_mb: 113.445
kv_bytes: 2359296
kv_mb: 2.250

-============================================================
Benchmark: fp32, batch_size=2
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 2
prompt_length: 64
prefill_ms: 20.495
prefill_kv_length: 64
first_next_token_ids: [4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 588.515
average_decode_ms: 18.391
generated_tokens: 64
tokens_per_second: 108.748
final_kv_length: 96
memory_after_model_load_mb: 1885.285
baseline_allocated_mb: 1893.410
peak_allocated_mb: 2100.426
incremental_peak_mb: 215.141
kv_bytes: 4718592
kv_mb: 4.500

-============================================================
Benchmark: fp32, batch_size=4
-============================================================

Benchmark result
----------------
dtype: fp32
batch_size: 4
prompt_length: 64
prefill_ms: 22.411
prefill_kv_length: 64
first_next_token_ids: [4128, 4128, 4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13, 13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 606.895
average_decode_ms: 18.965
generated_tokens: 128
tokens_per_second: 210.910
final_kv_length: 96
memory_after_model_load_mb: 1885.285
baseline_allocated_mb: 1893.410
peak_allocated_mb: 2310.160
incremental_peak_mb: 424.875
kv_bytes: 9437184
kv_mb: 9.000

-============================================================
Benchmark: fp16, batch_size=1
-============================================================

Benchmark result
----------------
dtype: fp16
batch_size: 1
prompt_length: 64
prefill_ms: 20.399
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 651.290
average_decode_ms: 20.353
generated_tokens: 32
tokens_per_second: 49.133
final_kv_length: 96
memory_after_model_load_mb: 955.022
baseline_allocated_mb: 955.022
peak_allocated_mb: 1020.686
incremental_peak_mb: 65.664
kv_bytes: 1179648
kv_mb: 1.125

-============================================================
Benchmark: fp16, batch_size=2
-============================================================

Benchmark result
----------------
dtype: fp16
batch_size: 2
prompt_length: 64
prefill_ms: 20.622
prefill_kv_length: 64
first_next_token_ids: [4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 657.931
average_decode_ms: 20.560
generated_tokens: 64
tokens_per_second: 97.275
final_kv_length: 96
memory_after_model_load_mb: 955.022
baseline_allocated_mb: 955.022
peak_allocated_mb: 1088.163
incremental_peak_mb: 133.141
kv_bytes: 2359296
kv_mb: 2.250

-============================================================
Benchmark: fp16, batch_size=4
-============================================================

Benchmark result
----------------
dtype: fp16
batch_size: 4
prompt_length: 64
prefill_ms: 30.638
prefill_kv_length: 64
first_next_token_ids: [4128, 4128, 4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13, 13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 746.790
average_decode_ms: 23.337
generated_tokens: 128
tokens_per_second: 171.400
final_kv_length: 96
memory_after_model_load_mb: 955.022
baseline_allocated_mb: 955.022
peak_allocated_mb: 1220.397
incremental_peak_mb: 265.375
kv_bytes: 4718592
kv_mb: 4.500

-============================================================
Benchmark: bf16, batch_size=1
-============================================================

Benchmark result
----------------
dtype: bf16
batch_size: 1
prompt_length: 64
prefill_ms: 19.643
prefill_kv_length: 64
first_next_token_ids: [4128]
all_logits_finite: True
final_decode_output next_token_ids: [13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 651.904
average_decode_ms: 20.372
generated_tokens: 32
tokens_per_second: 49.087
final_kv_length: 96
memory_after_model_load_mb: 957.241
baseline_allocated_mb: 957.241
peak_allocated_mb: 1022.905
incremental_peak_mb: 65.664
kv_bytes: 1179648
kv_mb: 1.125

-============================================================
Benchmark: bf16, batch_size=2
-============================================================

Benchmark result
----------------
dtype: bf16
batch_size: 2
prompt_length: 64
prefill_ms: 19.881
prefill_kv_length: 64
first_next_token_ids: [4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 639.788
average_decode_ms: 19.993
generated_tokens: 64
tokens_per_second: 100.033
final_kv_length: 96
memory_after_model_load_mb: 957.241
baseline_allocated_mb: 957.241
peak_allocated_mb: 1088.569
incremental_peak_mb: 131.328
kv_bytes: 2359296
kv_mb: 2.250

-============================================================
Benchmark: bf16, batch_size=4
-============================================================

Benchmark result
----------------
dtype: bf16
batch_size: 4
prompt_length: 64
prefill_ms: 20.519
prefill_kv_length: 64
first_next_token_ids: [4128, 4128, 4128, 4128]
all_logits_finite: True
final_decode_output next_token_ids: [13, 13, 13, 13]
prefill_logits_finite: True
decode_logits_finite: True
all_logits_finite: True
decode_steps: 32
total_decode_ms: 651.941
average_decode_ms: 20.373
generated_tokens: 128
tokens_per_second: 196.337
final_kv_length: 96
memory_after_model_load_mb: 957.241
baseline_allocated_mb: 957.241
peak_allocated_mb: 1221.709
incremental_peak_mb: 264.469
kv_bytes: 4718592
kv_mb: 4.500

