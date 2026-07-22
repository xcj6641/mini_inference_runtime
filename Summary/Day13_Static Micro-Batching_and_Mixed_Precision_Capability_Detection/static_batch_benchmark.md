
# Get median prefill/decode ms, test batch = 1,2,4, dtype:fp16, fp32, bf16
run 5 times for each batch_dtype corporation

# log result:
Your benchmark is now **functionally complete and the results are stable enough to support real conclusions**. The five repetitions also resolved the suspicious FP16 batch-4 result from the previous run. Your implementation covers warm-up, measured prefill/decode, correctness assertions, memory measurements, structured results, repeated trials, median aggregation, and logging. 

## Final median results

| Dtype | Batch |   Prefill | Decode/step |    Throughput |  Peak memory | KV memory |
| ----- | ----: | --------: | ----------: | ------------: | -----------: | --------: |
| FP32  |     1 | 20.934 ms |   18.618 ms |  53.713 tok/s | 1998.730 MiB | 2.250 MiB |
| FP32  |     2 | 20.654 ms |   18.503 ms | 108.089 tok/s | 2100.426 MiB | 4.500 MiB |
| FP32  |     4 | 22.396 ms |   18.754 ms | 213.293 tok/s | 2310.160 MiB | 9.000 MiB |
| FP16  |     1 | 20.796 ms |   19.737 ms |  50.666 tok/s | 1020.686 MiB | 1.125 MiB |
| FP16  |     2 | 20.542 ms |   19.793 ms | 101.044 tok/s | 1088.163 MiB | 2.250 MiB |
| FP16  |     4 | 21.115 ms |   19.899 ms | 201.017 tok/s | 1220.397 MiB | 4.500 MiB |
| BF16  |     1 | 19.765 ms |   19.517 ms |  51.237 tok/s | 1022.905 MiB | 1.125 MiB |
| BF16  |     2 | 20.067 ms |   19.686 ms | 101.594 tok/s | 1088.569 MiB | 2.250 MiB |
| BF16  |     4 | 20.969 ms |   19.902 ms | 200.989 tok/s | 1221.709 MiB | 4.500 MiB |

All nine configurations passed the important correctness checks: the prefill KV length was 64, the final KV length was 96, logits remained finite, output list sizes matched batch size, and identical greedy requests produced identical tokens. 

## The FP16 batch-4 “slowdown” was not real

Your previous one-run result showed:

```text
FP16 batch 4 prefill: 30.638 ms
FP16 batch 4 decode: 23.337 ms
```

The new five-run medians are:

```text
FP16 batch 4 prefill: 21.115 ms
FP16 batch 4 decode: 19.899 ms
```

Therefore, the earlier result was almost certainly a temporary outlier rather than a stable FP16 kernel problem.

This demonstrates exactly why adding repetitions and reporting the median was necessary.

## What the benchmark proves

### Static batching improves throughput

Across all dtypes, batch-size scaling is nearly linear:

```text
FP32:  53.7 → 108.1 → 213.3 tok/s
FP16:  50.7 → 101.0 → 201.0 tok/s
BF16:  51.2 → 101.6 → 201.0 tok/s
```

Moving from batch 1 to batch 4 gives:

* FP32: about `3.97×` throughput
* FP16: about `3.97×` throughput
* BF16: about `3.92×` throughput

Meanwhile, decode-step latency changes very little. This is strong evidence that the small batch-1 workload underutilizes available GPU parallelism.

### Reduced precision halves memory

Model-load allocation:

```text
FP32:  1885.285 MiB
FP16:   955.022 MiB
BF16:   957.241 MiB
```

KV memory is exactly halved:

```text
Batch 1: FP32 2.250 MiB → FP16/BF16 1.125 MiB
Batch 2: FP32 4.500 MiB → FP16/BF16 2.250 MiB
Batch 4: FP32 9.000 MiB → FP16/BF16 4.500 MiB
```

This verifies both the model weights and KV tensors are actually stored in the configured dtype.

### Reduced precision is not faster here

FP32 decode is consistently about 5–7% faster than FP16/BF16:

```text
Batch 4:
FP32:  18.754 ms/step
FP16:  19.899 ms/step
BF16:  19.902 ms/step
```

Because the results are now consistent across five repetitions, this is no longer likely to be random noise.

The appropriate conclusion is:

> With Qwen2.5-0.5B, short sequences, small static batches, legacy Hugging Face KV cache, and the current PyTorch model path, FP16/BF16 primarily provide memory savings rather than latency improvements.

It does not mean FP32 is generally faster for LLM inference.




# log
2026-07-22 01:20:50,472 INFO app.runtime.pytorch_model_runner - Loading model model=Qwen/Qwen2.5-0.5B-Instruct device=cuda dtype=torch.float32
2026-07-22 01:20:54,562 INFO app.runtime.pytorch_model_runner - Model loaded model=Qwen/Qwen2.5-0.5B-Instruct eos_token_id={151645}
2026-07-22 01:20:54,563 INFO root - 
2026-07-22 01:20:54,563 INFO root - -============================================================
2026-07-22 01:20:54,563 INFO root - Benchmark: fp32, batch_size=1
2026-07-22 01:20:54,563 INFO root - -============================================================
2026-07-22 01:20:54,563 INFO root - 
2026-07-22 01:20:55,613 INFO __main__ - Benchmark result
2026-07-22 01:20:55,613 INFO __main__ - ----------------
2026-07-22 01:20:55,613 INFO __main__ - dtype: fp32
2026-07-22 01:20:55,614 INFO __main__ - batch_size: 1
2026-07-22 01:20:55,614 INFO __main__ - prompt_length: 64
2026-07-22 01:20:55,614 INFO __main__ - prefill_ms: 21.172
2026-07-22 01:20:55,614 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:55,614 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:20:55,614 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:55,614 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:20:55,614 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:55,614 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:55,614 INFO __main__ - decode_steps: 32
2026-07-22 01:20:55,615 INFO __main__ - total_decode_ms: 604.200
2026-07-22 01:20:55,615 INFO __main__ - average_decode_ms: 18.881
2026-07-22 01:20:55,615 INFO __main__ - generated_tokens: 32
2026-07-22 01:20:55,615 INFO __main__ - tokens_per_second: 52.963
2026-07-22 01:20:55,615 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:55,615 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:55,615 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:55,615 INFO __main__ - peak_allocated_mib: 1998.730
2026-07-22 01:20:55,615 INFO __main__ - incremental_over_model_load_mib: 113.445
2026-07-22 01:20:55,615 INFO __main__ - incremental_over_baseline_mib: 105.320
2026-07-22 01:20:55,615 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:20:55,615 INFO __main__ - kv_mib: 2.250
2026-07-22 01:20:56,474 INFO __main__ - Benchmark result
2026-07-22 01:20:56,474 INFO __main__ - ----------------
2026-07-22 01:20:56,474 INFO __main__ - dtype: fp32
2026-07-22 01:20:56,474 INFO __main__ - batch_size: 1
2026-07-22 01:20:56,474 INFO __main__ - prompt_length: 64
2026-07-22 01:20:56,474 INFO __main__ - prefill_ms: 20.356
2026-07-22 01:20:56,474 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:56,474 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:20:56,474 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:56,474 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:20:56,474 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:56,475 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:56,475 INFO __main__ - decode_steps: 32
2026-07-22 01:20:56,475 INFO __main__ - total_decode_ms: 595.083
2026-07-22 01:20:56,475 INFO __main__ - average_decode_ms: 18.596
2026-07-22 01:20:56,475 INFO __main__ - generated_tokens: 32
2026-07-22 01:20:56,475 INFO __main__ - tokens_per_second: 53.774
2026-07-22 01:20:56,475 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:56,475 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:56,475 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:56,475 INFO __main__ - peak_allocated_mib: 1998.730
2026-07-22 01:20:56,475 INFO __main__ - incremental_over_model_load_mib: 113.445
2026-07-22 01:20:56,475 INFO __main__ - incremental_over_baseline_mib: 105.320
2026-07-22 01:20:56,475 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:20:56,475 INFO __main__ - kv_mib: 2.250
2026-07-22 01:20:57,338 INFO __main__ - Benchmark result
2026-07-22 01:20:57,338 INFO __main__ - ----------------
2026-07-22 01:20:57,338 INFO __main__ - dtype: fp32
2026-07-22 01:20:57,338 INFO __main__ - batch_size: 1
2026-07-22 01:20:57,338 INFO __main__ - prompt_length: 64
2026-07-22 01:20:57,338 INFO __main__ - prefill_ms: 20.722
2026-07-22 01:20:57,338 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:57,338 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:20:57,338 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:57,338 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:20:57,338 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:57,339 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:57,339 INFO __main__ - decode_steps: 32
2026-07-22 01:20:57,339 INFO __main__ - total_decode_ms: 599.766
2026-07-22 01:20:57,339 INFO __main__ - average_decode_ms: 18.743
2026-07-22 01:20:57,339 INFO __main__ - generated_tokens: 32
2026-07-22 01:20:57,339 INFO __main__ - tokens_per_second: 53.354
2026-07-22 01:20:57,339 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:57,339 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:57,339 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:57,339 INFO __main__ - peak_allocated_mib: 1998.730
2026-07-22 01:20:57,339 INFO __main__ - incremental_over_model_load_mib: 113.445
2026-07-22 01:20:57,339 INFO __main__ - incremental_over_baseline_mib: 105.320
2026-07-22 01:20:57,339 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:20:57,339 INFO __main__ - kv_mib: 2.250
2026-07-22 01:20:58,198 INFO __main__ - Benchmark result
2026-07-22 01:20:58,198 INFO __main__ - ----------------
2026-07-22 01:20:58,198 INFO __main__ - dtype: fp32
2026-07-22 01:20:58,198 INFO __main__ - batch_size: 1
2026-07-22 01:20:58,198 INFO __main__ - prompt_length: 64
2026-07-22 01:20:58,199 INFO __main__ - prefill_ms: 20.283
2026-07-22 01:20:58,199 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:58,199 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:20:58,199 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:58,199 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:20:58,199 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:58,199 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:58,199 INFO __main__ - decode_steps: 32
2026-07-22 01:20:58,199 INFO __main__ - total_decode_ms: 595.628
2026-07-22 01:20:58,199 INFO __main__ - average_decode_ms: 18.613
2026-07-22 01:20:58,199 INFO __main__ - generated_tokens: 32
2026-07-22 01:20:58,199 INFO __main__ - tokens_per_second: 53.725
2026-07-22 01:20:58,199 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:58,199 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:58,200 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:58,200 INFO __main__ - peak_allocated_mib: 1998.730
2026-07-22 01:20:58,200 INFO __main__ - incremental_over_model_load_mib: 113.445
2026-07-22 01:20:58,200 INFO __main__ - incremental_over_baseline_mib: 105.320
2026-07-22 01:20:58,200 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:20:58,200 INFO __main__ - kv_mib: 2.250
2026-07-22 01:20:59,066 INFO __main__ - Benchmark result
2026-07-22 01:20:59,066 INFO __main__ - ----------------
2026-07-22 01:20:59,066 INFO __main__ - dtype: fp32
2026-07-22 01:20:59,066 INFO __main__ - batch_size: 1
2026-07-22 01:20:59,066 INFO __main__ - prompt_length: 64
2026-07-22 01:20:59,066 INFO __main__ - prefill_ms: 20.957
2026-07-22 01:20:59,066 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:59,066 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:20:59,066 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:59,066 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:20:59,067 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:59,067 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:59,067 INFO __main__ - decode_steps: 32
2026-07-22 01:20:59,067 INFO __main__ - total_decode_ms: 602.376
2026-07-22 01:20:59,067 INFO __main__ - average_decode_ms: 18.824
2026-07-22 01:20:59,067 INFO __main__ - generated_tokens: 32
2026-07-22 01:20:59,067 INFO __main__ - tokens_per_second: 53.123
2026-07-22 01:20:59,067 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:59,067 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:59,067 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:59,067 INFO __main__ - peak_allocated_mib: 1998.730
2026-07-22 01:20:59,067 INFO __main__ - incremental_over_model_load_mib: 113.445
2026-07-22 01:20:59,067 INFO __main__ - incremental_over_baseline_mib: 105.320
2026-07-22 01:20:59,067 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:20:59,067 INFO __main__ - kv_mib: 2.250
2026-07-22 01:20:59,068 INFO root - Median prefill_ms for fp32, batch_size=1: 20.722
2026-07-22 01:20:59,068 INFO root - Median decode_ms for fp32, batch_size=1: 18.743
2026-07-22 01:20:59,068 INFO root - Median tok/s for fp32, batch_size=1: 53.354
2026-07-22 01:20:59,068 INFO root - Median peak memory for fp32, batch_size=1: 1998.730 MiB
2026-07-22 01:20:59,068 INFO root - Median KV memory for fp32, batch_size=1: 2.250 MiB
2026-07-22 01:20:59,068 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}}
2026-07-22 01:20:59,068 INFO root - 
2026-07-22 01:20:59,068 INFO root - -============================================================
2026-07-22 01:20:59,068 INFO root - Benchmark: fp32, batch_size=2
2026-07-22 01:20:59,068 INFO root - -============================================================
2026-07-22 01:20:59,068 INFO root - 
2026-07-22 01:20:59,948 INFO __main__ - Benchmark result
2026-07-22 01:20:59,948 INFO __main__ - ----------------
2026-07-22 01:20:59,948 INFO __main__ - dtype: fp32
2026-07-22 01:20:59,948 INFO __main__ - batch_size: 2
2026-07-22 01:20:59,948 INFO __main__ - prompt_length: 64
2026-07-22 01:20:59,948 INFO __main__ - prefill_ms: 21.241
2026-07-22 01:20:59,948 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:20:59,948 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:20:59,948 INFO __main__ - all_logits_finite: True
2026-07-22 01:20:59,948 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:20:59,948 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:20:59,948 INFO __main__ - decode_logits_finite: True
2026-07-22 01:20:59,948 INFO __main__ - decode_steps: 32
2026-07-22 01:20:59,949 INFO __main__ - total_decode_ms: 611.901
2026-07-22 01:20:59,949 INFO __main__ - average_decode_ms: 19.122
2026-07-22 01:20:59,949 INFO __main__ - generated_tokens: 64
2026-07-22 01:20:59,949 INFO __main__ - tokens_per_second: 104.592
2026-07-22 01:20:59,949 INFO __main__ - final_kv_length: 96
2026-07-22 01:20:59,949 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:20:59,949 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:20:59,949 INFO __main__ - peak_allocated_mib: 2100.426
2026-07-22 01:20:59,949 INFO __main__ - incremental_over_model_load_mib: 215.141
2026-07-22 01:20:59,949 INFO __main__ - incremental_over_baseline_mib: 207.016
2026-07-22 01:20:59,949 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:20:59,949 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:00,886 INFO __main__ - Benchmark result
2026-07-22 01:21:00,886 INFO __main__ - ----------------
2026-07-22 01:21:00,886 INFO __main__ - dtype: fp32
2026-07-22 01:21:00,886 INFO __main__ - batch_size: 2
2026-07-22 01:21:00,886 INFO __main__ - prompt_length: 64
2026-07-22 01:21:00,887 INFO __main__ - prefill_ms: 20.862
2026-07-22 01:21:00,887 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:00,887 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:00,887 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:00,887 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:00,887 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:00,887 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:00,887 INFO __main__ - decode_steps: 32
2026-07-22 01:21:00,887 INFO __main__ - total_decode_ms: 667.513
2026-07-22 01:21:00,887 INFO __main__ - average_decode_ms: 20.860
2026-07-22 01:21:00,887 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:00,887 INFO __main__ - tokens_per_second: 95.878
2026-07-22 01:21:00,887 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:00,887 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:00,888 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:00,888 INFO __main__ - peak_allocated_mib: 2100.426
2026-07-22 01:21:00,888 INFO __main__ - incremental_over_model_load_mib: 215.141
2026-07-22 01:21:00,888 INFO __main__ - incremental_over_baseline_mib: 207.016
2026-07-22 01:21:00,888 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:00,888 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:01,755 INFO __main__ - Benchmark result
2026-07-22 01:21:01,755 INFO __main__ - ----------------
2026-07-22 01:21:01,755 INFO __main__ - dtype: fp32
2026-07-22 01:21:01,755 INFO __main__ - batch_size: 2
2026-07-22 01:21:01,755 INFO __main__ - prompt_length: 64
2026-07-22 01:21:01,755 INFO __main__ - prefill_ms: 20.691
2026-07-22 01:21:01,755 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:01,755 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:01,755 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:01,755 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:01,756 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:01,756 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:01,756 INFO __main__ - decode_steps: 32
2026-07-22 01:21:01,756 INFO __main__ - total_decode_ms: 600.495
2026-07-22 01:21:01,756 INFO __main__ - average_decode_ms: 18.765
2026-07-22 01:21:01,756 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:01,756 INFO __main__ - tokens_per_second: 106.579
2026-07-22 01:21:01,756 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:01,756 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:01,756 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:01,756 INFO __main__ - peak_allocated_mib: 2100.426
2026-07-22 01:21:01,756 INFO __main__ - incremental_over_model_load_mib: 215.141
2026-07-22 01:21:01,756 INFO __main__ - incremental_over_baseline_mib: 207.016
2026-07-22 01:21:01,756 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:01,756 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:02,624 INFO __main__ - Benchmark result
2026-07-22 01:21:02,624 INFO __main__ - ----------------
2026-07-22 01:21:02,624 INFO __main__ - dtype: fp32
2026-07-22 01:21:02,624 INFO __main__ - batch_size: 2
2026-07-22 01:21:02,624 INFO __main__ - prompt_length: 64
2026-07-22 01:21:02,625 INFO __main__ - prefill_ms: 20.467
2026-07-22 01:21:02,625 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:02,625 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:02,625 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:02,625 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:02,625 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:02,625 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:02,625 INFO __main__ - decode_steps: 32
2026-07-22 01:21:02,625 INFO __main__ - total_decode_ms: 600.774
2026-07-22 01:21:02,625 INFO __main__ - average_decode_ms: 18.774
2026-07-22 01:21:02,625 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:02,625 INFO __main__ - tokens_per_second: 106.529
2026-07-22 01:21:02,625 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:02,626 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:02,626 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:02,626 INFO __main__ - peak_allocated_mib: 2100.426
2026-07-22 01:21:02,626 INFO __main__ - incremental_over_model_load_mib: 215.141
2026-07-22 01:21:02,626 INFO __main__ - incremental_over_baseline_mib: 207.016
2026-07-22 01:21:02,626 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:02,626 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:03,497 INFO __main__ - Benchmark result
2026-07-22 01:21:03,498 INFO __main__ - ----------------
2026-07-22 01:21:03,498 INFO __main__ - dtype: fp32
2026-07-22 01:21:03,498 INFO __main__ - batch_size: 2
2026-07-22 01:21:03,498 INFO __main__ - prompt_length: 64
2026-07-22 01:21:03,498 INFO __main__ - prefill_ms: 20.510
2026-07-22 01:21:03,498 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:03,498 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:03,498 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:03,498 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:03,498 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:03,498 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:03,498 INFO __main__ - decode_steps: 32
2026-07-22 01:21:03,498 INFO __main__ - total_decode_ms: 600.686
2026-07-22 01:21:03,498 INFO __main__ - average_decode_ms: 18.771
2026-07-22 01:21:03,499 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:03,499 INFO __main__ - tokens_per_second: 106.545
2026-07-22 01:21:03,499 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:03,499 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:03,499 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:03,499 INFO __main__ - peak_allocated_mib: 2100.426
2026-07-22 01:21:03,499 INFO __main__ - incremental_over_model_load_mib: 215.141
2026-07-22 01:21:03,499 INFO __main__ - incremental_over_baseline_mib: 207.016
2026-07-22 01:21:03,499 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:03,499 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:03,499 INFO root - Median prefill_ms for fp32, batch_size=2: 20.691
2026-07-22 01:21:03,499 INFO root - Median decode_ms for fp32, batch_size=2: 18.774
2026-07-22 01:21:03,499 INFO root - Median tok/s for fp32, batch_size=2: 106.529
2026-07-22 01:21:03,499 INFO root - Median peak memory for fp32, batch_size=2: 2100.426 MiB
2026-07-22 01:21:03,499 INFO root - Median KV memory for fp32, batch_size=2: 4.500 MiB
2026-07-22 01:21:03,500 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}}
2026-07-22 01:21:03,500 INFO root - 
2026-07-22 01:21:03,500 INFO root - -============================================================
2026-07-22 01:21:03,500 INFO root - Benchmark: fp32, batch_size=4
2026-07-22 01:21:03,500 INFO root - -============================================================
2026-07-22 01:21:03,500 INFO root - 
2026-07-22 01:21:04,380 INFO __main__ - Benchmark result
2026-07-22 01:21:04,380 INFO __main__ - ----------------
2026-07-22 01:21:04,380 INFO __main__ - dtype: fp32
2026-07-22 01:21:04,380 INFO __main__ - batch_size: 4
2026-07-22 01:21:04,380 INFO __main__ - prompt_length: 64
2026-07-22 01:21:04,380 INFO __main__ - prefill_ms: 22.248
2026-07-22 01:21:04,381 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:04,381 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:04,381 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:04,381 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:04,381 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:04,381 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:04,381 INFO __main__ - decode_steps: 32
2026-07-22 01:21:04,381 INFO __main__ - total_decode_ms: 601.527
2026-07-22 01:21:04,381 INFO __main__ - average_decode_ms: 18.798
2026-07-22 01:21:04,381 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:04,381 INFO __main__ - tokens_per_second: 212.792
2026-07-22 01:21:04,381 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:04,381 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:04,381 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:04,382 INFO __main__ - peak_allocated_mib: 2310.160
2026-07-22 01:21:04,382 INFO __main__ - incremental_over_model_load_mib: 424.875
2026-07-22 01:21:04,382 INFO __main__ - incremental_over_baseline_mib: 416.750
2026-07-22 01:21:04,382 INFO __main__ - kv_bytes: 9437184
2026-07-22 01:21:04,382 INFO __main__ - kv_mib: 9.000
2026-07-22 01:21:05,270 INFO __main__ - Benchmark result
2026-07-22 01:21:05,270 INFO __main__ - ----------------
2026-07-22 01:21:05,270 INFO __main__ - dtype: fp32
2026-07-22 01:21:05,270 INFO __main__ - batch_size: 4
2026-07-22 01:21:05,270 INFO __main__ - prompt_length: 64
2026-07-22 01:21:05,270 INFO __main__ - prefill_ms: 22.242
2026-07-22 01:21:05,271 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:05,271 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:05,271 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:05,271 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:05,271 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:05,271 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:05,271 INFO __main__ - decode_steps: 32
2026-07-22 01:21:05,271 INFO __main__ - total_decode_ms: 604.433
2026-07-22 01:21:05,271 INFO __main__ - average_decode_ms: 18.889
2026-07-22 01:21:05,271 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:05,271 INFO __main__ - tokens_per_second: 211.769
2026-07-22 01:21:05,271 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:05,271 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:05,272 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:05,272 INFO __main__ - peak_allocated_mib: 2310.160
2026-07-22 01:21:05,272 INFO __main__ - incremental_over_model_load_mib: 424.875
2026-07-22 01:21:05,272 INFO __main__ - incremental_over_baseline_mib: 416.750
2026-07-22 01:21:05,272 INFO __main__ - kv_bytes: 9437184
2026-07-22 01:21:05,272 INFO __main__ - kv_mib: 9.000
2026-07-22 01:21:06,163 INFO __main__ - Benchmark result
2026-07-22 01:21:06,163 INFO __main__ - ----------------
2026-07-22 01:21:06,163 INFO __main__ - dtype: fp32
2026-07-22 01:21:06,163 INFO __main__ - batch_size: 4
2026-07-22 01:21:06,163 INFO __main__ - prompt_length: 64
2026-07-22 01:21:06,163 INFO __main__ - prefill_ms: 22.985
2026-07-22 01:21:06,163 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:06,163 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:06,163 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:06,163 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:06,163 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:06,164 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:06,164 INFO __main__ - decode_steps: 32
2026-07-22 01:21:06,164 INFO __main__ - total_decode_ms: 608.162
2026-07-22 01:21:06,164 INFO __main__ - average_decode_ms: 19.005
2026-07-22 01:21:06,164 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:06,164 INFO __main__ - tokens_per_second: 210.470
2026-07-22 01:21:06,164 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:06,164 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:06,164 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:06,164 INFO __main__ - peak_allocated_mib: 2310.160
2026-07-22 01:21:06,164 INFO __main__ - incremental_over_model_load_mib: 424.875
2026-07-22 01:21:06,164 INFO __main__ - incremental_over_baseline_mib: 416.750
2026-07-22 01:21:06,164 INFO __main__ - kv_bytes: 9437184
2026-07-22 01:21:06,164 INFO __main__ - kv_mib: 9.000
2026-07-22 01:21:07,060 INFO __main__ - Benchmark result
2026-07-22 01:21:07,060 INFO __main__ - ----------------
2026-07-22 01:21:07,060 INFO __main__ - dtype: fp32
2026-07-22 01:21:07,060 INFO __main__ - batch_size: 4
2026-07-22 01:21:07,060 INFO __main__ - prompt_length: 64
2026-07-22 01:21:07,060 INFO __main__ - prefill_ms: 22.226
2026-07-22 01:21:07,060 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:07,060 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:07,060 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:07,060 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:07,060 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:07,060 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:07,060 INFO __main__ - decode_steps: 32
2026-07-22 01:21:07,060 INFO __main__ - total_decode_ms: 614.415
2026-07-22 01:21:07,061 INFO __main__ - average_decode_ms: 19.200
2026-07-22 01:21:07,061 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:07,061 INFO __main__ - tokens_per_second: 208.328
2026-07-22 01:21:07,061 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:07,061 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:07,061 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:07,061 INFO __main__ - peak_allocated_mib: 2310.160
2026-07-22 01:21:07,061 INFO __main__ - incremental_over_model_load_mib: 424.875
2026-07-22 01:21:07,061 INFO __main__ - incremental_over_baseline_mib: 416.750
2026-07-22 01:21:07,061 INFO __main__ - kv_bytes: 9437184
2026-07-22 01:21:07,061 INFO __main__ - kv_mib: 9.000
2026-07-22 01:21:07,956 INFO __main__ - Benchmark result
2026-07-22 01:21:07,956 INFO __main__ - ----------------
2026-07-22 01:21:07,956 INFO __main__ - dtype: fp32
2026-07-22 01:21:07,957 INFO __main__ - batch_size: 4
2026-07-22 01:21:07,957 INFO __main__ - prompt_length: 64
2026-07-22 01:21:07,957 INFO __main__ - prefill_ms: 22.668
2026-07-22 01:21:07,957 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:07,957 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:07,957 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:07,957 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:07,957 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:07,957 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:07,957 INFO __main__ - decode_steps: 32
2026-07-22 01:21:07,957 INFO __main__ - total_decode_ms: 612.765
2026-07-22 01:21:07,957 INFO __main__ - average_decode_ms: 19.149
2026-07-22 01:21:07,957 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:07,957 INFO __main__ - tokens_per_second: 208.889
2026-07-22 01:21:07,958 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:07,958 INFO __main__ - memory_after_model_load_mib: 1885.285
2026-07-22 01:21:07,958 INFO __main__ - baseline_allocated_mib: 1893.410
2026-07-22 01:21:07,958 INFO __main__ - peak_allocated_mib: 2310.160
2026-07-22 01:21:07,958 INFO __main__ - incremental_over_model_load_mib: 424.875
2026-07-22 01:21:07,958 INFO __main__ - incremental_over_baseline_mib: 416.750
2026-07-22 01:21:07,958 INFO __main__ - kv_bytes: 9437184
2026-07-22 01:21:07,958 INFO __main__ - kv_mib: 9.000
2026-07-22 01:21:07,958 INFO root - Median prefill_ms for fp32, batch_size=4: 22.248
2026-07-22 01:21:07,958 INFO root - Median decode_ms for fp32, batch_size=4: 19.005
2026-07-22 01:21:07,958 INFO root - Median tok/s for fp32, batch_size=4: 210.470
2026-07-22 01:21:07,958 INFO root - Median peak memory for fp32, batch_size=4: 2310.160 MiB
2026-07-22 01:21:07,959 INFO root - Median KV memory for fp32, batch_size=4: 9.000 MiB
2026-07-22 01:21:07,959 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}}
2026-07-22 01:21:08,150 INFO app.runtime.pytorch_model_runner - Loading model model=Qwen/Qwen2.5-0.5B-Instruct device=cuda dtype=torch.float16
2026-07-22 01:21:09,227 INFO app.runtime.pytorch_model_runner - Model loaded model=Qwen/Qwen2.5-0.5B-Instruct eos_token_id={151645}
2026-07-22 01:21:09,228 INFO root - 
2026-07-22 01:21:09,228 INFO root - -============================================================
2026-07-22 01:21:09,228 INFO root - Benchmark: fp16, batch_size=1
2026-07-22 01:21:09,228 INFO root - -============================================================
2026-07-22 01:21:09,228 INFO root - 
2026-07-22 01:21:10,209 INFO __main__ - Benchmark result
2026-07-22 01:21:10,209 INFO __main__ - ----------------
2026-07-22 01:21:10,209 INFO __main__ - dtype: fp16
2026-07-22 01:21:10,209 INFO __main__ - batch_size: 1
2026-07-22 01:21:10,209 INFO __main__ - prompt_length: 64
2026-07-22 01:21:10,209 INFO __main__ - prefill_ms: 20.744
2026-07-22 01:21:10,209 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:10,210 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:10,210 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:10,210 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:10,210 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:10,210 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:10,210 INFO __main__ - decode_steps: 32
2026-07-22 01:21:10,210 INFO __main__ - total_decode_ms: 639.777
2026-07-22 01:21:10,210 INFO __main__ - average_decode_ms: 19.993
2026-07-22 01:21:10,210 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:10,210 INFO __main__ - tokens_per_second: 50.017
2026-07-22 01:21:10,210 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:10,210 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:10,210 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:10,210 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:10,210 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:10,210 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:10,211 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:10,211 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:11,122 INFO __main__ - Benchmark result
2026-07-22 01:21:11,123 INFO __main__ - ----------------
2026-07-22 01:21:11,123 INFO __main__ - dtype: fp16
2026-07-22 01:21:11,123 INFO __main__ - batch_size: 1
2026-07-22 01:21:11,123 INFO __main__ - prompt_length: 64
2026-07-22 01:21:11,123 INFO __main__ - prefill_ms: 21.978
2026-07-22 01:21:11,123 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:11,123 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:11,123 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:11,123 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:11,123 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:11,123 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:11,123 INFO __main__ - decode_steps: 32
2026-07-22 01:21:11,123 INFO __main__ - total_decode_ms: 647.136
2026-07-22 01:21:11,123 INFO __main__ - average_decode_ms: 20.223
2026-07-22 01:21:11,123 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:11,124 INFO __main__ - tokens_per_second: 49.449
2026-07-22 01:21:11,124 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:11,124 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:11,124 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:11,124 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:11,124 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:11,124 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:11,124 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:11,124 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:12,033 INFO __main__ - Benchmark result
2026-07-22 01:21:12,033 INFO __main__ - ----------------
2026-07-22 01:21:12,033 INFO __main__ - dtype: fp16
2026-07-22 01:21:12,033 INFO __main__ - batch_size: 1
2026-07-22 01:21:12,033 INFO __main__ - prompt_length: 64
2026-07-22 01:21:12,033 INFO __main__ - prefill_ms: 21.242
2026-07-22 01:21:12,033 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:12,033 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:12,033 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:12,033 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:12,033 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:12,034 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:12,034 INFO __main__ - decode_steps: 32
2026-07-22 01:21:12,034 INFO __main__ - total_decode_ms: 645.575
2026-07-22 01:21:12,034 INFO __main__ - average_decode_ms: 20.174
2026-07-22 01:21:12,034 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:12,034 INFO __main__ - tokens_per_second: 49.568
2026-07-22 01:21:12,034 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:12,034 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:12,034 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:12,034 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:12,034 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:12,034 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:12,034 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:12,034 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:12,945 INFO __main__ - Benchmark result
2026-07-22 01:21:12,945 INFO __main__ - ----------------
2026-07-22 01:21:12,945 INFO __main__ - dtype: fp16
2026-07-22 01:21:12,945 INFO __main__ - batch_size: 1
2026-07-22 01:21:12,945 INFO __main__ - prompt_length: 64
2026-07-22 01:21:12,945 INFO __main__ - prefill_ms: 20.374
2026-07-22 01:21:12,946 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:12,946 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:12,946 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:12,946 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:12,946 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:12,946 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:12,946 INFO __main__ - decode_steps: 32
2026-07-22 01:21:12,946 INFO __main__ - total_decode_ms: 647.980
2026-07-22 01:21:12,946 INFO __main__ - average_decode_ms: 20.249
2026-07-22 01:21:12,946 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:12,946 INFO __main__ - tokens_per_second: 49.384
2026-07-22 01:21:12,946 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:12,946 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:12,946 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:12,946 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:12,947 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:12,947 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:12,947 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:12,947 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:13,858 INFO __main__ - Benchmark result
2026-07-22 01:21:13,858 INFO __main__ - ----------------
2026-07-22 01:21:13,858 INFO __main__ - dtype: fp16
2026-07-22 01:21:13,858 INFO __main__ - batch_size: 1
2026-07-22 01:21:13,858 INFO __main__ - prompt_length: 64
2026-07-22 01:21:13,858 INFO __main__ - prefill_ms: 20.897
2026-07-22 01:21:13,858 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:13,858 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:13,858 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:13,858 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:13,858 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:13,859 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:13,859 INFO __main__ - decode_steps: 32
2026-07-22 01:21:13,859 INFO __main__ - total_decode_ms: 648.402
2026-07-22 01:21:13,859 INFO __main__ - average_decode_ms: 20.263
2026-07-22 01:21:13,859 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:13,859 INFO __main__ - tokens_per_second: 49.352
2026-07-22 01:21:13,859 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:13,859 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:13,859 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:13,859 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:13,859 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:13,859 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:13,859 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:13,859 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:13,860 INFO root - Median prefill_ms for fp16, batch_size=1: 20.897
2026-07-22 01:21:13,860 INFO root - Median decode_ms for fp16, batch_size=1: 20.223
2026-07-22 01:21:13,860 INFO root - Median tok/s for fp16, batch_size=1: 49.449
2026-07-22 01:21:13,860 INFO root - Median peak memory for fp16, batch_size=1: 1024.155 MiB
2026-07-22 01:21:13,860 INFO root - Median KV memory for fp16, batch_size=1: 1.125 MiB
2026-07-22 01:21:13,860 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}}
2026-07-22 01:21:13,860 INFO root - 
2026-07-22 01:21:13,860 INFO root - -============================================================
2026-07-22 01:21:13,860 INFO root - Benchmark: fp16, batch_size=2
2026-07-22 01:21:13,860 INFO root - -============================================================
2026-07-22 01:21:13,860 INFO root - 
2026-07-22 01:21:14,786 INFO __main__ - Benchmark result
2026-07-22 01:21:14,787 INFO __main__ - ----------------
2026-07-22 01:21:14,787 INFO __main__ - dtype: fp16
2026-07-22 01:21:14,787 INFO __main__ - batch_size: 2
2026-07-22 01:21:14,787 INFO __main__ - prompt_length: 64
2026-07-22 01:21:14,787 INFO __main__ - prefill_ms: 22.018
2026-07-22 01:21:14,787 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:14,787 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:14,787 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:14,787 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:14,787 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:14,787 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:14,787 INFO __main__ - decode_steps: 32
2026-07-22 01:21:14,787 INFO __main__ - total_decode_ms: 644.976
2026-07-22 01:21:14,787 INFO __main__ - average_decode_ms: 20.155
2026-07-22 01:21:14,787 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:14,788 INFO __main__ - tokens_per_second: 99.229
2026-07-22 01:21:14,788 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:14,788 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:14,788 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:14,788 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:14,788 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:14,788 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:14,788 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:14,788 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:15,700 INFO __main__ - Benchmark result
2026-07-22 01:21:15,701 INFO __main__ - ----------------
2026-07-22 01:21:15,701 INFO __main__ - dtype: fp16
2026-07-22 01:21:15,701 INFO __main__ - batch_size: 2
2026-07-22 01:21:15,701 INFO __main__ - prompt_length: 64
2026-07-22 01:21:15,701 INFO __main__ - prefill_ms: 21.666
2026-07-22 01:21:15,701 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:15,701 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:15,701 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:15,701 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:15,701 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:15,701 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:15,701 INFO __main__ - decode_steps: 32
2026-07-22 01:21:15,701 INFO __main__ - total_decode_ms: 643.615
2026-07-22 01:21:15,701 INFO __main__ - average_decode_ms: 20.113
2026-07-22 01:21:15,702 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:15,702 INFO __main__ - tokens_per_second: 99.438
2026-07-22 01:21:15,702 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:15,702 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:15,702 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:15,702 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:15,702 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:15,702 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:15,702 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:15,702 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:16,611 INFO __main__ - Benchmark result
2026-07-22 01:21:16,611 INFO __main__ - ----------------
2026-07-22 01:21:16,611 INFO __main__ - dtype: fp16
2026-07-22 01:21:16,611 INFO __main__ - batch_size: 2
2026-07-22 01:21:16,611 INFO __main__ - prompt_length: 64
2026-07-22 01:21:16,612 INFO __main__ - prefill_ms: 20.901
2026-07-22 01:21:16,612 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:16,612 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:16,612 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:16,612 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:16,612 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:16,612 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:16,612 INFO __main__ - decode_steps: 32
2026-07-22 01:21:16,612 INFO __main__ - total_decode_ms: 642.428
2026-07-22 01:21:16,612 INFO __main__ - average_decode_ms: 20.076
2026-07-22 01:21:16,612 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:16,612 INFO __main__ - tokens_per_second: 99.622
2026-07-22 01:21:16,612 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:16,612 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:16,612 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:16,613 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:16,613 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:16,613 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:16,613 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:16,613 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:17,526 INFO __main__ - Benchmark result
2026-07-22 01:21:17,526 INFO __main__ - ----------------
2026-07-22 01:21:17,526 INFO __main__ - dtype: fp16
2026-07-22 01:21:17,526 INFO __main__ - batch_size: 2
2026-07-22 01:21:17,526 INFO __main__ - prompt_length: 64
2026-07-22 01:21:17,526 INFO __main__ - prefill_ms: 20.959
2026-07-22 01:21:17,526 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:17,526 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:17,526 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:17,526 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:17,527 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:17,527 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:17,527 INFO __main__ - decode_steps: 32
2026-07-22 01:21:17,527 INFO __main__ - total_decode_ms: 645.096
2026-07-22 01:21:17,527 INFO __main__ - average_decode_ms: 20.159
2026-07-22 01:21:17,527 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:17,527 INFO __main__ - tokens_per_second: 99.210
2026-07-22 01:21:17,527 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:17,527 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:17,527 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:17,527 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:17,527 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:17,527 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:17,528 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:17,528 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:18,439 INFO __main__ - Benchmark result
2026-07-22 01:21:18,439 INFO __main__ - ----------------
2026-07-22 01:21:18,439 INFO __main__ - dtype: fp16
2026-07-22 01:21:18,439 INFO __main__ - batch_size: 2
2026-07-22 01:21:18,439 INFO __main__ - prompt_length: 64
2026-07-22 01:21:18,439 INFO __main__ - prefill_ms: 20.624
2026-07-22 01:21:18,439 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:18,439 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:18,439 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:18,439 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:18,440 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:18,440 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:18,440 INFO __main__ - decode_steps: 32
2026-07-22 01:21:18,440 INFO __main__ - total_decode_ms: 645.723
2026-07-22 01:21:18,440 INFO __main__ - average_decode_ms: 20.179
2026-07-22 01:21:18,440 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:18,440 INFO __main__ - tokens_per_second: 99.114
2026-07-22 01:21:18,440 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:18,440 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:18,440 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:18,440 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:18,440 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:18,440 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:18,440 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:18,441 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:18,441 INFO root - Median prefill_ms for fp16, batch_size=2: 20.959
2026-07-22 01:21:18,441 INFO root - Median decode_ms for fp16, batch_size=2: 20.155
2026-07-22 01:21:18,441 INFO root - Median tok/s for fp16, batch_size=2: 99.229
2026-07-22 01:21:18,441 INFO root - Median peak memory for fp16, batch_size=2: 1091.631 MiB
2026-07-22 01:21:18,441 INFO root - Median KV memory for fp16, batch_size=2: 2.250 MiB
2026-07-22 01:21:18,441 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('fp16', 2): {'prefill_ms': 20.95920406281948, 'decode_ms': 20.15549565840047, 'median_tok_per_s': 99.22851979908685, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}}
2026-07-22 01:21:18,441 INFO root - 
2026-07-22 01:21:18,441 INFO root - -============================================================
2026-07-22 01:21:18,441 INFO root - Benchmark: fp16, batch_size=4
2026-07-22 01:21:18,442 INFO root - -============================================================
2026-07-22 01:21:18,442 INFO root - 
2026-07-22 01:21:19,358 INFO __main__ - Benchmark result
2026-07-22 01:21:19,358 INFO __main__ - ----------------
2026-07-22 01:21:19,358 INFO __main__ - dtype: fp16
2026-07-22 01:21:19,358 INFO __main__ - batch_size: 4
2026-07-22 01:21:19,358 INFO __main__ - prompt_length: 64
2026-07-22 01:21:19,358 INFO __main__ - prefill_ms: 21.291
2026-07-22 01:21:19,358 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:19,358 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:19,358 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:19,358 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:19,358 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:19,358 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:19,359 INFO __main__ - decode_steps: 32
2026-07-22 01:21:19,359 INFO __main__ - total_decode_ms: 643.315
2026-07-22 01:21:19,359 INFO __main__ - average_decode_ms: 20.104
2026-07-22 01:21:19,359 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:19,359 INFO __main__ - tokens_per_second: 198.969
2026-07-22 01:21:19,359 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:19,359 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:19,359 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:19,359 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:19,359 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:19,359 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:19,359 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:19,359 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:20,289 INFO __main__ - Benchmark result
2026-07-22 01:21:20,289 INFO __main__ - ----------------
2026-07-22 01:21:20,289 INFO __main__ - dtype: fp16
2026-07-22 01:21:20,289 INFO __main__ - batch_size: 4
2026-07-22 01:21:20,289 INFO __main__ - prompt_length: 64
2026-07-22 01:21:20,289 INFO __main__ - prefill_ms: 23.938
2026-07-22 01:21:20,289 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:20,290 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:20,290 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:20,290 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:20,290 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:20,290 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:20,290 INFO __main__ - decode_steps: 32
2026-07-22 01:21:20,290 INFO __main__ - total_decode_ms: 648.673
2026-07-22 01:21:20,290 INFO __main__ - average_decode_ms: 20.271
2026-07-22 01:21:20,290 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:20,290 INFO __main__ - tokens_per_second: 197.326
2026-07-22 01:21:20,290 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:20,290 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:20,290 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:20,290 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:20,291 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:20,291 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:20,291 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:20,291 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:21,218 INFO __main__ - Benchmark result
2026-07-22 01:21:21,218 INFO __main__ - ----------------
2026-07-22 01:21:21,218 INFO __main__ - dtype: fp16
2026-07-22 01:21:21,218 INFO __main__ - batch_size: 4
2026-07-22 01:21:21,218 INFO __main__ - prompt_length: 64
2026-07-22 01:21:21,219 INFO __main__ - prefill_ms: 21.368
2026-07-22 01:21:21,219 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:21,219 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:21,219 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:21,219 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:21,219 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:21,219 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:21,219 INFO __main__ - decode_steps: 32
2026-07-22 01:21:21,219 INFO __main__ - total_decode_ms: 653.733
2026-07-22 01:21:21,219 INFO __main__ - average_decode_ms: 20.429
2026-07-22 01:21:21,219 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:21,219 INFO __main__ - tokens_per_second: 195.799
2026-07-22 01:21:21,219 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:21,219 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:21,220 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:21,220 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:21,220 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:21,220 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:21,220 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:21,220 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:22,134 INFO __main__ - Benchmark result
2026-07-22 01:21:22,134 INFO __main__ - ----------------
2026-07-22 01:21:22,134 INFO __main__ - dtype: fp16
2026-07-22 01:21:22,134 INFO __main__ - batch_size: 4
2026-07-22 01:21:22,134 INFO __main__ - prompt_length: 64
2026-07-22 01:21:22,134 INFO __main__ - prefill_ms: 21.340
2026-07-22 01:21:22,134 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:22,134 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:22,134 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:22,134 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:22,135 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:22,135 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:22,135 INFO __main__ - decode_steps: 32
2026-07-22 01:21:22,135 INFO __main__ - total_decode_ms: 640.821
2026-07-22 01:21:22,135 INFO __main__ - average_decode_ms: 20.026
2026-07-22 01:21:22,135 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:22,135 INFO __main__ - tokens_per_second: 199.744
2026-07-22 01:21:22,135 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:22,135 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:22,135 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:22,135 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:22,135 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:22,135 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:22,135 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:22,135 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:23,057 INFO __main__ - Benchmark result
2026-07-22 01:21:23,057 INFO __main__ - ----------------
2026-07-22 01:21:23,057 INFO __main__ - dtype: fp16
2026-07-22 01:21:23,057 INFO __main__ - batch_size: 4
2026-07-22 01:21:23,057 INFO __main__ - prompt_length: 64
2026-07-22 01:21:23,058 INFO __main__ - prefill_ms: 21.686
2026-07-22 01:21:23,058 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:23,058 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:23,058 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:23,058 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:23,058 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:23,058 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:23,058 INFO __main__ - decode_steps: 32
2026-07-22 01:21:23,058 INFO __main__ - total_decode_ms: 649.066
2026-07-22 01:21:23,058 INFO __main__ - average_decode_ms: 20.283
2026-07-22 01:21:23,058 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:23,058 INFO __main__ - tokens_per_second: 197.207
2026-07-22 01:21:23,058 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:23,058 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:23,059 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:23,059 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:23,059 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:23,059 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:23,059 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:23,059 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:23,059 INFO root - Median prefill_ms for fp16, batch_size=4: 21.368
2026-07-22 01:21:23,059 INFO root - Median decode_ms for fp16, batch_size=4: 20.271
2026-07-22 01:21:23,059 INFO root - Median tok/s for fp16, batch_size=4: 197.326
2026-07-22 01:21:23,059 INFO root - Median peak memory for fp16, batch_size=4: 1223.866 MiB
2026-07-22 01:21:23,059 INFO root - Median KV memory for fp16, batch_size=4: 4.500 MiB
2026-07-22 01:21:23,060 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('fp16', 2): {'prefill_ms': 20.95920406281948, 'decode_ms': 20.15549565840047, 'median_tok_per_s': 99.22851979908685, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}, ('fp16', 4): {'prefill_ms': 21.368002984672785, 'decode_ms': 20.27103506407002, 'median_tok_per_s': 197.32588826161697, 'peak_memory_mb': 1223.86572265625, 'kv_memory_mb': 4.5}}
2026-07-22 01:21:23,219 INFO app.runtime.pytorch_model_runner - Loading model model=Qwen/Qwen2.5-0.5B-Instruct device=cuda dtype=torch.bfloat16
2026-07-22 01:21:23,829 INFO app.runtime.pytorch_model_runner - Model loaded model=Qwen/Qwen2.5-0.5B-Instruct eos_token_id={151645}
2026-07-22 01:21:23,830 INFO root - 
2026-07-22 01:21:23,830 INFO root - -============================================================
2026-07-22 01:21:23,830 INFO root - Benchmark: bf16, batch_size=1
2026-07-22 01:21:23,830 INFO root - -============================================================
2026-07-22 01:21:23,830 INFO root - 
2026-07-22 01:21:24,797 INFO __main__ - Benchmark result
2026-07-22 01:21:24,797 INFO __main__ - ----------------
2026-07-22 01:21:24,797 INFO __main__ - dtype: bf16
2026-07-22 01:21:24,797 INFO __main__ - batch_size: 1
2026-07-22 01:21:24,798 INFO __main__ - prompt_length: 64
2026-07-22 01:21:24,798 INFO __main__ - prefill_ms: 21.053
2026-07-22 01:21:24,798 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:24,798 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:24,798 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:24,798 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:24,798 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:24,798 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:24,798 INFO __main__ - decode_steps: 32
2026-07-22 01:21:24,798 INFO __main__ - total_decode_ms: 633.072
2026-07-22 01:21:24,798 INFO __main__ - average_decode_ms: 19.783
2026-07-22 01:21:24,798 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:24,798 INFO __main__ - tokens_per_second: 50.547
2026-07-22 01:21:24,799 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:24,799 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:24,799 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:24,799 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:24,799 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:24,799 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:24,799 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:24,799 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:25,706 INFO __main__ - Benchmark result
2026-07-22 01:21:25,706 INFO __main__ - ----------------
2026-07-22 01:21:25,706 INFO __main__ - dtype: bf16
2026-07-22 01:21:25,706 INFO __main__ - batch_size: 1
2026-07-22 01:21:25,706 INFO __main__ - prompt_length: 64
2026-07-22 01:21:25,706 INFO __main__ - prefill_ms: 20.706
2026-07-22 01:21:25,706 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:25,706 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:25,707 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:25,707 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:25,707 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:25,707 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:25,707 INFO __main__ - decode_steps: 32
2026-07-22 01:21:25,707 INFO __main__ - total_decode_ms: 643.470
2026-07-22 01:21:25,707 INFO __main__ - average_decode_ms: 20.108
2026-07-22 01:21:25,707 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:25,707 INFO __main__ - tokens_per_second: 49.730
2026-07-22 01:21:25,707 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:25,707 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:25,707 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:25,707 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:25,707 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:25,708 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:25,708 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:25,708 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:26,613 INFO __main__ - Benchmark result
2026-07-22 01:21:26,613 INFO __main__ - ----------------
2026-07-22 01:21:26,613 INFO __main__ - dtype: bf16
2026-07-22 01:21:26,613 INFO __main__ - batch_size: 1
2026-07-22 01:21:26,613 INFO __main__ - prompt_length: 64
2026-07-22 01:21:26,614 INFO __main__ - prefill_ms: 20.487
2026-07-22 01:21:26,614 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:26,614 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:26,614 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:26,614 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:26,614 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:26,614 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:26,614 INFO __main__ - decode_steps: 32
2026-07-22 01:21:26,614 INFO __main__ - total_decode_ms: 637.130
2026-07-22 01:21:26,614 INFO __main__ - average_decode_ms: 19.910
2026-07-22 01:21:26,614 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:26,614 INFO __main__ - tokens_per_second: 50.225
2026-07-22 01:21:26,614 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:26,614 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:26,615 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:26,615 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:26,615 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:26,615 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:26,615 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:26,615 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:27,510 INFO __main__ - Benchmark result
2026-07-22 01:21:27,510 INFO __main__ - ----------------
2026-07-22 01:21:27,510 INFO __main__ - dtype: bf16
2026-07-22 01:21:27,510 INFO __main__ - batch_size: 1
2026-07-22 01:21:27,510 INFO __main__ - prompt_length: 64
2026-07-22 01:21:27,510 INFO __main__ - prefill_ms: 20.943
2026-07-22 01:21:27,511 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:27,511 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:27,511 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:27,511 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:27,511 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:27,511 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:27,511 INFO __main__ - decode_steps: 32
2026-07-22 01:21:27,511 INFO __main__ - total_decode_ms: 632.760
2026-07-22 01:21:27,511 INFO __main__ - average_decode_ms: 19.774
2026-07-22 01:21:27,511 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:27,511 INFO __main__ - tokens_per_second: 50.572
2026-07-22 01:21:27,511 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:27,511 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:27,512 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:27,512 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:27,512 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:27,512 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:27,512 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:27,512 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:28,414 INFO __main__ - Benchmark result
2026-07-22 01:21:28,414 INFO __main__ - ----------------
2026-07-22 01:21:28,414 INFO __main__ - dtype: bf16
2026-07-22 01:21:28,414 INFO __main__ - batch_size: 1
2026-07-22 01:21:28,414 INFO __main__ - prompt_length: 64
2026-07-22 01:21:28,414 INFO __main__ - prefill_ms: 19.908
2026-07-22 01:21:28,414 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:28,414 INFO __main__ - first_next_token_ids: [4128]
2026-07-22 01:21:28,414 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:28,414 INFO __main__ - final_decode_output next_token_ids: [13]
2026-07-22 01:21:28,414 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:28,414 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:28,415 INFO __main__ - decode_steps: 32
2026-07-22 01:21:28,415 INFO __main__ - total_decode_ms: 640.020
2026-07-22 01:21:28,415 INFO __main__ - average_decode_ms: 20.001
2026-07-22 01:21:28,415 INFO __main__ - generated_tokens: 32
2026-07-22 01:21:28,415 INFO __main__ - tokens_per_second: 49.998
2026-07-22 01:21:28,415 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:28,415 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:28,415 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:28,415 INFO __main__ - peak_allocated_mib: 1024.155
2026-07-22 01:21:28,415 INFO __main__ - incremental_over_model_load_mib: 65.664
2026-07-22 01:21:28,415 INFO __main__ - incremental_over_baseline_mib: 65.664
2026-07-22 01:21:28,415 INFO __main__ - kv_bytes: 1179648
2026-07-22 01:21:28,415 INFO __main__ - kv_mib: 1.125
2026-07-22 01:21:28,416 INFO root - Median prefill_ms for bf16, batch_size=1: 20.706
2026-07-22 01:21:28,416 INFO root - Median decode_ms for bf16, batch_size=1: 19.910
2026-07-22 01:21:28,416 INFO root - Median tok/s for bf16, batch_size=1: 50.225
2026-07-22 01:21:28,416 INFO root - Median peak memory for bf16, batch_size=1: 1024.155 MiB
2026-07-22 01:21:28,416 INFO root - Median KV memory for bf16, batch_size=1: 1.125 MiB
2026-07-22 01:21:28,416 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('fp16', 2): {'prefill_ms': 20.95920406281948, 'decode_ms': 20.15549565840047, 'median_tok_per_s': 99.22851979908685, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}, ('fp16', 4): {'prefill_ms': 21.368002984672785, 'decode_ms': 20.27103506407002, 'median_tok_per_s': 197.32588826161697, 'peak_memory_mb': 1223.86572265625, 'kv_memory_mb': 4.5}, ('bf16', 1): {'prefill_ms': 20.705511095002294, 'decode_ms': 19.910303279175423, 'median_tok_per_s': 50.225252020441076, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}}
2026-07-22 01:21:28,416 INFO root - 
2026-07-22 01:21:28,416 INFO root - -============================================================
2026-07-22 01:21:28,416 INFO root - Benchmark: bf16, batch_size=2
2026-07-22 01:21:28,416 INFO root - -============================================================
2026-07-22 01:21:28,417 INFO root - 
2026-07-22 01:21:29,343 INFO __main__ - Benchmark result
2026-07-22 01:21:29,343 INFO __main__ - ----------------
2026-07-22 01:21:29,343 INFO __main__ - dtype: bf16
2026-07-22 01:21:29,343 INFO __main__ - batch_size: 2
2026-07-22 01:21:29,343 INFO __main__ - prompt_length: 64
2026-07-22 01:21:29,343 INFO __main__ - prefill_ms: 20.514
2026-07-22 01:21:29,343 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:29,343 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:29,343 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:29,343 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:29,344 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:29,344 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:29,344 INFO __main__ - decode_steps: 32
2026-07-22 01:21:29,344 INFO __main__ - total_decode_ms: 642.833
2026-07-22 01:21:29,344 INFO __main__ - average_decode_ms: 20.089
2026-07-22 01:21:29,344 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:29,344 INFO __main__ - tokens_per_second: 99.559
2026-07-22 01:21:29,344 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:29,344 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:29,344 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:29,344 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:29,344 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:29,344 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:29,344 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:29,345 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:30,258 INFO __main__ - Benchmark result
2026-07-22 01:21:30,258 INFO __main__ - ----------------
2026-07-22 01:21:30,259 INFO __main__ - dtype: bf16
2026-07-22 01:21:30,259 INFO __main__ - batch_size: 2
2026-07-22 01:21:30,259 INFO __main__ - prompt_length: 64
2026-07-22 01:21:30,259 INFO __main__ - prefill_ms: 20.966
2026-07-22 01:21:30,259 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:30,259 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:30,259 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:30,259 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:30,259 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:30,259 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:30,259 INFO __main__ - decode_steps: 32
2026-07-22 01:21:30,259 INFO __main__ - total_decode_ms: 646.382
2026-07-22 01:21:30,259 INFO __main__ - average_decode_ms: 20.199
2026-07-22 01:21:30,260 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:30,260 INFO __main__ - tokens_per_second: 99.013
2026-07-22 01:21:30,260 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:30,260 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:30,260 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:30,260 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:30,260 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:30,260 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:30,260 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:30,260 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:31,182 INFO __main__ - Benchmark result
2026-07-22 01:21:31,182 INFO __main__ - ----------------
2026-07-22 01:21:31,182 INFO __main__ - dtype: bf16
2026-07-22 01:21:31,183 INFO __main__ - batch_size: 2
2026-07-22 01:21:31,183 INFO __main__ - prompt_length: 64
2026-07-22 01:21:31,183 INFO __main__ - prefill_ms: 20.805
2026-07-22 01:21:31,183 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:31,183 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:31,183 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:31,183 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:31,183 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:31,183 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:31,183 INFO __main__ - decode_steps: 32
2026-07-22 01:21:31,183 INFO __main__ - total_decode_ms: 655.796
2026-07-22 01:21:31,183 INFO __main__ - average_decode_ms: 20.494
2026-07-22 01:21:31,183 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:31,183 INFO __main__ - tokens_per_second: 97.591
2026-07-22 01:21:31,184 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:31,184 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:31,184 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:31,184 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:31,184 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:31,184 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:31,184 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:31,184 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:32,090 INFO __main__ - Benchmark result
2026-07-22 01:21:32,090 INFO __main__ - ----------------
2026-07-22 01:21:32,090 INFO __main__ - dtype: bf16
2026-07-22 01:21:32,090 INFO __main__ - batch_size: 2
2026-07-22 01:21:32,090 INFO __main__ - prompt_length: 64
2026-07-22 01:21:32,091 INFO __main__ - prefill_ms: 21.234
2026-07-22 01:21:32,091 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:32,091 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:32,091 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:32,091 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:32,091 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:32,091 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:32,091 INFO __main__ - decode_steps: 32
2026-07-22 01:21:32,091 INFO __main__ - total_decode_ms: 638.904
2026-07-22 01:21:32,091 INFO __main__ - average_decode_ms: 19.966
2026-07-22 01:21:32,091 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:32,091 INFO __main__ - tokens_per_second: 100.172
2026-07-22 01:21:32,091 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:32,091 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:32,092 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:32,092 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:32,092 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:32,092 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:32,092 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:32,092 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:33,002 INFO __main__ - Benchmark result
2026-07-22 01:21:33,003 INFO __main__ - ----------------
2026-07-22 01:21:33,003 INFO __main__ - dtype: bf16
2026-07-22 01:21:33,003 INFO __main__ - batch_size: 2
2026-07-22 01:21:33,003 INFO __main__ - prompt_length: 64
2026-07-22 01:21:33,003 INFO __main__ - prefill_ms: 20.827
2026-07-22 01:21:33,003 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:33,003 INFO __main__ - first_next_token_ids: [4128, 4128]
2026-07-22 01:21:33,003 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:33,003 INFO __main__ - final_decode_output next_token_ids: [13, 13]
2026-07-22 01:21:33,003 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:33,003 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:33,003 INFO __main__ - decode_steps: 32
2026-07-22 01:21:33,003 INFO __main__ - total_decode_ms: 645.260
2026-07-22 01:21:33,004 INFO __main__ - average_decode_ms: 20.164
2026-07-22 01:21:33,004 INFO __main__ - generated_tokens: 64
2026-07-22 01:21:33,004 INFO __main__ - tokens_per_second: 99.185
2026-07-22 01:21:33,004 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:33,004 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:33,004 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:33,004 INFO __main__ - peak_allocated_mib: 1091.631
2026-07-22 01:21:33,004 INFO __main__ - incremental_over_model_load_mib: 133.141
2026-07-22 01:21:33,004 INFO __main__ - incremental_over_baseline_mib: 133.141
2026-07-22 01:21:33,004 INFO __main__ - kv_bytes: 2359296
2026-07-22 01:21:33,004 INFO __main__ - kv_mib: 2.250
2026-07-22 01:21:33,004 INFO root - Median prefill_ms for bf16, batch_size=2: 20.827
2026-07-22 01:21:33,005 INFO root - Median decode_ms for bf16, batch_size=2: 20.164
2026-07-22 01:21:33,005 INFO root - Median tok/s for bf16, batch_size=2: 99.185
2026-07-22 01:21:33,005 INFO root - Median peak memory for bf16, batch_size=2: 1091.631 MiB
2026-07-22 01:21:33,005 INFO root - Median KV memory for bf16, batch_size=2: 2.250 MiB
2026-07-22 01:21:33,005 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('fp16', 2): {'prefill_ms': 20.95920406281948, 'decode_ms': 20.15549565840047, 'median_tok_per_s': 99.22851979908685, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}, ('fp16', 4): {'prefill_ms': 21.368002984672785, 'decode_ms': 20.27103506407002, 'median_tok_per_s': 197.32588826161697, 'peak_memory_mb': 1223.86572265625, 'kv_memory_mb': 4.5}, ('bf16', 1): {'prefill_ms': 20.705511095002294, 'decode_ms': 19.910303279175423, 'median_tok_per_s': 50.225252020441076, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('bf16', 2): {'prefill_ms': 20.82672296091914, 'decode_ms': 20.164361274510156, 'median_tok_per_s': 99.18489223500511, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}}
2026-07-22 01:21:33,005 INFO root - 
2026-07-22 01:21:33,005 INFO root - -============================================================
2026-07-22 01:21:33,005 INFO root - Benchmark: bf16, batch_size=4
2026-07-22 01:21:33,005 INFO root - -============================================================
2026-07-22 01:21:33,005 INFO root - 
2026-07-22 01:21:33,927 INFO __main__ - Benchmark result
2026-07-22 01:21:33,927 INFO __main__ - ----------------
2026-07-22 01:21:33,927 INFO __main__ - dtype: bf16
2026-07-22 01:21:33,927 INFO __main__ - batch_size: 4
2026-07-22 01:21:33,927 INFO __main__ - prompt_length: 64
2026-07-22 01:21:33,927 INFO __main__ - prefill_ms: 21.953
2026-07-22 01:21:33,927 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:33,927 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:33,927 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:33,927 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:33,927 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:33,928 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:33,928 INFO __main__ - decode_steps: 32
2026-07-22 01:21:33,928 INFO __main__ - total_decode_ms: 649.433
2026-07-22 01:21:33,928 INFO __main__ - average_decode_ms: 20.295
2026-07-22 01:21:33,928 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:33,928 INFO __main__ - tokens_per_second: 197.095
2026-07-22 01:21:33,928 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:33,928 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:33,928 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:33,928 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:33,928 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:33,928 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:33,928 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:33,928 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:34,848 INFO __main__ - Benchmark result
2026-07-22 01:21:34,849 INFO __main__ - ----------------
2026-07-22 01:21:34,849 INFO __main__ - dtype: bf16
2026-07-22 01:21:34,849 INFO __main__ - batch_size: 4
2026-07-22 01:21:34,849 INFO __main__ - prompt_length: 64
2026-07-22 01:21:34,849 INFO __main__ - prefill_ms: 21.048
2026-07-22 01:21:34,849 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:34,849 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:34,849 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:34,849 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:34,849 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:34,849 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:34,849 INFO __main__ - decode_steps: 32
2026-07-22 01:21:34,849 INFO __main__ - total_decode_ms: 645.541
2026-07-22 01:21:34,849 INFO __main__ - average_decode_ms: 20.173
2026-07-22 01:21:34,850 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:34,850 INFO __main__ - tokens_per_second: 198.283
2026-07-22 01:21:34,850 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:34,850 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:34,850 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:34,850 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:34,850 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:34,850 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:34,850 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:34,850 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:35,783 INFO __main__ - Benchmark result
2026-07-22 01:21:35,783 INFO __main__ - ----------------
2026-07-22 01:21:35,783 INFO __main__ - dtype: bf16
2026-07-22 01:21:35,783 INFO __main__ - batch_size: 4
2026-07-22 01:21:35,783 INFO __main__ - prompt_length: 64
2026-07-22 01:21:35,783 INFO __main__ - prefill_ms: 22.320
2026-07-22 01:21:35,783 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:35,783 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:35,783 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:35,784 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:35,784 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:35,784 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:35,784 INFO __main__ - decode_steps: 32
2026-07-22 01:21:35,784 INFO __main__ - total_decode_ms: 657.401
2026-07-22 01:21:35,784 INFO __main__ - average_decode_ms: 20.544
2026-07-22 01:21:35,784 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:35,784 INFO __main__ - tokens_per_second: 194.706
2026-07-22 01:21:35,784 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:35,784 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:35,784 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:35,784 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:35,784 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:35,784 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:35,784 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:35,785 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:36,707 INFO __main__ - Benchmark result
2026-07-22 01:21:36,707 INFO __main__ - ----------------
2026-07-22 01:21:36,707 INFO __main__ - dtype: bf16
2026-07-22 01:21:36,708 INFO __main__ - batch_size: 4
2026-07-22 01:21:36,708 INFO __main__ - prompt_length: 64
2026-07-22 01:21:36,708 INFO __main__ - prefill_ms: 21.376
2026-07-22 01:21:36,708 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:36,708 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:36,708 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:36,708 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:36,708 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:36,708 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:36,708 INFO __main__ - decode_steps: 32
2026-07-22 01:21:36,708 INFO __main__ - total_decode_ms: 644.770
2026-07-22 01:21:36,708 INFO __main__ - average_decode_ms: 20.149
2026-07-22 01:21:36,708 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:36,708 INFO __main__ - tokens_per_second: 198.521
2026-07-22 01:21:36,709 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:36,709 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:36,709 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:36,709 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:36,709 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:36,709 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:36,709 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:36,709 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:37,631 INFO __main__ - Benchmark result
2026-07-22 01:21:37,631 INFO __main__ - ----------------
2026-07-22 01:21:37,631 INFO __main__ - dtype: bf16
2026-07-22 01:21:37,631 INFO __main__ - batch_size: 4
2026-07-22 01:21:37,631 INFO __main__ - prompt_length: 64
2026-07-22 01:21:37,631 INFO __main__ - prefill_ms: 20.839
2026-07-22 01:21:37,631 INFO __main__ - prefill_kv_length: 64
2026-07-22 01:21:37,631 INFO __main__ - first_next_token_ids: [4128, 4128, 4128, 4128]
2026-07-22 01:21:37,632 INFO __main__ - all_logits_finite: True
2026-07-22 01:21:37,632 INFO __main__ - final_decode_output next_token_ids: [13, 13, 13, 13]
2026-07-22 01:21:37,632 INFO __main__ - prefill_logits_finite: True
2026-07-22 01:21:37,632 INFO __main__ - decode_logits_finite: True
2026-07-22 01:21:37,632 INFO __main__ - decode_steps: 32
2026-07-22 01:21:37,632 INFO __main__ - total_decode_ms: 650.042
2026-07-22 01:21:37,632 INFO __main__ - average_decode_ms: 20.314
2026-07-22 01:21:37,632 INFO __main__ - generated_tokens: 128
2026-07-22 01:21:37,632 INFO __main__ - tokens_per_second: 196.910
2026-07-22 01:21:37,632 INFO __main__ - final_kv_length: 96
2026-07-22 01:21:37,632 INFO __main__ - memory_after_model_load_mib: 958.491
2026-07-22 01:21:37,632 INFO __main__ - baseline_allocated_mib: 958.491
2026-07-22 01:21:37,632 INFO __main__ - peak_allocated_mib: 1223.866
2026-07-22 01:21:37,632 INFO __main__ - incremental_over_model_load_mib: 265.375
2026-07-22 01:21:37,632 INFO __main__ - incremental_over_baseline_mib: 265.375
2026-07-22 01:21:37,633 INFO __main__ - kv_bytes: 4718592
2026-07-22 01:21:37,633 INFO __main__ - kv_mib: 4.500
2026-07-22 01:21:37,633 INFO root - Median prefill_ms for bf16, batch_size=4: 21.376
2026-07-22 01:21:37,633 INFO root - Median decode_ms for bf16, batch_size=4: 20.295
2026-07-22 01:21:37,633 INFO root - Median tok/s for bf16, batch_size=4: 197.095
2026-07-22 01:21:37,633 INFO root - Median peak memory for bf16, batch_size=4: 1223.866 MiB
2026-07-22 01:21:37,633 INFO root - Median KV memory for bf16, batch_size=4: 4.500 MiB
2026-07-22 01:21:37,633 INFO root - median_results: {('fp32', 1): {'prefill_ms': 20.722039975225925, 'decode_ms': 18.742686312180012, 'median_tok_per_s': 53.35414482982335, 'peak_memory_mb': 1998.73046875, 'kv_memory_mb': 2.25}, ('fp32', 2): {'prefill_ms': 20.690697943791747, 'decode_ms': 18.77418912044959, 'median_tok_per_s': 106.52923474716256, 'peak_memory_mb': 2100.42578125, 'kv_memory_mb': 4.5}, ('fp32', 4): {'prefill_ms': 22.247808054089546, 'decode_ms': 19.00506687525194, 'median_tok_per_s': 210.47018809540359, 'peak_memory_mb': 2310.16015625, 'kv_memory_mb': 9.0}, ('fp16', 1): {'prefill_ms': 20.89681220240891, 'decode_ms': 20.22299190866761, 'median_tok_per_s': 49.4486673641697, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('fp16', 2): {'prefill_ms': 20.95920406281948, 'decode_ms': 20.15549565840047, 'median_tok_per_s': 99.22851979908685, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}, ('fp16', 4): {'prefill_ms': 21.368002984672785, 'decode_ms': 20.27103506407002, 'median_tok_per_s': 197.32588826161697, 'peak_memory_mb': 1223.86572265625, 'kv_memory_mb': 4.5}, ('bf16', 1): {'prefill_ms': 20.705511095002294, 'decode_ms': 19.910303279175423, 'median_tok_per_s': 50.225252020441076, 'peak_memory_mb': 1024.15478515625, 'kv_memory_mb': 1.125}, ('bf16', 2): {'prefill_ms': 20.82672296091914, 'decode_ms': 20.164361274510156, 'median_tok_per_s': 99.18489223500511, 'peak_memory_mb': 1091.63134765625, 'kv_memory_mb': 2.25}, ('bf16', 4): {'prefill_ms': 21.376486867666245, 'decode_ms': 20.29477609175956, 'median_tok_per_s': 197.09505450637369, 'peak_memory_mb': 1223.86572265625, 'kv_memory_mb': 4.5}}
2026-07-22 01:21:37,792 INFO root - Benchmark log saved to: /home/ubuntu/CODE/mini_inference_runtime/logs/static_batch_benchmark.log
