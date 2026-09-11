Day 13 was a **real inference-runtime milestone**, not just “adding batching.” You implemented the full path, found measurement problems, corrected them, and used repeated experiments to reach defensible conclusions.

## Day 13 summary

You completed four major parts.

First, you built **static batched prefill**. Multiple requests with equal and variable prompt lengths were combined into tensors such as:

```text
input_ids:      [batch_size, max_prompt_length]
attention_mask: [batch_size, max_prompt_length]
```

You handled padding, request-to-batch mapping, last-token logits, and verified that batched prefill produced the same greedy next tokens as individual prefill.

Second, you implemented **batched decode with real KV cache**. Each decode step used:

```text
input_ids: [batch_size, 1]
```

while preserving each request’s KV-cache state and attention-mask length. You verified that KV sequence length grew from 64 after prefill to 96 after 32 decode steps, for every batch size and dtype. 

Third, you added **mixed-precision support and validation** for:

```text
FP32
FP16
BF16
```

You checked capability support, finite logits, output batch sizes, deterministic token IDs, and real KV-cache tensor memory.

Fourth, you built a proper **benchmarking framework** with:

* warm-up for prefill and decode;
* CUDA synchronization around timing;
* batch sizes 1, 2, and 4;
* five repetitions per configuration;
* median latency and throughput;
* model memory, peak memory, and KV-cache memory;
* correctness assertions;
* structured logging and result objects. 

## Main experimental results

Static batching produced nearly linear throughput scaling:

```text
FP32:  53.7 → 108.1 → 213.3 tokens/s
FP16:  50.7 → 101.0 → 201.0 tokens/s
BF16:  51.2 → 101.6 → 201.0 tokens/s
```

Batch size increased from 1 to 4, but per-step decode latency stayed around 18.5–19.9 ms. That showed that batching increased the amount of useful work completed per forward pass without multiplying latency proportionally.

Mixed precision reduced memory dramatically:

```text
Model memory:
FP32: 1885 MiB
FP16:  955 MiB
BF16:  957 MiB
```

KV-cache memory was exactly halved:

```text
Batch 4:
FP32: 9.0 MiB
FP16: 4.5 MiB
BF16: 4.5 MiB
```

You also learned an important systems lesson: FP16/BF16 did not automatically improve latency for this small Hugging Face inference workload. Their clear advantage here was memory efficiency, while FP32 remained slightly faster in decode.

Most importantly, you found that the original FP16 batch-4 slowdown was an outlier. The first run showed around 30.6 ms prefill, but the five-run median was about 21.1 ms. That demonstrated why a benchmark should not rely on one measurement.

---