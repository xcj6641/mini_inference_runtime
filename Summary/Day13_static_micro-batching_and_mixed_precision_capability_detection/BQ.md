
# Possible behavioral-question stories

## 1. “Tell me about a technically challenging project.”

In my mini LLM inference runtime project, I implemented static micro-batching with real KV-cache reuse and mixed-precision benchmarking.

The difficult part was that batching was not simply concatenating requests. During prefill, I had to construct padded input tensors and attention masks, preserve the mapping between requests and batch positions, and read the correct last-token logits for each sequence. During decode, every request contributed only one new token, but its attention mask and KV cache had to grow correctly at every step.

I implemented batched prefill and decode for batch sizes one, two, and four, then added FP32, FP16, and BF16 execution. I created correctness checks for tensor shapes, finite logits, output batch size, deterministic greedy tokens, and KV-cache sequence length.

Finally, I built a benchmark with warm-up, CUDA synchronization, peak-memory tracking, five repetitions, and median aggregation.

The results showed almost four-times throughput improvement when batch size increased from one to four, while decode latency remained nearly constant. FP16 and BF16 also reduced model and KV-cache memory by approximately half.

The project helped me understand batching not only as an API feature, but as a GPU-utilization and memory-management problem.

## 2. “Tell me about a time you found a misleading result.”

While benchmarking FP16 inference, I initially observed that batch size four was significantly slower than batch sizes one and two. Prefill latency increased from around 20 milliseconds to more than 30 milliseconds, and decode throughput dropped noticeably.

My first reaction was that FP16 might be selecting an inefficient GPU kernel for that shape. However, I realized that the benchmark used only one measured execution per configuration, so the conclusion was not statistically reliable.

I changed the benchmark to run every dtype and batch-size combination five times and report the median instead of relying on a single result. I also retained the individual samples for comparison.

After rerunning the experiment, FP16 batch-four prefill had a median of about 21 milliseconds, not 30 milliseconds. The earlier result was an outlier, likely caused by temporary GPU or runtime variation.

The lesson was that an apparently convincing performance result can still be wrong if the measurement methodology is weak. I improved the benchmark before trying to explain the result, rather than constructing a theory around noisy data.

This is one of your strongest stories because it demonstrates:

```text
skepticism
measurement discipline
debugging
avoiding confirmation bias
```

## 3. “Tell me about a time AI gave you an incorrect or incomplete answer.”

During my inference-runtime project, I used AI as a technical discussion partner while analyzing an FP16 benchmark result. The initial result showed that FP16 batch size four was much slower than expected, and one possible explanation was that the GPU had selected a less favorable kernel path.

I did not accept that explanation as a conclusion because the evidence consisted of only one run. I recognized that the benchmark itself was not robust enough to distinguish a real kernel issue from timing noise.

I added five repetitions for every configuration, calculated median prefill latency, median decode latency, and median throughput, and reran the full FP32, FP16, and BF16 matrix.

The repeated results showed that the FP16 slowdown was not reproducible. The earlier measurement was an outlier.

This experience reinforced how I use AI: it can generate useful hypotheses, but I treat those hypotheses as starting points. For technical decisions, I verify them with code, logs, assertions, and controlled experiments.

This is an excellent answer to:

> Have you ever found that AI made a mistake?

It does not attack AI. It shows mature use of AI as a hypothesis generator rather than an authority.

## 4. “Tell me about a time you improved an existing solution.”

My first benchmark implementation measured one execution for each batch size and dtype. It successfully produced latency, throughput, and memory numbers, but I realized that the result was not yet reliable enough for comparison.

I improved it in several stages.

First, I added a warm-up that executed both prefill and decode, because those paths use different tensor shapes and may trigger different initialization behavior.

Second, I placed CUDA synchronization around timed sections so that asynchronous GPU execution would not make the CPU timer inaccurate.

Third, I separated model-load memory, pre-run baseline memory, peak allocated memory, and actual KV-cache memory.

Fourth, after observing an unusual FP16 result, I added five repetitions and median aggregation.

The revised benchmark produced stable measurements across nine configurations: three dtypes and three batch sizes. It showed nearly linear throughput scaling and exact halving of KV-cache memory under FP16 and BF16.

The improvement was not a new model feature, but it transformed a demonstration script into a benchmark whose conclusions I could defend.

## 5. “Tell me about a time you paid attention to details.”

A concise version:

When implementing batched decode, I noticed that using a hard-coded request ID such as `benchmark-0` would still allow the model forward pass to run, but it would make the batch metadata incorrect for batch sizes greater than one.

I changed the decode batch to carry the complete `batch.request_ids` list and added assertions that the number of request IDs, next-token IDs, and tensor rows all matched the batch size. I also verified that request IDs were unique.

I added KV-cache assertions requiring the prefill length to equal the prompt length and the final length to equal the prompt length plus the number of decode steps.

These checks did not change the model’s arithmetic, but they prevented silent metadata corruption and ensured that the runtime remained correct as batching scaled beyond one request.

## 6. “Tell me about a trade-off you made.”

In my inference-runtime project, I compared FP32, FP16, and BF16 execution.

The common assumption is that lower precision should always be faster, but my measurements showed a more nuanced trade-off. FP16 and BF16 reduced model memory from about 1.9 GiB to about 0.96 GiB and cut KV-cache memory exactly in half. However, for the small model, short sequence length, small batches, and general Hugging Face execution path I used, their decode latency was slightly worse than FP32.

Instead of selecting a dtype based only on theoretical FLOPS, I evaluated both speed and memory.

My conclusion was that FP16 and BF16 were valuable primarily because they increased memory capacity and would allow more concurrent sequences or larger models, even though they did not improve latency in this particular setup.

The experience taught me to evaluate systems trade-offs using workload-specific measurements rather than relying only on general hardware expectations.

## Strongest three stories for interviews

I would prioritize these:

1. **Misleading FP16 result** — strongest debugging and experimental-method story.
2. **Building batched prefill/decode** — strongest technical-complexity story.
3. **AI proposed a hypothesis, but I verified it** — strongest AI-mistake story.

Day 13 gave you much more than one resume bullet. It gave you several stories about implementation, benchmarking, skepticism, memory trade-offs, and debugging—the kinds of stories that make the project sound like engineering rather than coursework.
