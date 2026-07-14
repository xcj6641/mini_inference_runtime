Absolutely. I think Day 12 ended up being much more valuable than we originally planned. We didn't just "add KV cache"; we built a correct mental model of how modern LLM inference manages state.

---

# Day 12 — Real KV Cache Lifecycle

## Goal

Move from a fake KV cache represented by simple counters like:

```python
cached_tokens += 1
```

to the **real `past_key_values`** returned by Hugging Face, and understand its complete lifecycle during inference.

---

# Part 1. Request Owns the KV Cache

Previously, the KV cache only existed as a local variable inside `generate()`:

```python
past_key_values = prefill_output.past_key_values
```

After the function returned, the cache disappeared.

We redesigned the runtime so that the `Request` object owns the inference state.

```python
@dataclass
class Request:
    request_id: str
    input_ids: list[int]

    generated_ids: list[int]

    past_key_values: Any | None

    prompt_tokens: int
    generated_tokens_count: int

    state: RequestState
```

The cache lifecycle became:

```text
Request Created
        │
        ▼
past_key_values = None

        │
        ▼
Prefill

        │
        ▼
Attach KV Cache

        │
        ▼
Decode

        │
        ▼
Update KV Cache

        │
        ▼
Finished / Cancelled / Failed

        │
        ▼
Release KV Cache
```

This architecture closely matches real inference runtimes such as vLLM, where request state is maintained outside the model.

---

# Part 2. Request Lifecycle APIs

The `Request` class became responsible for managing its own inference state.

We implemented:

```python
attach_kv_cache()

append_generated_token()

release_kv_cache()

mark_finished()

cancel()

mark_failed()
```

Important design decisions:

* Generated tokens are preserved after generation finishes.
* Runtime-only state (`past_key_values`) is released.
* All terminal states (`FINISHED`, `FAILED`, `CANCELLED`) perform cleanup consistently.

---

# Part 3. Refactor Generation Pipeline

Instead of maintaining a local cache variable, `generate_request()` now operates directly on the `Request`.

Old design:

```text
generate()

↓

local past_key_values

↓

decode()

↓

return
```

New design:

```text
Request

↓

prefill()

↓

Request.attach_kv_cache()

↓

decode()

↓

Request.attach_kv_cache(updated_cache)

↓

Request.mark_finished()
```

This makes the request self-contained and schedulable.

---

# Part 4. Inspect Real KV Cache

We implemented `kv_cache_utils.py` to inspect the actual cache returned by the Transformer.

Key utilities include:

```python
inspect_legacy_kv_cache()

get_kv_sequence_length()

calculate_actual_kv_bytes()

estimate_kv_bytes_per_token()

estimate_total_kv_bytes()

get_cuda_memory_snapshot()
```

These utilities allow us to inspect:

* KV tensor shapes
* Sequence length
* Actual memory usage
* Estimated memory usage
* CUDA memory statistics

---

# Part 5. Understand the KV Cache Structure

A major learning outcome was understanding the hierarchical layout of the cache.

```
legacy_cache
│
├── Layer 0
│     ├── Key Tensor
│     └── Value Tensor
│
├── Layer 1
│     ├── Key Tensor
│     └── Value Tensor
│
└── ...
```

Each Key tensor has shape:

```
[batch_size,
 num_kv_heads,
 sequence_length,
 head_dim]
```

Important realization:

* `legacy_cache` organizes **Transformer layers**.
* Each layer contains **one Key tensor and one Value tensor**.
* The tensor dimensions describe the numerical layout inside one layer.

---

# Part 6. Correct Mental Model of Transformer Layers

One of the biggest conceptual breakthroughs was understanding that:

A Transformer layer does **not** correspond to a token.

Instead,

```
Prompt

Token1
Token2
Token3
...

↓

Layer 0

↓

Layer 1

↓

Layer 2

↓

...
```

Every layer processes **every token**.

Therefore,

every layer owns its own Key and Value tensors.

This directly explains why KV memory scales with the number of Transformer layers.

---

# Part 7. Understand KV Cache Growth

During prefill:

```
Prompt Length = N

↓

KV Cache Length = N
```

During each decode step:

```
Process one new token

↓

Append one Key

Append one Value

↓

KV Cache Length = N + 1
```

The cache stores **all processed tokens**, not merely the current input token.

This explains why:

```python
input_ids.shape == [1, 1]
```

during decode, while:

```python
KV sequence length
```

continues to grow.

---

# Part 8. Understand Model Statelessness

A crucial architectural concept was recognizing that the Transformer model itself is stateless between forward passes.

The model does **not** remember previous requests.

Instead:

```
Runtime

↓

stores past_key_values

↓

passes it back into

↓

model(...)
```

The cache exists because the runtime explicitly manages it.

---

# Part 9. Understand `use_cache=True`

We clarified an important misconception.

`use_cache=True` does **not** mean:

> Use an existing cache.

Instead, it means:

> This forward pass should participate in KV caching.

Two cases arise:

**Prefill**

```
No cache supplied

↓

Generate a new cache

↓

Return it
```

**Decode**

```
Existing cache supplied

↓

Reuse it

↓

Append new K/V

↓

Return updated cache
```

---

# Part 10. Understand Greedy Decoding

We also reviewed how the next token is selected.

```
Transformer

↓

logits

↓

last_token_logits

↓

argmax

↓

next_token_id

↓

tokenizer.decode()

↓

Human-readable text
```

The model predicts a token ID directly from logits using greedy decoding.

---

# Part 11. Testing

We added comprehensive unit tests covering both lifecycle management and KV inspection.

### Request lifecycle tests

Verified:

* initial request state
* attaching KV cache
* generated token updates
* cache release
* terminal states
* output preservation

### KV cache utility tests

Verified:

* cache shape inspection
* sequence length extraction
* byte calculation
* theoretical memory formula
* CUDA memory snapshots
* invalid input handling

Result:

```
38 tests passed
```

---

# Key Takeaways

By the end of Day 12, the project evolved from a simplified scheduler using token counters to a runtime capable of managing real Transformer state.

More importantly, the conceptual understanding changed significantly:

* A KV cache stores the Key and Value vectors for every processed token at every Transformer layer.
* The runtime—not the model—owns the cache.
* Prefill creates the initial cache, while each decode step extends it.
* The cache grows linearly with sequence length and scales with the number of Transformer layers.
* Real inference systems depend on explicit KV cache management to avoid recomputing attention history.

This provides the foundation for Day 13, where you'll move from understanding the KV cache lifecycle to **measuring its memory usage and performance impact** through benchmarks and profiling.
