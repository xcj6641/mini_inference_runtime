I think Day 14 is a major milestone in the project. Compared with Day 13, we are no longer just proving that **batched inference works**. We have actually connected our **scheduler** with the **real model runtime**, making the whole runtime execute batched requests end-to-end. This is the first time our runtime behaves like a simplified production inference engine.

---

# Day 14 — Real Scheduler Integration & Batched Decode

## Goal

Replace the fake runtime with the real `PyTorchModelRunner`, allowing the scheduler to perform batched prefill and batched decode while managing each request's KV cache lifecycle correctly.

---

# What We Built

## Part 1 — Real Scheduler → Real Runner Integration

Previously:

```
Scheduler
      │
      ▼
FakeRunner
      │
fake logits
      │
fake KV
```

Now:

```
Scheduler
      │
      ▼
BatchBuilder
      │
      ▼
PyTorchModelRunner
      │
      ▼
Qwen2.5-0.5B
      │
      ▼
real logits
real past_key_values
```

The scheduler now drives the actual Hugging Face model instead of a simulated backend.

---

## Part 2 — Real Batched Prefill

Implemented:

```
waiting requests
        │
        ▼
build_prefill_batch()
        │
        ▼
prefill_batch()
        │
        ▼
batched KV cache
        │
        ▼
split_legacy_kv_cache()
        │
        ▼
Request.attach_kv_cache()
```

Each request now owns its own KV cache after the batched forward pass.

We verified:

* batched next-token generation
* correct KV splitting
* independent request-local caches
* scheduler state transition

```
WAITING
    │
    ▼
DECODING
```

---

## Part 3 — Real Batched Decode

Implemented:

```
active requests
        │
        ▼
build_equal_length_decode_batch()
        │
        ▼
decode_batch()
        │
        ▼
updated batched KV cache
        │
        ▼
split_legacy_kv_cache()
        │
        ▼
attach updated KV cache
        │
        ▼
append generated token
```

Every decode step now updates

* generated token
* KV cache
* request state

using the real model outputs.

---

## Part 4 — Real KV Cache Lifecycle

The scheduler now manages the complete KV lifecycle.

```
prefill
      │
      ▼
attach KV cache
      │
      ▼
decode
      │
      ▼
replace KV cache
      │
      ▼
finish
      │
      ▼
release KV cache
```

When a request finishes:

```
active
    │
    ▼
completed

past_key_values = None
```

Memory is explicitly released.

---

## Part 5 — Decode Batch Selection

One important discovery during development:

Legacy Hugging Face KV caches cannot batch together requests whose physical KV sequence lengths differ.

Instead of implementing padded variable-length decode (which production systems avoid), we introduced a scheduler policy:

```
active requests
        │
        ▼
group by physical KV length
        │
        ▼
select compatible requests
        │
        ▼
decode together
```

This keeps the runtime correct while preparing for future paged KV cache support.

---

## Part 6 — Request Lifecycle

The scheduler now owns the complete request lifecycle.

```
WAITING
    │
    ▼
PREFILL
    │
    ▼
DECODING
    │
    ▼
FINISHED
```

Completion can occur because of

* EOS token
* max_new_tokens

After completion:

```
remove from active

↓

release KV cache

↓

move to completed
```

---

# Testing

## Real Scheduler Integration

Verified

* scheduler → BatchBuilder
* BatchBuilder → Runner
* Runner → Scheduler

---

## Batched Prefill

Verified

* multiple requests execute together
* KV cache correctly split
* request-local KV ownership

---

## Batched Decode

Verified

* batched decode
* KV cache updated
* KV length grows by one every decode

---

## Completion

Verified

* finish by length
* finish cleanup
* KV release
* active → completed transition

---

## Mixed Request Lifecycle

Verified

```
Request A finishes

Request B continues decoding
```

The scheduler correctly shrinks the active decode batch after one request completes.

---

# Architecture After Day 14

```
                  ContinuousScheduler
                          │
          ┌───────────────┴───────────────┐
          │                               │
     Waiting Queue                 Active Requests
          │                               │
          └───────────────┬───────────────┘
                          ▼
                    BatchBuilder
                          │
          ┌───────────────┴───────────────┐
          │                               │
      Prefill Batch                 Decode Batch
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 PyTorchModelRunner
                          │
                          ▼
                 Hugging Face Model
                          │
                          ▼
                 Batched KV Cache
                          │
                          ▼
             split_legacy_kv_cache()
                          │
                          ▼
                 Request-local KV Cache
```

---

# Behavioral Interview Stories

### 1. Integrating the scheduler with a real runtime

> I replaced a fake inference backend with a real Hugging Face runtime while preserving the scheduler interface. The biggest challenge was that batched outputs had to be split back into independent request-local KV caches, ensuring each request could continue decoding independently across scheduler ticks.

---

### 2. Discovering decode batching constraints

> During implementation, I found that batched decoding was fundamentally different from batched prefill. Legacy KV caches require identical physical sequence lengths, so requests with different KV lengths cannot simply be concatenated. Instead of implementing an artificial padding solution, I introduced scheduler-side grouping by compatible KV length. This mirrors a real engineering tradeoff and naturally motivates why systems like vLLM adopt paged KV caches.

---

### 3. Designing a complete KV cache lifecycle

> I implemented the entire KV cache lifecycle: attaching caches after prefill, updating them after every decode step, releasing them when requests finish, and validating each stage with integration tests. This helped ensure both correctness and predictable GPU memory management.

---

# Why Day 14 Matters

Day 14 is where the project transitions from **building inference primitives** to **building an inference runtime**.

Before Day 14, you had individual components—a scheduler, batch builder, model runner, and KV cache utilities—that worked in isolation. After Day 14, those components operate together as a coherent runtime capable of managing multiple real inference requests end-to-end.

More importantly, this stage exposes a limitation of the traditional Hugging Face `past_key_values` representation: efficient batched decoding depends on compatible dense KV layouts. Recognizing that limitation sets the stage for the next evolution of the project.

**Day 15 is no longer about adding another feature—it is about replacing the underlying memory management model.** The introduction of a real Block Manager will decouple logical sequences from dense tensor layouts, laying the groundwork for paged KV cache, prefix cache, and continuous scheduling techniques used in production systems like vLLM. This is the point where your mini runtime starts converging toward the architecture of a modern LLM inference engine.
