Absolutely. Day 15 turned out to be more valuable than simply “implement a block manager,” because several problems forced us to reason about **resource planning, batching, padding, logical vs. physical KV state, and lifecycle management**.

# Day 15 — Real KV Block Manager + Scheduler Integration

## 1. What we wanted to accomplish

Before Day 15, KV cache was still primarily represented as:

```text
Request
    ↓
past_key_values
    ↓
HF contiguous KV tensors
```

We knew how many KV tokens existed, but we didn't have a real allocator answering questions such as:

> How much KV capacity does this request own?
> Can another request be admitted?
> What happens when KV capacity runs out?
> When a request finishes, how is memory returned?

Day 15 introduced:

```text
KVBlockManager
```

with fixed-size physical blocks.

For example:

```text
num_blocks = 8
block_size = 4 tokens
```

gives:

```text
physical KV blocks

0  1  2  3  4  5  6  7
```

A request maintains:

```python
request.block_table = [2, 5, 7]
```

meaning:

```text
logical block 0 → physical block 2
logical block 1 → physical block 5
logical block 2 → physical block 7
```

This is an important conceptual step toward paged KV.

---

# 2. KVBlockManager

We implemented the core allocator.

### Block requirement calculation

For block size `4`:

```text
tokens     blocks

0          0
1          1
4          1
5          2
8          2
9          3
```

using:

```python
(num_tokens + block_size - 1) // block_size
```

### Allocation

The manager maintains:

```python
_free_blocks
_owners
```

For example:

```text
initial:

free = [0, 1, 2, 3]
owners = {}
```

After A requests two blocks:

```text
A.block_table = [0, 1]

free = [2, 3]

owners:
0 → A
1 → A
```

We also made allocation failure atomic: if a request needs more blocks than are available, it receives **none**, rather than partially allocating and leaving corrupted state.

### Incremental growth

The manager doesn't allocate the entire request's possible future KV requirement upfront.

Instead:

```python
additional_blocks_required(
    request,
    total_tokens,
)
```

calculates only additional capacity.

Example:

```text
block_size = 4

A owns 1 block
capacity = 4 tokens

KV requirement = 3
→ +0 blocks

KV requirement = 4
→ +0 blocks

KV requirement = 5
→ +1 block
```

This gives us incremental KV growth during decoding.

### Batch reservation

We added:

```python
ensure_batch_capacity(...)
```

which first calculates the **whole batch requirement**, verifies capacity, and only then performs allocations.

That gives us:

```text
plan
↓
validate entire allocation
↓
commit
```

instead of partially allocating requests before discovering that the rest of the batch cannot fit.

---

# 3. Scheduler integration

This was the more interesting part of Day 15.

The scheduler now plans a tick roughly as:

```text
START TICK

        ↓

select decode requests
        ↓
capacity-aware decode planning
        ↓
reserve decode KV growth

        ↓

select prefill requests
        ↓
observe remaining KV capacity
        ↓
reserve prefill KV

        ↓

execute prefill
execute decode

        ↓

finish requests
free blocks

END TICK
```

The important policy is:

> **Active decode requests receive resource priority over new prefills.**

So if:

```text
3 blocks available

decode requires 2
prefill requires 2
```

we get:

```text
decode reserves 2

remaining = 1

prefill cannot be admitted
```

rather than allowing prefill to consume capacity that an already-active decode request needs.

---

# 4. The biggest issue we discovered: padding broke KV accounting

This was probably the most valuable debugging moment of Day 15.

Suppose:

```text
A prompt = 3 tokens
B prompt = 4 tokens
```

Batched prefill needs rectangular tensors:

```text
A: [PAD, a, b, c]
B: [d,   e, f, g]
```

So the HF KV tensors can both have:

```text
physical KV sequence length = 4
```

Originally, we used:

```python
get_kv_sequence_length(
    request.past_key_values
)
```

as the source of truth for block allocation.

That created a bug.

For A:

```text
real prompt tokens = 3
physical KV tensor length = 4
```

On its next decode, the allocator incorrectly reasoned:

```text
physical KV:
4 → 5

block_size = 4

1 block → 2 blocks required
```

But logically:

```text
real KV:
3 → 4

still only 1 block required
```

So A could incorrectly be rejected under memory pressure.

### The fix: `kv_tokens`

We introduced:

```python
request.kv_tokens
```

Now we explicitly distinguish:

```text
past_key_values.shape[seq]
        ↓
physical tensor representation
may contain padding


request.kv_tokens
        ↓
logical valid KV occupancy
does NOT contain padding
```

Lifecycle:

```text
new request
kv_tokens = 0

prefill
kv_tokens = prompt_tokens

decode
kv_tokens += 1

finish
kv_tokens = 0
```

And block allocation now uses:

```python
request.kv_tokens + 1
```

rather than physical HF tensor length.

This gave us a very important invariant:

```text
logical KV occupancy
        ≠
physical padded tensor length
```

---

# 5. Physical KV length still has a purpose

We didn't completely remove:

```python
get_kv_sequence_length(
    request.past_key_values
)
```

because our current HF implementation still needs physically compatible tensors for:

```python
build_equal_length_decode_batch(...)
```

So we ended Day 15 with a clean distinction:

```text
request.kv_tokens
    ↓
resource accounting / block allocation


physical past_key_values length
    ↓
current tensor batching compatibility
```

That distinction will become even more important when we move to paged KV.

---

# 6. KV lifecycle

We completed the whole lifecycle:

```text
Request admitted
      ↓
allocate blocks
      ↓
prefill
      ↓
DECODING
      ↓
grow blocks as necessary
      ↓
FINISHED
      ↓
free(request)
      ↓
blocks return to pool
      ↓
another request can reuse them
```

We specifically tested that:

```text
A owns block 0
B owns block 1

A finishes

block 0 → free
block 1 → still owned by B
```

and that a waiting request can later reuse block `0`.

---

# 7. Testing architecture

Another improvement was separating two types of tests.

### FakeRunner tests

Used for:

```text
scheduler policy
capacity pressure
allocation
free/reuse
EOS
batch selection
exact lifecycle behavior
```

They're fast and deterministic.

We updated the old FakeRunner from opaque fake KV such as dictionaries/`object()` to structurally valid tensor KV caches:

```text
layer
 ├─ key   [B, Hkv, S, D]
 └─ value [B, Hkv, S, D]
```

### PyTorchModelRunner integration tests

We kept a smaller set of real-model tests to verify:

```text
real HF KV
real padding
real prefill
real decode
logical kv_tokens
scheduler
block manager
```

In particular, we tested the exact padding scenario:

```text
A prompt = 3
B prompt = 4

physical HF KV:
A = 4
B = 4

logical:
A.kv_tokens = 3
B.kv_tokens = 4
```

and verified that A can decode without requiring another logical block while B requires one.

Finally:

```text
✓ integration tests green
✓ non-integration tests green
```

So we have a clean regression baseline before the next architectural change.

---

# Day 15 architecture at completion

We are currently here:

```text
                    ContinuousScheduler
                           │
              ┌────────────┴────────────┐
              │                         │
        scheduling                KVBlockManager
              │                         │
              │                  physical block IDs
              │                         │
              ▼                         ▼
           Request ─────────────── block_table
              │
              ├── kv_tokens
              │     logical KV occupancy
              │
              └── past_key_values
                    │
                    ▼
              HF KV tensors
              actual KV storage
```

The remaining limitation is obvious now:

> `block_table` manages **capacity**, but doesn't yet point to the actual KV storage.

That's exactly why the next phase is paged KV storage.

---

# BQ stories from Day 15

And yes — there are several good behavioral-interview stories here. I wouldn't make them all independent STAR stories, though. There are **three particularly useful stories**, each demonstrating a different engineering competency.

## BQ Story 1 — Discovered a resource-planning race inside one scheduler iteration

**Useful for:** “Tell me about a difficult bug,” “Tell me about a time you identified a design flaw,” “Tell me about a time you improved system reliability.”

**Situation:** While integrating a KV block manager into my mini LLM inference runtime, I realized that decode requests were selected based on available KV capacity before prefill execution. Prefill could then consume some of that capacity before decode actually ran.

**Task:** I needed to guarantee that active decode requests wouldn't be admitted based on stale capacity and then fail during execution.

**Action:** I traced the full scheduler iteration rather than treating allocation as a local function problem. I considered simply executing decode before prefill, but that would couple resource planning to execution and move away from the iteration-planning architecture I wanted. Instead, I changed the scheduler to plan decode first and reserve its required KV growth, then admit prefill using the remaining capacity. I also made batch allocation atomic so a failed reservation couldn't partially mutate allocator state.

**Result:** Decode and prefill now share a consistent view of KV capacity for each iteration. Active requests receive priority without requiring separate decode-first execution, and the test suite covers insufficient-capacity and partial-admission cases.

A concise interview version would sound like:

> “The interesting part was that the bug wasn't in allocation itself. Each allocation check was individually correct, but the scheduler was making decisions at different points against changing resource state. I fixed it by moving from check-then-execute behavior to iteration-level reservation: reserve active decode growth first, then admit prefill against the remaining KV capacity.”

That's a strong systems answer.

---

## BQ Story 2 — A passing abstraction turned out to be wrong because of padding

**Useful for:** “Tell me about a time your initial approach was wrong,” “Tell me about a subtle bug,” “Tell me about a time you challenged an assumption.”

**Situation:** Initially I used the sequence dimension of Hugging Face `past_key_values` as the request's KV length. That seemed natural because it represented the actual KV tensor produced by the model.

**Task:** While testing memory-aware decode admission with variable-length prompts, I found a case where a three-token request was rejected even though its existing KV block still had enough logical capacity.

**Action:** I traced the discrepancy and found that batched prefill had padded the three-token prompt to four positions. The physical KV tensor therefore had length four, even though only three positions represented meaningful request tokens. Instead of patching the allocator for padding, I changed the abstraction: I introduced a separate `kv_tokens` field representing logical KV occupancy. The scheduler uses `kv_tokens` for capacity planning, while physical KV tensor length is used only where tensor-shape compatibility matters.

**Result:** Block allocation became independent of padding artifacts. A request with three valid KV tokens correctly remains within one four-token block after its next decode, even when its physical HF KV tensor was padded to length four. I added both fake-runner and real PyTorch-model regression tests for this case.

The strongest sentence in this story is:

> “I realized the problem wasn't the formula for block allocation; the problem was that I had chosen the wrong source of truth.”

That demonstrates good abstraction/debugging judgment.

---

## BQ Story 3 — Migrating a test suite after an architectural change

**Useful for:** “Tell me about maintaining quality during a refactor,” “Tell me about testing,” “Tell me about dealing with legacy code.”

**Situation:** Adding the block manager and structured KV lifecycle changed several assumptions in my existing scheduler tests. Older fake runners represented KV as opaque dictionaries or `object()`, and some tests encoded scheduling behavior that was no longer valid.

**Task:** I wanted to preserve regression coverage without simply modifying assertions until the suite became green.

**Action:** I classified failures into three categories: outdated test setup, intentionally changed behavior, and actual production regressions. I upgraded the FakeRunner to return structurally valid batched KV tensors, added block-manager fixtures, updated expectations only where scheduler semantics intentionally changed, and kept a smaller real `PyTorchModelRunner` integration suite to validate real padding and KV behavior.

**Result:** Both the integration and non-integration suites were green again, with a clearer testing boundary: FakeRunner tests cover deterministic scheduling/resource policies, while real-runner tests validate compatibility with actual model/KV behavior.

This can be summarized in an interview as:

> “I didn't treat a failing old test as automatically meaning either the code or test was wrong. For each failure I asked whether the contract had changed, whether the test double no longer represented production, or whether I had introduced a regression.”

That's a very good testing/refactoring story.

## BQ Story 4 — Improving Decode Batch Selection Under KV Pressure

**Situation**

While building a continuous scheduler for a mini LLM inference runtime, I added memory-aware decode scheduling. Because my current Hugging Face-based implementation stores per-request KV caches as contiguous tensors, requests in the same decode batch need compatible physical KV sequence lengths so their caches can be stacked.

My first implementation used the first active decode request to establish the batch's target KV length.

**Task**

I needed the scheduler to maximize useful GPU work under limited KV-cache capacity while still respecting the equal-physical-KV-length constraint of the current decode batch builder.

**Action**

While reviewing the selection logic, I noticed a subtle inefficiency.

Suppose the scheduler sees:

```text
Request A:
physical KV length = 4
needs one additional KV block
cannot fit

Request B:
physical KV length = 5
needs no additional block
can fit
```

My original code allowed Request A to set:

```python
target_kv_length = 4
```

before checking whether A could actually obtain the KV capacity required for its next decode.

A was then rejected because of memory pressure, but its physical KV length had already constrained the batch. Request B was skipped because its physical KV length was 5, even though B was perfectly runnable.

I changed the order of operations so the scheduler first:

1. validates the request state and KV cache,
2. calculates its next logical KV requirement using `kv_tokens`,
3. checks whether the required KV blocks are available,
4. admits the request,
5. and only then, if it is the first admitted request, sets `target_kv_length`.

Conceptually, I changed the policy from:

```text
first candidate
→ determines batch shape
→ check whether it can actually run
```

to:

```text
first runnable candidate
→ determines batch shape
```

I also kept logical and physical KV state separate: `kv_tokens` determines memory admission, while physical KV tensor length is used only for compatibility with the current Hugging Face decode batching path.

**Result**

An unschedulable request can no longer prevent a later runnable request from forming a decode batch simply because it appeared first.

This improved scheduler utilization under KV pressure and gave me a more general scheduling principle:

> Do not commit batch-level constraints based on work that has not actually been admitted.

It also reinforced an architectural distinction that became important throughout the project: logical KV occupancy should drive resource accounting, while physical tensor shape should only constrain the execution mechanism that currently requires it.


---

## Which story is strongest?

For an AI Infra / inference role, I'd rank them:

**#1 Padding → logical vs. physical KV distinction** is probably the strongest technically. It lets you naturally discuss batching, attention masks, KV cache, block allocation, padding waste, and why production paged-KV systems maintain per-sequence metadata separately from physical storage.

**#2 Decode reservation before prefill** is strongest for scheduler/system-design questions. It demonstrates resource consistency, admission control, active-request priority, and iteration planning.

**#3 Test migration** is a good secondary BQ for engineering-quality questions.

More importantly, Day 15 gave you a coherent project narrative:

```text
I started with HF past_key_values
        ↓
introduced explicit KV block ownership
        ↓
integrated allocation with scheduling
        ↓
encountered resource-planning inconsistency
        ↓
introduced iteration-level reservation
        ↓
encountered padding corrupting memory accounting
        ↓
separated logical KV occupancy from physical tensors
        ↓
implemented allocation/free/reuse lifecycle
        ↓
validated with fake + real model tests
        ↓
next: make block_table map to actual paged KV storage
```

That narrative is much stronger than saying, “I implemented a `KVBlockManager` class.” It shows **why each abstraction exists and which concrete failure forced the architecture to evolve**.
