# Architecture

You have now demonstrated the complete lifecycle:

```text
Request arrives
    ↓
KVBlockManager allocates physical blocks
    ↓
request.block_table records mapping
    ↓
Prefill produces temporary contiguous KV
    ↓
valid KV written into PagedKVCache
    ↓
PAD KV discarded
    ↓
Decode:
PagedKVCache materializes logical KV
    ↓
different lengths temporarily left-padded
    ↓
HF-compatible DecodeBatch
    ↓
model produces updated KV
    ↓
temporary padding discarded again
    ↓
updated logical KV written to PagedKVCache
    ↓
request finishes
    ↓
KVBlockManager frees blocks
    ↓
physical blocks can safely be reused
```

And importantly, you've removed:

```python
request.past_key_values
```

as persistent request state.

So the architecture has genuinely changed from:

```text
Request
└── past_key_values   ← actual KV storage

BlockManager
└── block_table       ← capacity bookkeeping
```

to:

```text
Request
├── kv_tokens
└── block_table
       │
       ▼
PagedKVCache
└── actual persistent KV storage
```

There's still one intentional inefficiency:

```text
PagedKVCache
    ↓ materialize
contiguous KV
    ↓ pad
HF model
```

A production PagedAttention implementation would ideally read the physical blocks directly:

```text
Q
+
block_table
+
PagedKVCache
    ↓
PagedAttention kernel
```

without reconstructing a contiguous KV tensor every decode step.

That's not a Day 16 bug. It's the boundary we deliberately chose so you could build the **memory-management architecture first** while still using the Hugging Face model implementation.

I recommend the next thing we do is **Day 16 summary**, rather than immediately starting Day 17. There are several interview-important concepts worth consolidating now: why models having KV cache doesn't eliminate runtime KV management, logical vs physical KV, why blocks contain all layers/heads for token slots, padding, block reuse, materialization, and exactly how our implementation differs from production PagedAttention.

After that, we can move into **Day 17 Prefix Cache**, where `block_table` becomes even more useful because multiple requests can start by referencing shared physical prefix blocks.

# Detail
Absolutely. Day 16 was one of the most important days in the project because we changed the **ownership model of KV cache**, rather than merely adding another class.

# Day 16 — Real Paged KV Cache Summary

## 1. What was the architectural goal?

At the beginning of Day 16, our architecture was effectively:

```text
Request
├── past_key_values       ← actual persistent KV data
├── kv_tokens
└── block_table           ← only says "I reserved these blocks"
```

So although Day 15 gave us a real `KVBlockManager`, the blocks didn't actually contain KV data.

The request still owned a large contiguous Hugging Face KV tensor.

Day 16 changed this to:

```text
Request
├── kv_tokens
├── block_table
└── generated_ids
        │
        │ block IDs
        ▼
PagedKVCache
┌─────────────────────────────┐
│ physical block 0            │
│ physical block 1            │
│ physical block 2            │
│ ...                         │
└─────────────────────────────┘
        ↑
actual persistent K/V
```

So the central achievement is:

> **`block_table` now points to real KV storage.**

And `Request.past_key_values` is no longer persistent runtime state.

---

# 2. What exactly is `PagedKVCache`?

We created a shared KV pool conceptually containing two 5-D tensors:

```python
key_cache.shape = (
    num_layers,
    num_blocks,
    num_kv_heads,
    block_size,
    head_dim,
)

value_cache.shape = (
    num_layers,
    num_blocks,
    num_kv_heads,
    block_size,
    head_dim,
)
```

For example:

```text
num_layers   = 24
num_blocks   = 100
num_kv_heads = 2
block_size   = 16
head_dim     = 64
```

means:

```text
K: [24, 100, 2, 16, 64]
V: [24, 100, 2, 16, 64]
```

A lookup like:

```python
key_cache[layer, block, head, slot, :]
```

returns one K vector:

```text
[head_dim]
```

We spent quite a bit of time clarifying what a "block" actually means.

The important mental model is:

> A physical block contains `block_size` **token positions**, but each token position contains KV vectors across all layers and KV heads.

So if:

```text
block_size = 4
block_table = [2, 9]
```

then:

```text
logical token 0 → physical block 2 / slot 0
logical token 1 → physical block 2 / slot 1
logical token 2 → physical block 2 / slot 2
logical token 3 → physical block 2 / slot 3

logical token 4 → physical block 9 / slot 0
logical token 5 → physical block 9 / slot 1
```

But block 9 / slot 1 doesn't contain only:

```text
layer 0 / head 0
```

It conceptually contains:

```text
token 5
├── layer 0
│   ├── head 0 → K vector + V vector
│   ├── head 1 → K vector + V vector
│   └── ...
├── layer 1
│   ├── head 0 → K vector + V vector
│   └── ...
└── ...
```

This resolved one of the biggest conceptual confusions we had during Day 16.

---

# 3. Token → block addressing

We implemented the fundamental paged mapping:

```python
logical_block_index = (
    token_index // block_size
)

slot_index = (
    token_index % block_size
)

physical_block_id = (
    block_table[logical_block_index]
)
```

For:

```text
block_size = 4
block_table = [7, 2, 5]
```

we get:

```text
token 0 → block 7 / slot 0
token 3 → block 7 / slot 3

token 4 → block 2 / slot 0
token 7 → block 2 / slot 3

token 8 → block 5 / slot 0
```

This mapping is important because it's also the basic idea a real PagedAttention implementation needs.

---

# 4. Writing real KV into blocks

We implemented:

```python
write_request_kv(...)
```

which converts contiguous HF-style KV:

```text
[layer]
K/V:
[1, num_kv_heads, seq_len, head_dim]
```

into our physical paged representation.

We tested:

```text
✓ fewer than one block
✓ exactly one block
✓ crossing a block boundary
✓ multiple layers
✓ multiple KV heads
✓ K and V separately
```

For example:

```text
6 tokens
block_size = 4
block_table = [2, 9]
```

becomes:

```text
block 2:
slot 0 ← token 0
slot 1 ← token 1
slot 2 ← token 2
slot 3 ← token 3

block 9:
slot 0 ← token 4
slot 1 ← token 5
```

---

# 5. Materializing paged KV

Because we're still using Hugging Face attention, the model doesn't directly consume:

```text
block_table + PagedKVCache
```

So we implemented the reverse operation:

```python
materialize_request_kv(...)
```

which converts:

```text
PagedKVCache
+
block_table
+
kv_tokens
```

back into:

```text
[layer]
K/V [1, H_kv, seq_len, D]
```

We then proved the round trip:

```text
contiguous KV
      ↓
write_request_kv()
      ↓
PagedKVCache
      ↓
materialize_request_kv()
      ↓
same contiguous KV
```

using:

```python
torch.testing.assert_close(...)
```

---

# 6. Why do we need PagedKVCache if Transformer models already support KV cache?

This was one of the most valuable conceptual discussions.

The distinction we arrived at was:

### Transformer/model responsibility

The model knows how to **compute K and V** and how to use previously computed K/V during attention.

An interface such as:

```python
past_key_values=...
```

allows the caller to provide previous KV.

### Inference runtime responsibility

The runtime must decide:

```text
Where does KV live?
How much GPU memory is available?
Which request owns which memory?
When should blocks be allocated?
When should they be freed?
How can freed memory be reused?
How can prefixes eventually be shared?
How do thousands of requests coexist?
```

A basic per-request contiguous KV approach can work functionally, but it doesn't provide the serving-level memory-management architecture we need.

So our interview answer became roughly:

> Transformer models implement KV-cache computation and consumption, but a production inference runtime needs to manage KV cache as a shared GPU-memory resource across many concurrent requests. A paged KV cache gives the runtime explicit allocation, reuse, non-contiguous growth, and eventually sharing of physical KV blocks.

---

# 7. Why allocate one big pool?

We also clarified why this is useful compared with continually creating and destroying per-request tensors.

Conceptually:

```text
Startup:

CUDA/GPU allocation
       ↓
large PagedKVCache pool
       ↓
[block0][block1][block2]...[blockN]
```

During serving:

```text
Request A → blocks 2, 7
Request B → blocks 3, 9, 12
Request C → block 5
```

When A finishes:

```text
blocks 2, 7
→ returned to free pool
→ future request can reuse them
```

Instead of treating every request's entire KV as a separately managed long-lived contiguous allocation.

---

# 8. The padding problem

This became the hardest integration issue of Day 16.

Suppose:

```text
A prompt length = 3
B prompt length = 6
```

Batched HF prefill uses left padding:

```text
A: [PAD PAD PAD A0 A1 A2]
B: [B0  B1  B2  B3 B4 B5]
```

The model's physical KV output has:

```text
seq_len = 6
```

for **both** requests.

But logically:

```text
A.kv_tokens = 3
B.kv_tokens = 6
```

We realized that `split_legacy_kv_cache()` only splits the **batch dimension**. It does not remove padding.

So we added:

```python
physical_kv_length = (
    per_request_cache[0][0].shape[2]
)

source_start = (
    physical_kv_length
    - request.kv_tokens
)
```

For A:

```text
physical = 6
logical  = 3

source_start = 3
```

and persist only:

```text
A0 A1 A2
```

not:

```text
PAD PAD PAD A0 A1 A2
```

Therefore our persistent cache obeys:

> **PagedKVCache contains logical KV only. Temporary padding is not persistent runtime state.**

---

# 9. The variable-length decode problem

This was probably the most important design issue we discovered.

Originally decode batching required requests to have the same physical KV length:

```python
if len(kv_lengths) != 1:
    raise ValueError(...)
```

That worked when requests retained padded `past_key_values`.

But once we correctly removed padding from `PagedKVCache`:

```text
A materialized KV = length 3
B materialized KV = length 6
```

we could no longer simply stack them.

At first this looked like a contradiction:

```text
Paged KV removes padding
        ↓
different sequence lengths
        ↓
HF batched attention wants rectangular tensors
```

We considered three choices:

```text
1. Only batch equal-length requests
2. Implement real PagedAttention immediately
3. Temporarily re-pad materialized KV
```

We chose **#3**.

That was the right choice for the current project.

---

# 10. Temporary left-padding during decode

Our current transitional decode architecture is:

```text
PagedKVCache
      ↓
materialize logical KV

A length 3
B length 6
      ↓
BatchBuilder temporarily left-pads

A: [PAD PAD PAD A0 A1 A2]
B: [B0  B1  B2  B3 B4 B5]
      ↓
HF model
```

We build the corresponding:

```text
attention_mask
position_ids
```

For A/B:

```text
attention mask:

A: 0 0 0 1 1 1 | 1(new decode token)
B: 1 1 1 1 1 1 | 1(new decode token)
```

and the current decode token gets its **logical** position:

```text
A → position 3
B → position 6
```

rather than both receiving position 6 just because their temporary tensors have equal physical lengths.

---

# 11. Attention mask vs position IDs

We also clarified that these aren't redundant.

`attention_mask` answers:

> **Which token positions should attention treat as valid?**

For example:

```text
[0, 0, 0, 1, 1, 1, 1]
```

means the temporary PAD positions should not contribute.

`position_ids` answers:

> **What logical sequence position does this token have?**

For A:

```text
logical old KV = 3
current decode token position = 3
```

For B:

```text
logical old KV = 6
current decode token position = 6
```

So:

```text
attention_mask → validity
position_ids   → position
```

Different jobs.

---

# 12. Decode write-back

After HF decode, both requests again have the same physical padded KV length.

For:

```text
A old logical = 3
B old logical = 6
```

after decode:

```text
A new logical = 4
B new logical = 7

physical returned length = 7 for both
```

For A:

```text
source_start
= 7 - 4
= 3
```

So:

```text
[PAD PAD PAD A0 A1 A2 A3]
             ↓
PagedKVCache stores
[A0 A1 A2 A3]
```

For B:

```text
source_start
= 7 - 7
= 0
```

and all seven positions are valid.

Thus temporary padding never becomes persistent.

---

# 13. Block-boundary growth

We connected Day 15's allocator with Day 16's actual storage.

For:

```text
block_size = 4
kv_tokens = 4
```

the request owns one block:

```text
block_table = [2]
```

Next decode requires:

```text
5 KV tokens
```

so the scheduler asks the block manager for additional capacity:

```text
blocks_required(5) = 2
```

and gets:

```text
block_table = [2, 7]
```

The fifth token maps to:

```text
token_index = 4

logical_block = 4 // 4 = 1
slot          = 4 % 4  = 0

physical block
= block_table[1]
= 7
```

We tested that the actual new K/V lands in:

```text
block 7 / slot 0
```

This proved:

```text
scheduler
+
KVBlockManager
+
block_table
+
PagedKVCache
```

are now one functioning system.

---

# 14. Real-model validation

We didn't stop at `FakeRunner`.

Using Qwen through `PyTorchModelRunner`, we proved:

```text
real prefill KV
      ↓
PagedKVCache.write()
      ↓
materialize()
      ↓
identical to original
```

and then:

```text
real prefill
↓
paged storage
↓
materialize
↓
real decode
↓
KV length N + 1
↓
write back
↓
materialize
↓
matches real model output
```

This is important because it proves the storage representation isn't merely compatible with our fake tensors.

---

# 15. Removing `Request.past_key_values`

Once equivalence was established, we removed the transitional architecture.

Before:

```text
Request
├── past_key_values
├── kv_tokens
└── block_table
```

Now:

```text
Request
├── input_ids
├── generated_ids
├── kv_tokens
├── block_table
└── state
```

There are **no model tensors inside Request** anymore.

`past_key_values` still exists, but only temporarily:

```text
PagedKVCache
↓
materialize
↓
past_key_values
↓
HF model
↓
updated past_key_values
↓
PagedKVCache
```

That is a model-interface representation, not persistent request state.

---

# 16. Resource ownership cleanup

We found another subtle design problem during cleanup.

Previously:

```python
Request.release_kv_cache()
```

could do:

```python
self.block_table.clear()
```

But the `KVBlockManager` also tracks:

```text
physical block → owner
```

So clearing `block_table` directly could produce:

```text
Request:
block_table = []

BlockManager:
block 2 → still owned by A
block 7 → still owned by A
```

That's a resource leak.

We established a much cleaner ownership rule:

> **The object that owns allocation state must perform the free operation.**

Therefore:

```text
Request
→ records block IDs

KVBlockManager
→ allocates them
→ owns free-list metadata
→ frees them
→ clears request.block_table
```

`Request.mark_finished()` should only change logical request state.

---

# 17. Why don't we reset `kv_tokens`?

Another useful distinction emerged.

When a request finishes:

```text
block_table
```

represents physical resources, so it must be cleared after those resources are freed.

But:

```text
kv_tokens
```

is just logical/history information.

So a finished request might retain:

```text
prompt_tokens = 10
generated_tokens_count = 5
kv_tokens = 14

block_table = []
state = FINISHED
```

Keeping `kv_tokens` can help metrics/debugging without retaining GPU resources.

---

# 18. Block reuse test

Finally, we proved:

```text
A
↓
gets physical block 0
↓
writes A KV
↓
finishes
↓
block 0 freed
↓
B gets block 0
↓
writes B KV
↓
materialize B
↓
only B's valid KV is observed
```

We also learned that we don't necessarily need to zero every freed block.

If B has:

```text
kv_tokens = 2
```

only its valid two slots are materialized.

Old bytes in unused slots don't become part of B's logical KV.

---

# Issues we encountered and what they taught us

Here are the major problems from Day 16.

| Issue                                           | Root cause                                  | Solution                                                 | Lesson                                                    |
| ----------------------------------------------- | ------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| Confusion about block/slot across heads/layers  | Thinking a slot stores one head vector      | A slot represents one token position across layers/heads | Block size counts token positions                         |
| Where is V stored?                              | Initially focused on K layout               | Separate `key_cache` and `value_cache`                   | Every logical token has both K and V                      |
| Padding persisted into KV                       | HF batched KV has physical padded length    | `source_start = physical - logical`                      | Logical and physical KV lengths differ                    |
| `split_legacy_kv_cache()` didn't remove PAD     | It only splits batch dimension              | Strip via `source_start` when writing                    | Splitting requests ≠ removing padding                     |
| Variable-length paged KV couldn't be stacked    | Materialized tensors have different lengths | Temporarily left-pad in BatchBuilder                     | Paged storage and HF execution formats can differ         |
| Decode scheduler grouped by old physical length | Legacy `past_key_values` assumption         | Select using logical state/resources instead             | Scheduler shouldn't depend on temporary tensor shape      |
| FakeRunner expected `list[Request]`             | Runner interface changed to batches         | Make FakeRunner consume `DecodeBatch`                    | Fake components should follow real architecture           |
| FakeRunner read `request.past_key_values`       | Old ownership model                         | Read `batch.past_key_values`                             | Runner should consume batch inputs, not request internals |
| FakeRunner missing `decode_counts`              | New fake state wasn't initialized           | Initialize counter in constructor                        | Keep test doubles internally consistent                   |
| `head_dim` mismatch                             | FakeRunner generated different KV shape     | Configure fixture with `head_dim=2`                      | Fake/model/cache tensor contracts must agree              |
| Decode output contained PAD again               | HF returned rectangular batch KV            | Strip valid suffix during write-back                     | Temporary padding must never become persistent            |
| Block boundary 4→5                              | Needed allocator/storage coordination       | Allocate second block before decode                      | Capacity planning must happen before KV write             |
| Old tests referenced `past_key_values`          | Tests encoded transitional architecture     | Rewrite against FakeRunner/PagedKVCache                  | Tests must follow architectural source of truth           |
| `release_kv_cache()` could clear block table    | Request didn't own allocator metadata       | `KVBlockManager.free()` owns cleanup                     | Resource ownership boundaries matter                      |
| Reuse test expected same block immediately      | FIFO deque put freed block at tail          | Use one-block pool for deterministic reuse test          | Don't confuse allocator policy with correctness           |

The last one is a particularly nice testing lesson. We initially wrote:

```python
assert blocks_b == blocks_a
```

with four blocks available.

But:

```text
initial deque:
[0, 1, 2, 3]

A allocate:
[1, 2, 3]

A frees 0:
[1, 2, 3, 0]

B allocate:
1
```

Nothing was wrong with the allocator. The **test assumption was wrong**.

We changed the test to a one-block pool so reuse was forced:

```text
[0]
↓ A allocate
[]
↓ A free
[0]
↓ B allocate
[]
```

That's a good example of debugging by separating system correctness from policy assumptions.

# What our implementation is NOT yet

This distinction is extremely important for interviews.

We have implemented:

```text
Paged KV STORAGE
```

but not:

```text
PagedAttention
```

Our decode path still does:

```text
PagedKVCache
      ↓
materialize contiguous KV
      ↓
left-pad
      ↓
HF attention
```

A more production-oriented PagedAttention path is conceptually:

```text
Q
+
block_table
+
PagedKVCache
       ↓
PagedAttention kernel
       ↓
attention output
```

The kernel reads non-contiguous physical KV blocks according to `block_table`.

That avoids our expensive:

```text
materialize
+
copy
+
padding
```

step.

So if an interviewer asks:

> Did you implement PagedAttention?

The accurate answer is:

> I implemented the paged KV memory-management layer: a shared physical KV pool, logical-to-physical block tables, block allocation/reuse, padding-aware writes, and materialization for compatibility with a Hugging Face attention backend. I did not yet implement a custom PagedAttention CUDA/Triton kernel, so my current runner materializes paged KV into contiguous tensors before attention.

That's a strong answer because it shows you understand exactly where your implementation sits.

# Day 16 final architecture

The cleanest picture is:

```text
                       Request
              ┌─────────────────────┐
              │ kv_tokens           │
              │ block_table         │
              │ generated_ids       │
              │ state               │
              └──────────┬──────────┘
                         │
                         │ logical→physical mapping
                         ▼
                 KVBlockManager
              allocate / free / reuse
                         │
                         ▼
                   PagedKVCache
              ┌──────────────────────┐
              │ shared physical K/V  │
              │ [L,B,H,S,D]          │
              └──────────┬───────────┘
                         │
                         │ materialize
                         ▼
                   BatchBuilder
              ┌──────────────────────┐
              │ temporary left PAD   │
              │ attention_mask       │
              │ position_ids         │
              └──────────┬───────────┘
                         ▼
                PyTorchModelRunner
                         │
                  updated padded KV
                         │
                         ▼
                    Scheduler
                  remove PAD region
                         │
                         ▼
                   PagedKVCache
```

## The one sentence to remember

Day 15 was:

> **Blocks became real allocatable GPU-memory resources.**

Day 16 was:

> **Those blocks became the actual persistent storage for request KV cache.**

And the next step—Prefix Cache—is where this architecture starts enabling something that would have been awkward with per-request contiguous KV: **two requests can reference the same physical prefix blocks instead of storing duplicate prefix KV.**
