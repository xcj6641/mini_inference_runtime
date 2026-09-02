Absolutely. Day 17 ended up being much bigger than the original “Prefix Cache” task. What we built is no longer just “cache a prefix and reuse it”; it is now a fairly complete **batched heterogeneous cached-prefill path**.

# Day 17 — Prefix Cache Integration & Batched Cached Prefill

## 1. What we wanted to solve

The original goal was simple:

```text
Request A:
[10, 11, 12, 13, 14, 15, 16, 17]

                ↓ prefill

PrefixCache:
[10, 11, 12, 13]         → cached blocks
[10, 11, 12, 13, 14, 15, 16, 17] → cached blocks


Request B:
[10, 11, 12, 13, 99, 100]

                ↓

cache hit = [10, 11, 12, 13]

only compute:
[99, 100]
```

Instead of recomputing the entire prompt, a new request can reuse the already-computed KV blocks and prefill only its uncached suffix.

But once we integrated this with continuous batching, the real problem became much harder:

> Can multiple requests with different cached-prefix lengths **and** different suffix lengths share one real model forward pass?

By the end of Day 17, the answer is **yes**.

---

# 2. Final architecture

The cached-prefill path now looks roughly like:

```text
WAITING requests
      │
      ▼
PrefixCache.lookup_longest_prefix()
      │
      ├── MISS ──────► normal full prefill
      │
      └── HIT
           │
           ▼
    attach cached blocks
           │
           ▼
 request.kv_tokens = cached length
           │
           ▼
KVBlockManager.ensure_capacity()
     only for missing suffix
           │
           ▼
_run_cached_prefill_batch()
           │
           ├── materialize cached KV
           ├── LEFT-pad cached KV
           ├── RIGHT-pad suffix tokens
           ├── build attention_mask
           ├── build position_ids
           │
           ▼
 real_runner.prefill_with_past()
           │
           ▼
 write only REAL suffix KV
 into request's physical blocks
           │
           ▼
 request.kv_tokens = prompt length
           │
           ▼
 maybe insert new prefix cache entries
           │
           ▼
 append first generated token
           │
           ▼
       DECODING
```

This is the main achievement of Day 17.

---

# 3. Prefix-cache lookup

A request no longer assumes:

```text
kv_tokens = 0
```

when it enters prefill.

Instead, it may already own cached KV:

```text
prompt length = 10
cached prefix = 8

request.kv_tokens = 8
```

Therefore:

```text
suffix_length
    =
prompt_length - request.kv_tokens

    =
10 - 8

    =
2
```

This turned `kv_tokens` into an important semantic field:

> **How many logical tokens of this request already have valid KV?**

That distinction became especially important during scheduler integration.

---

# 4. Prefix cache works at block boundaries

Our prefix cache doesn't cache arbitrary token counts.

With:

```text
block_size = 4
```

a prompt:

```text
[1,2,3,4,5,6,7,8,9,10]
```

can expose reusable full-block prefixes such as:

```text
[1,2,3,4]          → 1 block
[1,2,3,4,5,6,7,8] → 2 blocks
```

but not:

```text
[1,2,3]
```

or:

```text
[1,2,3,4,5,6]
```

as arbitrary partial cached blocks.

This matches the block-based memory model we established earlier.

---

# 5. Reference counting became critical

A cached block may be referenced by both:

```text
PrefixCache
+
active Request
```

For example:

```text
block 7

PrefixCache ─────┐
                 ├── block 7
Request B ───────┘

ref_count = 2
```

When Request B finishes:

```text
Request B releases
        ↓
ref_count = 1
```

The block must **not** return to the free pool because PrefixCache still owns it.

This led to an important bug/fix around:

```text
free
vs
release
```

The scheduler should release its reference, not blindly free a shared block.

---

# 6. Single cached prefill → batched cached prefill

Initially we supported essentially:

```text
Request A
cached prefix
    +
suffix
    ↓
prefill_with_past()
```

Then we extended it to:

```text
Request A ─┐
Request B ─┼──► one prefill_with_past()
Request C ─┘
```

This is important because otherwise prefix caching would undermine continuous batching.

We don't want:

```text
cache hit A → model.forward()
cache hit B → model.forward()
cache hit C → model.forward()
```

when compatible work can become:

```text
[A, B, C]
    ↓
one model.forward()
```

---

# 7. Variable suffix lengths

The first batching version effectively assumed equal suffix lengths.

We then supported:

```text
A suffix:
[99]

B suffix:
[100,101,102]
```

by **right padding**:

```text
A: [99, PAD, PAD]
B: [100,101,102]
```

This created an important issue.

We cannot simply use:

```python
logits[:, -1, :]
```

because for A, `-1` corresponds to PAD.

So we introduced:

```python
suffix_lengths: list[int]
```

The runner now knows:

```text
suffix_lengths = [1, 3]
```

and selects:

```text
A → logits[0, 0]
B → logits[1, 2]
```

using:

```python
last_suffix_indices = suffix_lengths_tensor - 1
```

This was a good API design decision because `attention_mask` represents model visibility while `suffix_lengths` explicitly represents runtime sequence semantics. The real runner now implements that contract.  

---

# 8. Variable cached-prefix lengths

This was the final major extension.

Suppose:

```text
Request A:
cached prefix length = 4

Request B:
cached prefix length = 8
```

Their cached KV tensors cannot directly batch:

```text
A KV: length 4
B KV: length 8
```

So we left-pad the shorter cached KV:

```text
A:
[PAD PAD PAD PAD 10 11 12 13]

B:
[20 21 22 23 24 25 26 27]
```

Now:

```text
physical cached KV length = 8
```

for both requests.

This gave us a very important conceptual distinction:

```text
logical cached length
≠
physical batched cached length
```

For A:

```text
logical cached length  = 4
physical cached length = 8
```

---

# 9. Why LEFT-pad prefix KV but RIGHT-pad suffix?

This was one of the best conceptual lessons from Day 17.

Suppose:

```text
A prefix length = 4
B prefix length = 8

A suffix length = 1
B suffix length = 3
```

The physical batch becomes:

```text
A:
[PAD PAD PAD PAD 10 11 12 13] | [99 PAD PAD]

B:
[20 21 22 23 24 25 26 27]    | [100 101 102]
```

Why?

### Cached KV: align the END

The cached history needs to terminate at the common suffix boundary:

```text
                         ↓
[LEFT padded cached KV]  | suffix
```

Therefore cached KV is **left padded**.

### Suffix: align the START

Every suffix should begin immediately at that common boundary:

```text
cached KV | [real suffix RIGHT padding]
            ↑
        common start
```

Therefore suffixes are **right padded**.

The resulting physical representation:

```text
[PAD ... real cached KV | real suffix ... PAD]
```

is valid.

Both padding regions are batching artifacts rather than logical request tokens.

---

# 10. Attention mask

This means our mask must describe both kinds of padding.

For:

```text
A:
prefix logical = 4
prefix physical = 8
suffix logical = 1
suffix physical = 3
```

we get:

```text
KV                    suffix
↓                       ↓

[0 0 0 0 1 1 1 1 | 1 0 0]
```

For B:

```text
[1 1 1 1 1 1 1 1 | 1 1 1]
```

The model therefore ignores:

```text
left KV padding
+
right suffix padding
```

while attending to the real sequence.

---

# 11. Position IDs remain logical

Another subtle point:

Physical positions and model positions are not the same thing.

For A:

```text
physical cached layout:

[PAD PAD PAD PAD 10 11 12 13]
 0   1   2   3   4  5  6  7
```

But its real logical prefix still has positions:

```text
10 → 0
11 → 1
12 → 2
13 → 3
```

Therefore its suffix starts at logical position:

```text
99 → position_id 4
```

not:

```text
99 → position_id 8
```

So we explicitly construct suffix position IDs from:

```python
cached_prefix_length
```

rather than `max_cached_prefix_length`.

That is crucial for model correctness.

---

# 12. Physical source vs logical destination

This may be the most important PagedKV insight from Day 17.

After the model runs, A's suffix KV physically begins after the **maximum padded prefix length**:

```text
physical model output:

0 1 2 3 4 5 6 7 | 8 ...
                  ↑
               suffix
```

But in A's own logical KV:

```text
0 1 2 3 | 4
          ↑
        suffix
```

Therefore write-back needs:

```text
source_start      = 8
destination_start = 4
```

or generally:

```python
source_start=max_cached_prefix_length
destination_start=cached_prefix_length
```

This explains why `destination_start` cannot simply equal the physical source position.

It is converting:

```text
batched physical layout
        ↓
per-request logical layout
```

---

# 13. RealRunner verification

We did not rely only on FakeRunner.

That was important.

FakeRunner verified things like:

```text
scheduler selection
batch size
state transitions
logical KV lengths
one cached-prefill call
ordering
```

But FakeRunner cannot prove that transformer semantics are correct.

So we added real-model integration verification.

The critical equivalence became:

```text
Path A

full prompt
    ↓
full prefill
    ↓
next token X


Path B

prefix
    ↓
cached KV
    +
suffix
    ↓
prefill_with_past
    ↓
next token X
```

Then we extended that to heterogeneous batches:

```text
different prefix lengths
+
different suffix lengths
+
real model
+
one cached-prefill batch

→ same next tokens as independent full prefill
```

This validates the real runner's handling of cached KV, masks, position IDs, padding and suffix-logit selection. 

Finally, we tested:

```text
ContinuousScheduler
+
PrefixCache
+
KVBlockManager
+
PagedKVCache
+
PyTorchModelRunner
```

end-to-end.

That passed too.

---

# 14. Important bugs/issues we encountered

Several problems were especially valuable.

### Stale tests after API evolution

`prefill_with_past()` changed from effectively returning/accepting single-request semantics to:

```python
suffix_lengths: list[int]

return list[int], past_key_values
```

Old integration tests still called:

```python
prefill_with_past(...)
```

without `suffix_lengths`, causing:

```text
TypeError:
missing required keyword-only argument
'suffix_lengths'
```

The correct fix was to update the test—not weaken the new API.

### FakeRunner vs RealRunner fixture confusion

At one point, scheduler tests inspected FakeRunner call tracking while accidentally using the real-runner scheduler fixture.

That reinforced the importance of explicitly knowing which layer a test is validating.

### Old rejection tests became invalid

Tests that previously asserted:

```text
variable suffix lengths → error
```

became stale after we intentionally implemented the feature.

Those tests needed to become positive capability tests.

### Padding semantics became much more complicated

We had to distinguish:

```text
full prefill     → left padding
decode           → left padding
cached prefix KV → left padding
cached suffix    → right padding
```

rather than adopting a simplistic rule that all batches should use the same padding direction.

---

# 15. Testing strategy we ended with

Our testing strategy became layered:

```text
             ┌───────────────────┐
             │ Real end-to-end   │
             │ Scheduler + Model │
             └─────────▲─────────┘
                       │
             ┌─────────┴─────────┐
             │ RealRunner tests  │
             │ semantic equality │
             └─────────▲─────────┘
                       │
             ┌─────────┴─────────┐
             │ FakeRunner tests  │
             │ scheduler logic   │
             └─────────▲─────────┘
                       │
             ┌─────────┴─────────┐
             │ Unit tests        │
             │ blocks/cache/etc. │
             └───────────────────┘
```

This is better than relying entirely on either FakeRunner or expensive integration tests.

FakeRunner gives us deterministic, localized failures.

RealRunner tells us whether our assumptions actually work with a transformer.

---

# 16. Final capability after Day 17

Before Day 17, conceptually:

```text
Request
   ↓
prefill whole prompt
   ↓
KV
```

After Day 17:

```text
                  PrefixCache
                       │
                 longest match
                       │
                       ▼
Request ──────► cached KV blocks
                       │
                only missing suffix
                       │
                       ▼
             ContinuousScheduler
                       │
           heterogeneous batching
                /             \
               /               \
      LEFT-pad KV         RIGHT-pad suffix
               \               /
                \             /
                 attention mask
                 position IDs
                       │
                       ▼
              one model forward
                       │
                       ▼
            extract real suffix KV
                       │
                       ▼
                PagedKVCache
```

And heterogeneous here really means:

```text
different cached-prefix lengths ✓
different suffix lengths        ✓
multiple requests               ✓
real model                      ✓
```

That is a substantial improvement over the initial single-request prefix-cache implementation.

---


Day 17 is complete. ✓
