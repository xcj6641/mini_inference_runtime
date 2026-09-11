_select_prefill_requests 2 ways
# Run scheduler tests and find the bug in _select_prefill_request
## Tell me about a time you improved code quality/reliability./Tell me about a time you thought beyond the immediate task
Yes. This can be a **different BQ story from the debugging story**. The core message is not “I found a bug,” but:

> **I deliberately kept a regression test suite as the system evolved, and it protected old invariants when new features changed shared infrastructure.**

That is a very good engineering-practice story.

It can answer several BQs, especially: **“Tell me about a time you improved code quality/reliability,” “How do you ensure your changes don't break existing functionality?”, “Tell me about a time testing helped you catch an unexpected issue,” “Tell me about a process improvement you introduced,”** or **“Tell me about a time you thought beyond the immediate task.”** It can also be adapted to “Tell me about a bug you caught before production.”

### Situation

I was building a mini LLM inference runtime incrementally. The scheduler started relatively simple, but over several iterations I added continuous batching, KV block management, paged KV cache, heterogeneous decode batching, and eventually prefix caching.

Because these features all touched the scheduler, I kept the tests from earlier milestones instead of replacing them whenever the architecture changed. I had unit tests for scheduler behavior, FakeRunner-based tests for deterministic request lifecycles, and integration tests that exercised the scheduler with real KV-cache behavior.

### Task

When I integrated Prefix Cache, I had to significantly modify `_select_prefill_requests()`.

The new implementation needed to look for a reusable prefix, retain shared KV blocks, calculate the remaining block requirement, and decide whether the request could be admitted.

My immediate goal was to make Prefix Cache work, but I also wanted to make sure that changing such a central scheduler function didn't silently break behavior that had already been working.

### Action

After the Prefix Cache implementation passed its new feature-specific tests, I ran the entire existing scheduler test suite instead of only the new tests.

Several older tests failed.

Initially, some failures were legitimate consequences of architecture changes. For example, older tests still expected KV state to live directly on the Request, while the new architecture stored persistent KV in the paged cache. I updated those tests to validate the new invariants.

However, I didn't assume that every old failure meant the test was obsolete.

One FakeRunner scheduler test showed that after a request was successfully prefetched, it was still present in the waiting queue.

I traced that behavior back to my new `_select_prefill_requests()` implementation. During the Prefix Cache integration, I had changed the selection logic to iterate over waiting requests and append admitted requests to `selected`, but I had accidentally stopped removing those requests from the waiting queue.

That violated an older scheduler invariant:

```text
WAITING
   ↓ admission
PREFILL
   ↓
DECODING
```

A request should never be both logically active for decode and still eligible for prefill.

Because the older tests exercised multiple scheduler ticks, they caught a behavior that the new Prefix Cache tests did not: on a later tick, the same request could be selected for decode and then selected for prefill again.

I fixed the queue transition so `_select_prefill_requests()` consumes requests from the waiting queue and only puts requests back when they cannot yet be admitted.

I then reran the full unit, FakeRunner, and integration suites.

### Result

The regression suite passed and the Prefix Cache functionality remained intact.

More importantly, it demonstrated why I had kept tests from earlier stages of the project. New feature tests answered, "Does Prefix Cache work?" The older regression tests answered a different question: "Does the rest of the scheduler still work after adding Prefix Cache?"

Since then, I treat tests not only as validation for the code that exists today, but as executable contracts for behavior that future changes should preserve.

The experience reinforced a principle I use when modifying infrastructure code: the more central the component, the more important it is to run tests for behaviors that seem unrelated to the feature being changed, because shared state machines and resource-management code often create non-local regressions.

I especially like the line:

> **“New feature tests answered, ‘Does Prefix Cache work?’ The regression tests answered, ‘Does everything that worked before still work after adding Prefix Cache?’”**

That's memorable and demonstrates mature testing thinking.

There is also a nice technical detail that makes the story credible. Your corrected `_select_prefill_requests()` now explicitly consumes the waiting queue with `popleft()`, preserves requests that weren't admitted in `remaining`, and removes successfully admitted requests from the waiting lifecycle. 

For interviews, I would keep **both versions** because they answer different questions. Use the regression-test version for *reliability, testing, quality, process improvement,* or *preventing regressions*. Use the earlier version for *difficult bug/debugging/root-cause analysis*. Same event, different emphasis—and that's completely normal for BQ preparation.

## Tell me about a difficult bug / regression you found
Yes. I actually think **this is a stronger BQ story** than the `kv_tokens` one, especially for SWE / AI Infra interviews.

The interesting part is not simply “old tests caught a bug.” The stronger engineering story is:

> **While integrating a new Prefix Cache feature into an existing Continuous Scheduler, I reused regression tests from earlier milestones. Those tests exposed that my change had accidentally broken a fundamental scheduler lifecycle invariant: admitted prefill requests were selected but not removed from the waiting queue.**

That is a very realistic regression story.

### Situation

I was building a mini LLM inference runtime with continuous batching, paged KV cache, and prefix caching.

The Continuous Scheduler had already been implemented and tested in earlier milestones. I had several layers of existing tests, including scheduler unit tests, FakeRunner-based tests, and integration tests.

Later, when I integrated Prefix Cache into the scheduler, I modified `_select_prefill_requests()` because admission now needed to consider whether a request could reuse cached prefix blocks before allocating new KV blocks.

The new Prefix Cache tests worked, but when I ran the full regression suite, several older Continuous Scheduler tests started failing.

### Task

I needed to determine whether the failures were simply outdated tests caused by the new architecture, or whether the Prefix Cache integration had introduced an actual scheduling regression.

This distinction mattered because several interfaces had legitimately changed—for example, KV state had moved from `request.past_key_values` into the paged KV cache—so some old assertions really were obsolete.

At the same time, I didn't want to update tests blindly and accidentally hide a real scheduler bug.

### Action

I classified the failures into two categories.

Some failures were caused by intentional architectural changes, such as old tests still checking `request.past_key_values`. I updated those tests to validate the new paged-KV invariants instead.

But another test exposed unexpected behavior: after a request was successfully prefetched, it was still present in the scheduler's waiting queue.

I traced the problem to the new `_select_prefill_requests()` implementation.

The Prefix Cache version iterated over `self.waiting` and added admitted requests to `selected`, but unlike the previous scheduler implementation, it never removed those requests from the waiting queue.

That violated a fundamental lifecycle invariant:

`WAITING → admitted for prefill → DECODING`

A request that had successfully completed prefill should never remain in `WAITING`.

The bug became even more visible in a continuous-batching test. Request A was prefetched on the first scheduler tick. On the second tick, A was selected for decode, but because it was still in the waiting queue, the scheduler could also select it for prefill again. That created inconsistent KV lifecycle behavior and even caused duplicate completion events.

I fixed `_select_prefill_requests()` by consuming requests from the waiting deque and explicitly rebuilding the queue with only requests that were not admitted:

```text
waiting
   ↓
popleft()
   ↓
Can admit?
 ↙       ↘
yes       no
 ↓         ↓
selected   remaining
             ↓
          waiting
```

Successfully admitted requests were therefore removed permanently from `waiting`, while requests that could not be admitted because of memory or batch limits remained queued.

I then reran the older unit, FakeRunner, and integration tests rather than validating only the new Prefix Cache tests.

### Result

The full scheduler regression suite passed again, and the fix restored the intended request lifecycle while preserving Prefix Cache functionality.

More importantly, the experience reinforced a testing principle I now use when modifying infrastructure components: new feature tests prove that the new feature works, but existing regression tests prove that the old invariants still hold.

In this case, tests written several milestones earlier caught a subtle regression introduced by a much later Prefix Cache integration.



There are **three particularly strong interview points** here.

First, you didn't automatically assume that every failing old test represented a bug. Some genuinely were stale because the architecture had changed from request-owned `past_key_values` to paged KV. You distinguished **expected interface evolution from unexpected behavioral regression**.

Second, the actual invariant you discovered is very clean:

```text
Before prefill:
WAITING

Successfully admitted
    ↓
Prefill
    ↓
DECODING

It must NOT become:

WAITING
   +
DECODING
```

Your actual revised `_select_prefill_requests()` now `popleft()`s from `waiting`, puts requests that cannot be admitted into `remaining`, and leaves successfully selected requests out of the rebuilt waiting queue. 

Third, there is a nice **testing pyramid** angle:

```text
Unit tests
    ↓
catch local scheduler invariants

FakeRunner scheduler tests
    ↓
catch request lifecycle / batching behavior

Real integration tests
    ↓
catch interaction with real KV/model execution
```

That gives you a strong answer if an interviewer asks:

> “Why didn't your new Prefix Cache tests catch this?”

You can say:

> “The new tests primarily validated prefix lookup, block reuse, and reference counting. The regression wasn't in prefix caching itself—it was in an older scheduler lifecycle invariant. That's why keeping and running the existing scheduler regression suite was important.”

I would definitely save this as one of the project's main **“Tell me about a difficult bug / regression you found”** stories. It demonstrates regression testing, debugging across components, state-machine reasoning, and LLM inference infrastructure knowledge all in one story.


# Should kv_tokens be reset to 0
## a real engineering tradeoff: resource cleanup vs. preserving useful runtime state for observability/debugging
### Situation

While implementing paged KV-cache lifecycle management in my mini LLM inference runtime, I added cleanup logic for completed requests. When a request finished, the scheduler released its physical KV blocks so those blocks could be reused by other requests.

During testing, one test expected `kv_tokens` to be reset to zero after completion.

### Task

I needed to decide whether `kv_tokens` should be treated as physical resource state that must be cleared, or logical request metadata that could remain after the request finished.

The important requirement was to guarantee that completed requests released all real KV memory and could never affect future scheduling, while still keeping enough state to debug the request afterward.

### Action

I traced how `kv_tokens` and `block_table` were used throughout the scheduler.

I found that `block_table` represented actual ownership of physical KV-cache blocks. Therefore, when a request completed, those blocks absolutely had to be released and the block table cleared.

However, `kv_tokens` was only a logical count describing how many tokens had been represented in KV during the request's lifetime. Completed requests were removed from the active scheduling set, so the scheduler would never use their `kv_tokens` again for admission, decode batching, or memory allocation.

Instead of resetting both fields blindly, I separated their responsibilities:

* `block_table` is physical resource state and is cleared on completion.
* `kv_tokens` is retained as historical/debugging metadata.
* completed requests are removed from `active`, preventing their retained metadata from affecting execution.

I then updated the tests to validate the actual invariant rather than assuming every runtime field must be zero after cleanup.

### Result

The runtime now releases all physical KV resources correctly while preserving useful diagnostic information on completed requests.

This also made the lifecycle model clearer:

`block_table` answers, "What KV memory does this request currently own?"

`kv_tokens` answers, "How much KV state did this request logically accumulate?"

The change avoided unnecessary state destruction, improved post-request observability, and kept the scheduler behavior unchanged because completed requests cannot re-enter decode or admission.

## Tell me about a time you disagreed with a teammate.
### Situation

While implementing request cleanup for a paged KV-cache system in an LLM inference runtime, a teammate and I disagreed about what should happen to `kv_tokens` after a request finishes.

Each request maintained two pieces of KV-related state:

* `block_table`, which identifies the physical KV-cache blocks owned by the request.
* `kv_tokens`, which records how many tokens have been represented in the request's KV cache.

When a request finished, we both agreed that its physical KV blocks had to be released immediately.

The disagreement was about `kv_tokens`.

My teammate suggested resetting it to zero during cleanup because the request no longer had physical KV blocks. I preferred keeping the value so that completed requests retained useful information for debugging and observability.

### Task

Rather than treating this as a preference about code style, I wanted to determine which behavior better represented the semantics of the two fields without introducing correctness or resource-management risks.

The key question became:

**Does `kv_tokens` represent currently allocated memory, or does it represent the logical KV progress of the request?**

### Action

I traced how both fields were used throughout the request lifecycle.

I found that `block_table` was the actual resource-ownership mechanism. The block manager used it to determine which physical KV blocks belonged to a request. Therefore, completion had to release those blocks and clear the block table.

`kv_tokens`, however, was different. It represented logical request progress. During execution, it was useful for determining how much KV had been accumulated, but once a request entered `FINISHED`, the scheduler removed it from the active set.

I verified that decode selection only operated on active requests in the `DECODING` state. Therefore, retaining `kv_tokens` on a completed request could not cause the scheduler to allocate blocks, perform another decode, or consume GPU KV memory.

I proposed defining the lifecycle invariant explicitly:

```text
Running request:
    block_table != []
    kv_tokens > 0

Completed request:
    block_table == []
    kv_tokens = historical final KV length
```

This separated **resource ownership** from **historical metadata**.

I also considered the alternative my teammate proposed. Resetting `kv_tokens` would have been safe from an execution perspective, but it would destroy useful information. If we later inspected a completed request after an unexpected generation result, we would no longer know how far its KV state had progressed.

Based on that analysis, I argued that cleanup should release resources without unnecessarily erasing diagnostic state.

We kept `kv_tokens` and updated the tests so that completion required `block_table` to be empty while allowing `kv_tokens` to preserve its final logical value.

### Result

We preserved correct KV memory reclamation while improving post-request observability.

More importantly, the discussion led us to clarify the semantics of our runtime state:

* `block_table` answers: **What physical KV resources does this request own now?**
* `kv_tokens` answers: **How far did this request's KV state progress?**

That distinction made the lifecycle easier to reason about and made subsequent tests more precise because they validated resource ownership separately from diagnostic metadata.

The experience reinforced an engineering principle for me: when disagreeing about cleanup behavior, I try to identify the invariant and ownership semantics first rather than arguing from intuition. In infrastructure systems, "finished" should mean resources are released, but it doesn't necessarily mean all useful state should be erased.

# BQ Story — Distinguishing Numerical Drift from a Real Regression

**Situation**

While updating integration tests for my LLM inference runtime, I changed an old batched-prefill test to match the new architecture. Previously, each `Request` directly owned its KV cache, but after introducing paged KV management, KV tensors were returned by the model runner and managed separately.

After rewriting the test, the functional checks passed: batched prefill and single-request prefill produced the same next tokens, and the KV-cache shapes and sequence lengths were correct. However, the test still failed because I used `torch.allclose(..., atol=1e-3, rtol=1e-3)` to compare the KV tensors.

**Task**

I needed to determine whether this difference indicated a bug in batched inference or was simply expected floating-point variation. I did not want to loosen the assertion just to make the test pass, because that could hide a real batching error.

**Action**

I inspected the actual tensor values instead of immediately changing the tolerance.

For example, I saw values such as:

```text
single:   9.0547
batched:  9.0625
```

The difference was small and occurred in `float16`.

I then considered the execution paths. Single-request prefill runs with a tensor shaped roughly like:

```text
[1, sequence_length, ...]
```

while batched prefill runs with:

```text
[batch_size, sequence_length, ...]
```

Even though the mathematical operation is equivalent, GPU libraries may select different kernels or perform floating-point accumulation in a different order for different batch shapes. With FP16, these changes can introduce small numerical differences.

I therefore changed the test strategy rather than changing production code.

The test now verifies three levels of correctness:

1. **Functional correctness:** batched and single inference must produce the same next token.
2. **Structural correctness:** KV-cache shapes and sequence lengths must match.
3. **Numerical consistency:** KV tensors must be close within a reasonable FP16 tolerance, rather than bitwise identical.

I relaxed the KV comparison to approximately `1e-2`, but I also added a rule for myself: if the difference became much larger, I would inspect the maximum absolute error instead of continuing to increase the tolerance.

**Result**

The updated test passed while still preserving meaningful correctness guarantees.

More importantly, I avoided two bad fixes:

* modifying correct production code just to satisfy an overly strict test;
* blindly relaxing the test until it passed.

The test now reflects the actual contract of GPU FP16 inference: functional outputs should remain equivalent, structural properties must remain correct, while intermediate tensors may have small numerical differences depending on execution shape.

**What I learned**

A failing test is not automatically evidence of a production bug. Especially in GPU and ML systems, tests must distinguish between mathematical correctness, numerical stability, and exact reproducibility.

I also learned that test assertions should match the abstraction being validated. For a model runner, next-token equivalence and valid KV structure are stronger correctness signals than requiring intermediate FP16 tensors from different GPU execution paths to be identical.


# What was the hardest part of implementing prefix caching?

If an interviewer asks:

**“What was the hardest part of implementing prefix caching?”**

A strong concise answer is:

> The hard part wasn't hashing prompts; it was integrating prefix reuse with continuous batching. Different requests can hit different prefix lengths and have different uncached suffix lengths, so their cached KV tensors and suffix inputs don't naturally have compatible shapes. I left-padded cached KV to align the end of each history, right-padded suffix tokens to align the start of new computation, built masks covering both padding regions, and kept position IDs based on logical rather than physical sequence positions. After the batched forward pass, I copied only real suffix KV from the shared physical suffix boundary back into each request's logical paged-KV positions. I validated the scheduler with fake runners and then compared the real model's cached-prefill outputs against full-prefill reference outputs.

That answer demonstrates that you understand **why** the implementation works, not merely that you implemented a `PrefixCache` class.

# Some draft BQs
Definitely. Day 17 produced several strong Behavioral Question stories because we encountered genuine design disagreements, incorrect assumptions, regressions, and progressively harder requirements—not just “I implemented prefix cache.”

I’d keep about **6 core stories**, with some reusable across multiple BQs.

### 1. Disagreement: keep `kv_tokens` semantics instead of resetting it

**Good for:** “Tell me about a disagreement with a teammate,” “Tell me about a technical decision you defended,” “Tell me about a time you challenged a suggestion.”

**Situation:** After adding prefix caching, a cache-hit request entered prefill with `kv_tokens > 0`. One proposed solution was to reset `kv_tokens` so the old scheduler logic continued to work.

**Your position:** `kv_tokens` should continue to mean:

> Number of logical request tokens for which valid KV already exists.

For a prompt of length 10 with an 8-token cache hit:

```text
prompt_tokens = 10
kv_tokens     = 8
suffix        = 2
```

Resetting it to zero would make existing code easier to reuse, but destroy useful state and blur the distinction between cached and uncached computation.

**Action:** You kept the semantic invariant and instead modified scheduler selection/admission logic to understand partially-prefilled requests.

**Result:** The same state representation now naturally supports cache hits, suffix-only prefill, block allocation, and subsequent decode.

**Strong lesson:** *When integrating a new optimization, preserve meaningful invariants rather than mutating state to fit old control flow.*

---

### 2. Regression tests caught an unrelated scheduler bug

**Good for:** “Tell me about a difficult bug,” “How do you ensure code quality?”, “Tell me about a time testing changed your approach.”

While integrating PrefixCache, old continuous-scheduler tests started failing.

It would have been easy to classify them as stale because the architecture had changed substantially. But investigation showed that some failures exposed a real bug in scheduler selection rather than obsolete expectations.

The important part of this story isn't the specific bug. It's the engineering process:

```text
new PrefixCache feature
        ↓
old scheduler regression fails
        ↓
don't immediately rewrite test
        ↓
identify whether invariant is still valid
        ↓
discover real regression
        ↓
fix scheduler
```

This is a strong answer to:

> “Why do you maintain regression tests?”

because you have a concrete example where they caught something the new feature tests didn't.

---

### 3. FakeRunner passed, but you insisted on RealRunner validation

**Good for:** “Tell me about a time you went beyond the obvious solution,” “How do you validate complex systems?”, “Tell me about a risk you identified.”

This one came directly from your concern today.

The FakeRunner tests showed:

```text
✓ requests batched
✓ KV lengths correct
✓ scheduler states correct
✓ one prefill_with_past call
```

But you realized FakeRunner's KV tensors were synthetic—mostly zeros—and therefore couldn't prove that the real transformer accepted your unusual layout:

```text
[PAD ... cached KV | suffix ... PAD]
```

In particular, FakeRunner couldn't reliably detect mistakes in:

```text
attention_mask
position_ids
padding semantics
logit selection
real past_key_values behavior
```

So you added RealRunner integration tests comparing:

```text
full prefill
     vs
cached prefix + suffix prefill
```

and later:

```text
independent full prefills
     vs
heterogeneous batched cached prefill
```

The outputs matched.

This is one of your strongest Day 17 stories because **you yourself identified the weakness of the test strategy rather than waiting for a production failure.**

A strong closing sentence:

> Unit tests told me the scheduler behaved as designed; real-model equivalence tests told me the design itself was correct.

---

### 4. Generalizing from equal-length batching to heterogeneous batching

**Good for:** “Tell me about a technically challenging project,” “Tell me about a time requirements changed,” “Tell me about a complex problem you simplified.”

Initially cached batching worked for easy cases:

```text
A prefix = 8, suffix = 2
B prefix = 8, suffix = 2
```

Then you generalized it twice.

First:

```text
same prefix length
different suffix lengths
```

which required right-padding suffixes and introducing:

```python
suffix_lengths: list[int]
```

because `logits[:, -1]` was no longer correct.

Then:

```text
different prefix lengths
different suffix lengths
```

which required left-padding cached KV, new attention masks, logical position IDs, and separating physical from logical KV positions.

Final capability:

```text
A: prefix=4, suffix=1
B: prefix=8, suffix=3

→ one model forward
```

This is probably your best **deep technical challenge** story.

---

### 5. Physical KV layout vs logical KV layout

**Good for:** “Tell me about a difficult debugging problem,” “Tell me about a subtle technical problem,” “How do you reason about correctness?”

This could be a more technical variation of #4.

The key discovery was that batching introduces two notions of position.

For a request with a 4-token prefix in a batch whose maximum prefix is 8:

```text
physical:

[PAD PAD PAD PAD A B C D | E]
 0   1   2   3  4 5 6 7   8


logical:

[A B C D | E]
 0 1 2 3   4
```

Therefore suffix KV must be copied:

```text
physical source = 8
logical destination = 4
```

This led to the explicit distinction:

```python
source_start=max_cached_prefix_length
destination_start=cached_prefix_length
```

and also explained why suffix `position_ids` must begin at `4`, not `8`.

The broader engineering lesson:

> Once batching introduced padding, physical tensor layout could no longer be treated as the request's logical sequence layout. Making that distinction explicit prevented several classes of indexing bugs.

That sounds much stronger in an interview than merely saying “I fixed an index.”

---

### 6. Reference counting: `free()` vs `release()`

**Good for:** “Tell me about a bug with serious consequences,” “Tell me about resource management,” “Tell me about a time your initial design was wrong.”

Prefix caching introduced shared ownership:

```text
              ┌── Request
KV block 7 ───┤
              └── PrefixCache
```

The old scheduler cleanup logic effectively treated request completion as:

```text
request finished
→ block is free
```

But now that is wrong.

The correct semantics became:

```text
request finished
→ release request's reference

ref_count 2 → 1

PrefixCache still owns block
→ block remains allocated
```

You discovered this through a failing test expecting the cache to retain one reference after request completion.

This is a particularly useful story because it's recognizable systems engineering: **ownership, lifetime, shared resources, reference counting, and avoiding use-after-free/resource corruption.**

---

There are also two smaller stories worth keeping as backups.

**API evolution / stale tests:** when `prefill_with_past()` changed from single-request semantics to batched semantics with `suffix_lengths`, old tests failed. Rather than changing production code to satisfy obsolete tests, you classified whether each failure represented a real regression or an intentionally changed contract. That's good for “Tell me about maintaining a large codebase.”

**Choosing explicit metadata over clever inference:** you considered deriving suffix lengths from `attention_mask`, but chose explicit `suffix_lengths`. The mask answers “which positions can the model attend to?” while `suffix_lengths` answers “where is this request's last real suffix token?” Keeping those concepts separate made the runner contract clearer. That's useful for “Tell me about a design tradeoff.”

### The stories I would prioritize for interviews

If you only prepare four thoroughly, make them:

| Priority | Story                                 | Best BQ                       |
| -------- | ------------------------------------- | ----------------------------- |
| **1**    | FakeRunner → RealRunner validation    | Quality / taking initiative   |
| **2**    | `kv_tokens` disagreement              | Conflict / technical judgment |
| **3**    | Heterogeneous cached-prefill batching | Hard technical challenge      |
| **4**    | `free` → `release` reference counting | Difficult bug / ownership     |

Stories #4 and #5 can also become your **technical project deep-dive**, rather than purely behavioral answers.

Most importantly, don't make each BQ a separate unrelated story. Day 17 gives you one large project episode with multiple angles. For example, the same heterogeneous-batching work can answer “hardest challenge,” “ambiguity,” “design decision,” or “learned something quickly,” but you change which decision and result you emphasize.

I would write these four as full **STAR interview answers next**, especially the `kv_tokens` disagreement and FakeRunner→RealRunner stories—they have very clean behavioral arcs.

## STAR BQ stories
Great. I’d write these as **interview-ready STAR stories**, but keep them natural enough that you can tell them conversationally rather than memorize a script.

## BQ 1 — Disagreement: Preserve `kv_tokens` semantics

**Hits:** disagreement with teammate, technical decision you defended, challenging an existing approach, making a design tradeoff.

### Disagreement: Preserve `kv_tokens` Semantics

**Situation**

While integrating prefix caching into my LLM inference runtime, I found that the scheduler's original assumptions no longer held.

Before prefix caching, a request waiting for prefill always had `kv_tokens = 0`. With prefix caching, a request might have a 10-token prompt but already have valid KV cache for the first 8 tokens, so `kv_tokens = 8`.

A teammate suggested resetting `kv_tokens` to zero when the request entered the prefill path. That would have made the request compatible with the existing scheduler logic.

**Task**

I needed to decide whether to adapt the new prefix-cache behavior to the old scheduler invariant or change the scheduler to support partially-prefilled requests.

**Action**

I argued that we should preserve the semantic meaning of `kv_tokens`: the number of logical tokens for which the request already has valid KV state.

For example, if a request has a 10-token prompt and an 8-token cache hit, `kv_tokens = 8` contains important information. Resetting it to zero would make the scheduler appear simpler, but other components would no longer know that eight tokens had already been computed.

Instead of changing the state to fit the old code, I modified the scheduler selection and prefill logic to explicitly support requests where:

`0 < kv_tokens < prompt_length`.

I also added regression tests covering cache hits, normal prefill, and subsequent decode to make sure the new invariant worked consistently across the runtime.

**Result**

The resulting design supported prefix caching without introducing a special or misleading interpretation of `kv_tokens`. The same field now works consistently for normal prefill, cached prefill, paged-KV allocation, and decode.

It also made later features easier. When I implemented batched cached prefill, suffix length naturally became:

`suffix_length = prompt_length - kv_tokens`.

The main lesson for me was that when integrating an optimization into an existing system, preserving a meaningful invariant is often better than mutating state just to keep old control flow unchanged.

**Interview note:** Don't frame this as “I insisted and my teammate was wrong.” Say you understood why resetting it was attractive—it minimized scheduler changes—but you preferred preserving the stronger invariant.

---

## BQ 2 — I didn't trust FakeRunner alone

**Hits:** raising quality bar, taking initiative, identifying risk, testing strategy, going beyond requirements.

### Going Beyond FakeRunner Validation

**Situation**

I was implementing batched prefix-cache reuse in an LLM inference runtime. Multiple requests could have different cached-prefix lengths and different uncached suffix lengths, but I wanted to process them in one model forward pass.

I first built scheduler tests using a FakeRunner. Those tests all passed. They verified request states, batch sizes, KV lengths, and that multiple cache-hit requests were combined into one cached-prefill call.

**Task**

I needed to determine whether passing those tests was enough to consider the implementation correct.

**Action**

I realized there was an important weakness in the tests.

The FakeRunner generated synthetic KV tensors, mostly containing zeros. That meant a mistake in KV positioning could still pass because reading from the wrong region of an all-zero tensor looked exactly like reading from the correct region.

More importantly, it couldn't prove that my attention masks, position IDs, padding strategy, and `past_key_values` handling were actually valid for a transformer.

So I added a second validation layer using the real PyTorch model.

I computed a reference result by running a normal full prefill over the complete prompt. Then I independently prefetched the prefix, reused its KV cache, processed only the suffix through `prefill_with_past`, and compared the generated token with the full-prefill result.

After that passed for a single request, I extended the test to a heterogeneous batch where requests had different prefix and suffix lengths.

The batched cached-prefill results had to match independent full-prefill results for every request.

**Result**

The real-model tests passed and gave me confidence that the implementation wasn't merely internally consistent with my FakeRunner assumptions—it was semantically correct for the actual transformer.

I kept both layers of testing because they serve different purposes: FakeRunner tests are fast and isolate scheduler behavior, while RealRunner equivalence tests validate the actual model semantics.

The lesson I took from this was that a test double can tell me whether my code behaves according to my model of the system, but sometimes I also need an independent test to verify that my model itself is correct.

This is probably my favorite Day 17 BQ for you. There's a very clear moment of engineering judgment:

> “The tests passed, but I realized the tests couldn't detect the failure I was worried about.”

That's a strong signal in an interview.

---

## BQ 3 — Hardest technical challenge: heterogeneous cached prefill

**Hits:** most challenging project, complex technical problem, ambiguity, designing a system, learning something difficult.

### Supporting Heterogeneous Batched Cached Prefill

**Situation**

I was adding prefix caching to a continuous-batching LLM inference runtime.

The initial implementation could reuse cached KV for one request, but processing every cache hit individually would reduce the benefit of continuous batching.

The difficult case was batching requests with both different cached-prefix lengths and different suffix lengths.

For example, one request might have four cached tokens and one new token, while another has eight cached tokens and three new tokens.

**Task**

My goal was to process both requests in one model forward pass while preserving exactly the same model semantics as if each full prompt had been processed independently.

**Action**

I broke the problem into physical tensor layout and logical sequence layout.

First, cached KV tensors had different sequence dimensions, so I left-padded the shorter cached KV. That aligned the end of every cached history at a common physical boundary.

Second, the suffixes also had different lengths. I right-padded suffix tokens so every real suffix began immediately after that common cached-KV boundary.

That produced layouts like:

`[PAD ... cached KV | real suffix ... PAD]`.

I then built an attention mask covering both kinds of padding.

A second issue was position IDs. Although a request with four cached tokens might physically occupy eight positions after padding, its suffix still logically starts at position four, not eight. So I generated position IDs using each request's logical cached-prefix length.

A third issue was next-token logits. With right-padded suffixes, `logits[:, -1]` could correspond to padding. I added explicit `suffix_lengths` to the runner API and selected the logits at `suffix_length - 1` for each request.

Finally, after the forward pass, the physical and logical KV positions differed. A short request's new KV might physically start at index eight but logically belong at index four. I therefore separated `source_start`, based on the maximum padded prefix length, from `destination_start`, based on the request's logical cached length.

I validated the design first with scheduler unit tests and then with real-model equivalence tests against full prefill.

**Result**

The final implementation supports multiple cache-hit requests with heterogeneous cached-prefix and suffix lengths in a single model forward pass.

More importantly, I came away with a much clearer distinction between physical batched tensor layout and logical per-request sequence state. That distinction became the key invariant that made the implementation understandable and testable.

For a behavioral interview, shorten the detailed padding explanation unless the interviewer asks. For an **AI Infra technical deep dive**, this is where you can draw the physical/logical layout on the whiteboard.

---

## BQ 4 — Shared ownership bug: `free` vs `release`

**Hits:** difficult bug, failure, ownership, debugging, mistake, preventing serious correctness problems.

### Debugging Shared KV-Block Ownership

**Situation**

My inference runtime originally had a KV block manager where a block was effectively owned by one request. When the request finished, the scheduler could free its blocks.

Prefix caching changed that ownership model.

A KV block could now simultaneously be referenced by an active request and by the prefix cache so that future requests could reuse it.

During integration, one of my tests failed because after a request finished, I expected a cached block to retain one reference, but its ownership state was incorrect.

**Task**

I needed to determine whether the problem was in prefix-cache insertion, request cleanup, or the block manager's reference-counting logic.

**Action**

I traced the lifetime of one block rather than debugging the entire request.

Before reuse, the PrefixCache held one reference.

When another request reused the prefix, its reference count increased to two.

When that request finished, the expected transition was two to one, because the PrefixCache still owned the block.

I found that the scheduler's cleanup path was using semantics equivalent to `free`, which came from the old single-owner design. That assumption was no longer valid once PrefixCache introduced shared ownership.

I changed request cleanup to `release` its ownership instead. The block manager decrements the reference count and only returns the block to the free pool when the count reaches zero.

I added tests specifically checking the reference count after a request using cached blocks finishes.

**Result**

The fix prevented cached blocks from being prematurely reclaimed while still referenced by the PrefixCache.

It also clarified the ownership model across the runtime: components acquire and release references, while physical reclamation happens only when no owners remain.

The broader lesson was that adding caching wasn't just a performance optimization—it changed resource lifetime and ownership semantics. When introducing shared resources, I now explicitly revisit cleanup paths rather than assuming the old lifecycle remains valid.

This story is especially good for **SDE/Systems interviews** because it sounds like a classic shared-resource lifetime problem rather than an ML-specific story.

---

### How I'd map these during an interview

Don't memorize four long answers. Memorize four **story anchors**:

```text
1. kv_tokens
   → disagreement / invariant / technical judgment

2. FakeRunner → RealRunner
   → quality / initiative / testing risk

3. heterogeneous cached prefill
   → hardest technical challenge / complexity

4. free → release
   → debugging / ownership / failure
```

Then adapt the same underlying event to the BQ.

There's actually a fifth story I'd prepare later that may be particularly strong for **“Tell me about a failure or mistake”**: during Day 17 we repeatedly encountered old tests failing after architectural changes, and had to distinguish **a stale test from a genuine regression**. That gives you a story where the interesting skill isn't finding a bug—it's deciding whether the code or the test is wrong.


# Solved a difficult problem

Yes. This is actually a **different and very useful BQ angle** from the “heterogeneous batching was technically difficult” story. The emphasis here should be your **problem-solving methodology**: instead of trying to solve the most general case at once, you deliberately decomposed it so that each stage introduced only one new dimension of complexity.

### Breaking a Complex Prefix-Caching Problem Into Incremental Stages

**Situation**

While building prefix caching for my LLM inference runtime, I reached a point where I wanted to batch multiple cache-hit requests together.

The general case was complicated because requests could differ along two dimensions at the same time: they could have different cached-prefix lengths and different uncached suffix lengths.

For example:

`Request A: prefix=4, suffix=1`

`Request B: prefix=8, suffix=3`

Trying to support the fully general case immediately would require me to reason about cached-KV padding, suffix padding, attention masks, position IDs, output-logit selection, and KV write-back simultaneously. If something failed, it would also be difficult to know which assumption was wrong.

**Task**

I needed to implement heterogeneous cached-prefill batching while keeping each change small enough to reason about and test independently.

**Action**

Instead of implementing the general solution in one step, I deliberately divided the work into three stages.

**Stage A: equal prefix + equal suffix**

I started with the simplest batching case:

`prefix lengths: 4, 4`

`suffix lengths: 2, 2`

There was no padding complexity yet. My goal was simply to prove that multiple requests could share one `prefill_with_past` call and that I could correctly split and write the resulting KV cache back to individual requests.

Once that worked, I treated it as my baseline.

**Stage B: equal prefix + variable suffix**

Next I changed only one dimension:

`prefix lengths: 4, 4`

`suffix lengths: 1, 3`

Because the cached prefixes were still equal, I could focus entirely on suffix batching.

I introduced right padding for suffix tokens and updated the attention mask accordingly.

This exposed another issue: I could no longer use `logits[:, -1]` to select the next token because the last physical position of a shorter suffix could be padding.

I therefore added explicit `suffix_lengths` to the runner interface and selected each request's logits from its last real suffix position.

I tested this stage independently before moving on.

**Stage C: variable prefix + variable suffix**

Only after variable suffixes were stable did I remove the equal-prefix assumption:

`prefix lengths: 4, 8`

`suffix lengths: 1, 3`

Now I could focus specifically on the new prefix problem.

I left-padded shorter cached KV tensors so all cached histories ended at the same physical boundary. This led me to distinguish physical batched positions from logical request positions.

For example, a four-token cached prefix could physically occupy positions four through seven after padding, while logically those tokens were still positions zero through three.

That distinction affected both position IDs and KV write-back. The suffix might physically begin at index eight but logically belong at index four, so I separated the physical `source_start` from the logical `destination_start`.

Because Stage B had already validated suffix padding and logit selection, when something failed in Stage C I could focus on cached-prefix alignment rather than questioning the entire batching implementation.

At every stage, I added tests for the new capability while keeping the previous-stage tests as regression tests. After the staged unit tests passed, I added real-model equivalence tests comparing cached prefill against normal full prefill.

**Result**

I eventually supported requests with both heterogeneous cached-prefix lengths and heterogeneous suffix lengths in a single model forward pass.

More importantly, decomposing the problem made debugging much more controlled. Each stage changed essentially one major assumption:

`Stage A: Can cached requests batch at all?`

`Stage B: Can suffix lengths differ?`

`Stage C: Can cached-prefix lengths differ?`

Instead of debugging several interacting dimensions at once, I always had a previously working stage to use as a reference.

The experience reinforced a problem-solving approach I now use for complex systems work: identify the simplest constrained version that proves the core mechanism, then relax one constraint at a time while preserving the previous stages as regression tests.

I think this story is **stronger for “How do you solve a difficult problem?”** than simply describing how hard heterogeneous batching was.

The key sentence I'd remember for an interview is:

> **“I didn't start by implementing the most general case. I identified the independent dimensions of complexity and relaxed one constraint at a time.”**

And your three-stage structure is very easy to explain on a whiteboard:

```text
Stage A
equal prefix + equal suffix
        │
        │ prove batching itself
        ▼
Stage B
equal prefix + variable suffix
        │
        │ solve suffix padding
        ▼
Stage C
variable prefix + variable suffix
        │
        │ solve cached-KV alignment
        ▼
General heterogeneous cached prefill
```

It also gives you a particularly good answer if the interviewer follows with **“Why didn't you just implement the general solution directly?”**: doing so would change prefix layout, suffix layout, masking, positions, logit selection, and KV write-back simultaneously, making failures much harder to isolate. Your staged approach deliberately kept the debugging search space small.


### Best-fit BQs
BQ	Fit	What to emphasize
Tell me about a time you solved a difficult/complex problem.	⭐⭐⭐⭐⭐	The problem had multiple interacting dimensions, so you decomposed it A → B → C.
How do you approach a large or ambiguous problem?	⭐⭐⭐⭐⭐	Identify independent dimensions of complexity and relax one constraint at a time.
Tell me about a time you broke a large problem into smaller pieces.	⭐⭐⭐⭐⭐	This is almost exactly your story.
Tell me about your problem-solving process.	⭐⭐⭐⭐⭐	Constrained baseline → add one dimension → regression test → generalize.
Tell me about a challenging technical project.	⭐⭐⭐⭐	Explain why heterogeneous cached prefill was hard, then focus on decomposition.
Tell me about a time you had to deal with complexity.	⭐⭐⭐⭐	Avoid changing prefix/suffix/padding/masks/positions simultaneously.
Tell me about a time you worked through ambiguity.	⭐⭐⭐⭐	You didn't initially know the final padding/writeback design; incremental stages let evidence guide the design.
Tell me about a time you demonstrated good judgment.	⭐⭐⭐	You deliberately chose slower-looking incremental development over implementing the general solution immediately.
Tell me about a time you reduced risk.	⭐⭐⭐	Each stage limited the debugging surface and retained previous tests as regression coverage.
Tell me about a time your original solution had to evolve.	⭐⭐⭐	Single/equal batching → variable suffix → fully heterogeneous batching.

The three strongest prompts are therefore:

“Tell me about a time you faced a difficult technical problem. How did you solve it?”

“Tell me about a time you had to break a large problem into smaller, manageable pieces.”

“How do you approach a complex problem when you don't immediately know the solution?”

The core message you want the interviewer to remember

The impressive part isn't merely:

“I implemented variable-length cached prefill.”

It's:

“I recognized that the general problem had two independent dimensions of complexity—prefix length and suffix length—so instead of changing everything simultaneously, I relaxed one constraint at a time.”

And there is a particularly good sentence for the “Why did you approach it this way?” follow-up:

“If I had implemented the general case immediately, a failure could have come from KV padding, suffix padding, attention masks, position IDs, logit selection, or KV write-back. By relaxing one constraint at a time, I always knew which new dimension was responsible for a regression.”