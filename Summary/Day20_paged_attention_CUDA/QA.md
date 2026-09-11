3. If in production environmen, what should we do to improve it?
dim3 softmax_block(num_tokens);
means num_tokens must fit within the GPU's maximum threads-per-block, typically 1024. That's fine for our Day 20 correctness experiment, but it is not a production design.

2. When implementing weighted value, why cannot we use dim3 paged_weighted_value_kernel(num_tokens, kv_head)? If we define in this way, does this mean more CUDA blocks needed?
You can use a grid like:

```cpp
dim3 grid(num_tokens, num_kv_heads);
```

for the weighted-V stage, but it changes **who owns the final output** and creates an extra reduction problem.

Right now we use:

```cpp
dim3 value_grid(num_kv_heads);
dim3 value_block(head_dim);
```

So with:

```text
num_tokens = 20
num_kv_heads = 2
head_dim = 64
```

we launch only:

```text
2 CUDA thread blocks
× 64 threads
```

Each CUDA thread block owns one head:

```text
CUDA block 0 → head 0
CUDA block 1 → head 1
```

and each thread owns one output dimension. For example:

```text
block 0, thread 7
→ output[head=0, dim=7]
```

That thread loops across all 20 tokens:

```cpp
float accumulator = 0.0f;

for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    accumulator += weight[token_idx, 0]
                 * V[token_idx, 0, 7];
}
```

So **one thread completely owns one output element**:

```text
output[0, 0, 7]
```

No other thread writes to it.

---

If instead you use:

```cpp
dim3 grid(num_tokens, num_kv_heads);
dim3 block(head_dim);
```

then yes, you launch more CUDA thread blocks:

```text
20 tokens × 2 heads
= 40 CUDA thread blocks
```

Each block could handle one:

```text
(token, head)
```

just like our QK kernel.

For example:

```text
CUDA block (token=15, head=0)
```

with 64 threads could calculate:

```text
thread 0:
weight[15,0] * V[15,0,0]

thread 1:
weight[15,0] * V[15,0,1]

...

thread 63:
weight[15,0] * V[15,0,63]
```

That sounds attractive because now all tokens can be processed in parallel.

But here's the problem.

For output:

```text
output[head=0, dim=7]
```

we need:

```text
weight[0,0]  * V[0,0,7]
+
weight[1,0]  * V[1,0,7]
+
...
+
weight[19,0] * V[19,0,7]
```

With the new grid, these 20 terms are produced by **20 different CUDA thread blocks**:

```text
block(token=0,  head=0), thread 7
block(token=1,  head=0), thread 7
block(token=2,  head=0), thread 7
...
block(token=19, head=0), thread 7
```

They all contribute to the same:

```text
output[0, 0, 7]
```

And that's where it gets harder.

CUDA thread blocks cannot use:

```cpp
__syncthreads()
```

to synchronize with other blocks.

`__syncthreads()` only works **inside one CUDA thread block**.

So this would be incorrect:

```cpp
output[output_offset] += weight * value;
```

because many CUDA blocks could update the same memory address simultaneously. That's a race condition.

You would need something like:

```cpp
atomicAdd(&output[output_offset], weight * value);
```

or you could write partial results first:

```text
partial[token, head, dim]
```

and then launch another reduction kernel:

```text
partial[token, head, dim]
          ↓
sum across token
          ↓
output[head, dim]
```

So the two designs are:

```text
Current simple design

grid = [heads]
block = [head_dim]

one thread:
    loops over all tokens

           ↓

directly produces one final output element

Advantages:
✓ simple
✓ no atomics
✓ no cross-block reduction
✓ easy to verify

Disadvantage:
✗ token loop is serial inside each thread
```

Versus:

```text
More-parallel design

grid = [tokens, heads]
block = [head_dim]

one thread:
    computes contribution from one token

           ↓

many CUDA blocks contribute to same output

           ↓

need atomicAdd
or
another reduction kernel

Advantages:
✓ tokens processed in parallel

Disadvantages:
✗ more complicated
✗ synchronization/reduction required
✗ potentially expensive atomics
```

So your intuition is correct:

> Using `grid(num_tokens, num_kv_heads)` creates more CUDA blocks and gives us more parallelism over tokens.

The reason we didn't do it is **not because CUDA can't**. It's because Day 20 is correctness-first, and we'd immediately have to solve a second reduction problem.

This is actually very similar to what happened with Q·K, but in reverse.

For QK we need:

```text
sum over head_dim
```

so we placed all `head_dim` threads inside **the same CUDA block**, allowing:

```cpp
__shared__
__syncthreads()
```

to perform the reduction.

For weighted V we need:

```text
sum over tokens
```

If each token belongs to a different CUDA block, the values that need to be summed are now distributed across different blocks.

That is the key issue.

So for Day 20:

```text
one CUDA block per head
+
one thread per dim
+
each thread loops through tokens
```

is intentionally naive but much easier to reason about.

Later, when optimizing PagedAttention, **parallelizing the token dimension and efficiently reducing across tokens is exactly one of the places where production kernels become much more sophisticated.**


1. The key_cache's data type is FP16, why do you change it to kFloat32?
auto scores = torch::empty(
        {
            num_tokens,
            num_kv_heads,
        },
        key_cache.options().dtype(
            torch::kFloat32
        )
    );
    
Answer:
Why our PagedAttention uses both FP16 and FP32

The PagedKVCache and query are stored in FP16:

const __half* query;
const __half* key_cache;
const __half* value_cache;

But we convert values to FP32 for computation:

float q = __half2float(query[...]);
float k = __half2float(key_cache[...]);
float v = __half2float(value_cache[...]);

Our current computation path is therefore:

Q / K / V
FP16 storage
    ↓
convert to FP32
    ↓
Q·K accumulation
FP32
    ↓
softmax
FP32
    ↓
weight × V accumulation
FP32
    ↓
attention output
FP32

The main reason is numerical accuracy. Dot products and weighted sums involve accumulating many values:

$$ Q\cdot K = \sum_d Q_dK_d $$

and

$$ O_d = \sum_t P_tV_{t,d} $$

Accumulating these directly in FP16 would introduce more rounding error, so our simple correctness-first CUDA kernel accumulates in FP32.

Storage dtype ≠ computation dtype

This is the key idea to remember:

Storing Q/K/V in FP16 does not mean all attention arithmetic must be FP16.

For our Day 20 implementation:

FP16 → memory/storage efficiency
FP32 → intermediate computation/accumulation accuracy

Later, if we want the API to return FP16, we can still calculate internally in FP32 and cast only the final result:

__half result =
    __float2half(accumulator);

giving:

FP16 Q/K/V
    ↓
FP32 computation
    ↓
FP16 final output

For Day 20, keeping the output FP32 is useful because our goal is correctness rather than production-level memory/performance optimization.