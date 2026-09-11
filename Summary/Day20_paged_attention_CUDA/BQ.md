# Implement CUDA
## Tell me about a time you had to learn something new quickly

One example was when I was building a small LLM inference runtime and decided to implement a minimal CUDA PagedAttention kernel.

At that point, I understood PagedAttention conceptually and had already implemented paged KV-cache management in Python and PyTorch, but my practical CUDA knowledge was very limited. I wasn't comfortable yet with concepts like grids, thread blocks, thread indexing, shared memory, synchronization, or parallel reduction.

I also had a time constraint because my semester was about to start, so I couldn't spend several weeks learning CUDA systematically before continuing the project.

My approach was to combine just-in-time learning with hands-on implementation.

I first defined a very narrow goal: given one decode query and a paged KV cache, I wanted to follow the block table directly from CUDA and compute correct attention without materializing the KV cache.

I started with a very small CUDA extension and gradually added functionality. Whenever I got stuck on something for much longer than I expected, I treated that as a signal that there was probably a concept I didn't understand yet, rather than continuing to randomly change the code.

For example, when I first saw the CUDA launch configuration using `grid`, `blockIdx`, `threadIdx`, and `dim3`, I couldn't clearly explain how those concepts mapped to my paged KV cache. So I stopped implementing and used ChatGPT interactively to ask very specific questions about how the GPU executed the kernel. If an explanation still contained something I couldn't understand, I kept drilling down and asked it to walk through concrete examples.

I then reconstructed those examples myself. I traced a logical token to its logical block and slot, through the block table to a physical KV block, and finally to the CUDA threads reading each head dimension. That's when I understood an important distinction: a CUDA thread block is an execution concept, while a KV-cache block is a memory-management concept.

Once I understood that concept, I tried to apply it myself rather than just continuing from generated code.

That became my learning loop. For QK computation, the next gap was how 64 CUDA threads could combine their partial products, so I learned shared memory, `__syncthreads()`, and parallel reduction, and then applied them to the kernel. Later, softmax made me realize that the parallelization dimension had changed from head dimension to tokens, so I had to understand why a different CUDA execution layout was needed.

As I became more comfortable, I deliberately tried writing the next weighted-V kernel myself based on what I had learned. Then I used ChatGPT more like a reviewer to check my reasoning and implementation. It found issues such as an argument-order mismatch and tensor allocation options. I understood why those were bugs, fixed them, and validated the implementation against a PyTorch reference.

So my fast-learning loop was essentially: when I'm stuck, identify the underlying knowledge gap; learn that concept through targeted resources and questions; work through a concrete example until I understand it; then try to apply it independently and use tests or review to verify my understanding.

After I got the end-to-end PagedAttention implementation working, I also realized that just-in-time learning can leave gaps. So I started going back and learning CUDA more systematically—reading the basic concepts from technical articles and NVIDIA's documentation, and using AI when I needed a deeper explanation of a particular detail.

By the end of that process, I had gone from very limited practical CUDA experience to implementing and explaining an end-to-end PagedAttention path with direct paged K/V access, shared-memory reduction, synchronization, softmax, and different CUDA execution mappings.

The main thing I learned about learning quickly is that I use two modes. When I need to become productive quickly, I learn on demand around a concrete problem and immediately apply each new concept. Once I'm productive, I go back and study the technology systematically so that the knowledge isn't limited to just that one implementation.


## Tell me about a difficult technical problem, “How do you approach an ambiguous problem?” or “Tell me about a time you broke down a complex problem.

One example was when I was building a small LLM inference runtime and wanted to understand PagedAttention at a deeper level.

I had already implemented paged KV-cache management in Python and PyTorch, but I realized that I still didn't really understand how a production-style attention kernel could operate directly on paged KV memory. CUDA was relatively new to me, and I had limited time before my semester started, so I gave myself a very narrow goal: instead of trying to learn CUDA broadly, I would learn just enough to implement the smallest correct PagedAttention kernel.

I broke the problem into several checkpoints.

First, I built a minimal PyTorch C++/CUDA extension just to prove that I could pass tensors from Python through C++ and launch CUDA code.

Then I implemented only paged memory access. I learned how CUDA grids, thread blocks, and threads map to the problem, and how a logical token maps through a block table to a physical KV-cache block and slot. One important realization for me was that a CUDA thread block and a KV-cache block are completely different concepts—one describes execution and the other describes memory organization.

Once direct paged access worked, I added the actual attention math incrementally. I implemented QK dot products using one CUDA block per token and head, with threads cooperating on the head dimension through shared-memory reduction. Then I added scaling and softmax. That forced me to rethink the parallelization because softmax requires communication across tokens rather than across the head dimension.

Finally, I implemented the weighted V computation. I initially considered parallelizing across both tokens and heads, but I realized that multiple CUDA blocks would then contribute to the same output element, which would require atomics or another reduction. Since my goal was correctness first, I chose a simpler mapping where one thread owns one output dimension and accumulates across tokens.

To make sure I wasn't just getting plausible results, I built a PyTorch reference and compared the CUDA output against it. I tested full blocks, partial blocks, multiple blocks, and scrambled physical block mappings. All of those cases passed.

The biggest thing I learned from that experience was how to learn a technically difficult topic under a time constraint. Instead of trying to study CUDA comprehensively first, I defined a concrete end goal, decomposed it into independently testable milestones, and learned each CUDA concept exactly when the implementation required it.

By the end, I had gone from having very limited CUDA experience to implementing and explaining an end-to-end, correctness-tested PagedAttention path that directly accessed paged KV memory without materializing it into contiguous tensors.

# Choose grid(num_kv_heads) rather than grid(num_tokens, num_kv_heads) when implementing paged_weighted_value_kernel(weight * Value_cache).
## Tell me about a time you chose simplicity over a more sophisticated solution

Tell me about a time you disagreed with your initial approach”
Tell me about a time you chose simplicity over a sophisticated solution.
Tell me about a technical tradeoff you made.
Tell me about a time you avoided overengineering.
How do you balance performance and maintainability/correctness?
Tell me about a time you had to make a decision with multiple technical approaches.
Tell me about a time you prioritized.

One example was when I was implementing a CUDA PagedAttention kernel for a small LLM inference runtime.

My goal at that stage wasn't to build a production-optimized kernel. I wanted to prove that attention could directly access K and V from paged KV storage through a block table, without first materializing them into contiguous tensors.

When I got to the weighted-V part of attention, I needed to compute, for every head and output dimension, the weighted sum of V across all tokens.

My first thought was to parallelize across both tokens and heads. For example, I could launch a CUDA grid with `num_tokens × num_heads` blocks, similar to what I had done for the QK kernel. Each block would process one token and one head. That looked attractive because it exposed much more GPU parallelism.

But when I worked through the ownership of the output, I realized there was a complication. Multiple CUDA blocks would produce partial results for the same output element. Since CUDA blocks can't synchronize with each other using `__syncthreads()`, I would need either atomic operations or another kernel to reduce those partial results.

At that point I asked myself what I was actually trying to accomplish in this stage. The requirement was correctness and direct paged-KV access, not maximum performance.

So I chose a much simpler execution layout. I launched one CUDA block per head and one thread per output dimension. Each thread owned exactly one final output element and simply looped through all tokens, followed the block table to find each physical V location, multiplied it by the attention weight, and accumulated the result locally.

That design had less token-level parallelism, so I knew it wouldn't be the fastest implementation. But it had several advantages: there were no atomics, no cross-block reduction, no race conditions, and the ownership of every output element was very easy to reason about.

I then compared the CUDA output against a PyTorch reference and tested one full block, partial blocks, multiple blocks, and scrambled physical block mappings. All of the cases passed.

I deliberately stopped there instead of immediately optimizing it. I documented the serial token loop as a known performance limitation and kept token-level parallelization and reduction as a future optimization.

The lesson for me was that a more parallel or sophisticated design isn't automatically the better engineering choice. I try to optimize for the current requirement. In this case, a simple implementation gave me a trustworthy correctness baseline and isolated the core PagedAttention behavior. Once that baseline is established, I can benchmark it, identify whether that part is actually a bottleneck, and optimize from a known-correct implementation rather than mixing correctness and performance problems together.


