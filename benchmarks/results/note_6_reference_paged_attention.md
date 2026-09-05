At seq_len=128:

24-layer contiguous attention   ~2.0 ms
materialize-only               ~171 ms
reference PagedAttention       ~295 ms

The reason is that the reference PagedAttention does even more fine-grained work than materialization. Materialization reads each token and copies K/V into contiguous tensors; then PyTorch performs efficient batched matmuls. Your reference PagedAttention instead repeatedly does tiny per-token GPU operations for score computation and value accumulation.

Conceptually:

Materialize path

for each layer/token:
    read K/V and copy
↓
one efficient QK matmul
softmax
one efficient PV matmul

while the reference path is closer to:

for each layer/token:
    read K
    tiny multiply
    tiny reduction

softmax

for each layer/token:
    read V
    tiny multiply
    tiny accumulation

So you removed the intermediate contiguous KV copy, but you replaced highly optimized matrix operations with thousands of tiny Python-driven tensor operations.

That means the result does not imply PagedAttention is slower than materialization in production. It implies:

A token-wise PyTorch reference implementation is useful for correctness and understanding, but it is not a meaningful performance implementation of PagedAttention.