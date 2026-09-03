# Time node - materialize

Step 3 — Calculate bytes copied

Before sweeping sequence lengths, let's understand how little data we're actually moving.

For each token, each layer has:

K: num_kv_heads × head_dim
V: num_kv_heads × head_dim

Therefore:

bytes =
    seq_len
  × num_layers
  × num_kv_heads
  × head_dim
  × 2              # K + V
  × element_size

For your configuration:

seq_len      = 128
num_layers   = 24
num_kv_heads = 2
head_dim     = 64
FP16         = 2 bytes

So:

128 × 24 × 2 × 64 × 2 × 2
= 1,572,864 bytes
≈ 1.5 MiB

We're copying only about 1.5 MiB, yet taking about 166 ms.

That immediately tells us:

The 166 ms cannot reasonably be explained by raw GPU memory bandwidth.

This strongly supports what we discussed earlier: your token-by-token materialization implementation is dominated by fine-grained operation/dispatch overhead, not by the amount of memory being copied.