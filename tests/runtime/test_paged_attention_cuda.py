# tests/runtime/test_paged_attention_cuda.py

import torch

from app.runtime.cuda.paged_attention import paged_attention

def test_paged_attention_cuda_reads_paged_key_cache() -> None:
    device = torch.device("cuda")

    num_layers = 2
    num_blocks = 4
    num_kv_heads = 2
    block_size = 16
    head_dim = 64

    key_cache = torch.randn(
        num_layers,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    value_cache = torch.randn_like(key_cache)

    query = torch.randn(
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    # Logical block 0 -> physical block 3
    # Logical block 1 -> physical block 1
    block_table = torch.tensor(
        [3, 1],
        device=device,
        dtype=torch.int32,
    )

    layer_idx = 1
    num_tokens = 20

    output = paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        num_tokens=num_tokens,
        layer_idx=layer_idx,
    )

    expected = torch.cat(
        [
            key_cache[layer_idx, 3, :, :, :].permute(1, 0, 2),
            key_cache[layer_idx, 1, :, :4, :].permute(1, 0, 2),
        ],
        dim=0,
    )

    assert output.shape == (20, 2, 64)

    torch.testing.assert_close(
        output,
        expected,
    )