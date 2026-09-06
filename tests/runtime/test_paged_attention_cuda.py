# tests/runtime/test_paged_attention_cuda.py

import torch

from app.runtime.cuda.paged_attention import paged_attention


def test_paged_attention_cuda_extension_smoke() -> None:
    device = torch.device("cuda")

    num_layers = 2
    num_blocks = 4
    num_kv_heads = 2
    block_size = 16
    head_dim = 64

    query = torch.randn(
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

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

    block_table = torch.tensor(
        [0],
        device=device,
        dtype=torch.int32,
    )

    layer_idx = 1
    num_tokens = 16

    output = paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        num_tokens=num_tokens,
        layer_idx=layer_idx,
    )

    assert output.shape == query.shape
    assert output.dtype == torch.float16
    assert output.device.type == "cuda"

    torch.testing.assert_close(
        output,
        torch.zeros_like(query),
    )