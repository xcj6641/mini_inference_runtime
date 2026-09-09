import math

import torch

from app.runtime.cuda.paged_attention import (
    paged_attention,
)


def test_paged_attention_cuda_matches_reference() -> None:
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

    value_cache = torch.randn_like(
        key_cache
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

    assert output.shape == (
        1,
        num_kv_heads,
        head_dim,
    )

    assert output.dtype == torch.float32

    # ------------------------------------------------------------
    # Build logical contiguous K/V using PyTorch.
    #
    # This is only the test oracle.
    # The CUDA implementation does NOT materialize them.
    # ------------------------------------------------------------

    logical_keys = []
    logical_values = []

    for token_idx in range(num_tokens):
        logical_block = (
            token_idx // block_size
        )

        slot = (
            token_idx % block_size
        )

        physical_block = int(
            block_table[
                logical_block
            ].item()
        )

        key = key_cache[
            layer_idx,
            physical_block,
            :,
            slot,
            :,
        ]

        value = value_cache[
            layer_idx,
            physical_block,
            :,
            slot,
            :,
        ]

        logical_keys.append(key)
        logical_values.append(value)

    # [tokens, heads, head_dim]
    logical_keys = torch.stack(
        logical_keys,
        dim=0,
    ).float()

    logical_values = torch.stack(
        logical_values,
        dim=0,
    ).float()

    # ------------------------------------------------------------
    # Q · K
    #
    # query[0]:
    # [heads, head_dim]
    #
    # logical_keys:
    # [tokens, heads, head_dim]
    #
    # result:
    # [tokens, heads]
    # ------------------------------------------------------------

    scores = (
        query[0].float().unsqueeze(0)
        * logical_keys
    ).sum(dim=-1)

    scores = (
        scores
        / math.sqrt(head_dim)
    )

    # Softmax across TOKENS for each head.
    weights = torch.softmax(
        scores,
        dim=0,
    )

    # ------------------------------------------------------------
    # Weighted V sum:
    #
    # weights:
    # [tokens, heads]
    #
    # logical_values:
    # [tokens, heads, head_dim]
    #
    # -> [heads, head_dim]
    # ------------------------------------------------------------

    expected = (
        weights.unsqueeze(-1)
        * logical_values
    ).sum(dim=0)

    expected = expected.unsqueeze(0)

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )