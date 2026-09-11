import math

import pytest
import torch

from app.runtime.cuda.paged_attention import paged_attention


@pytest.mark.parametrize(
    (
        "num_tokens",
        "block_table_values",
    ),
    [
        # Exactly one full block.
        (16, [2]),

        # Partial second block.
        (20, [3, 1]),

        # Exactly two full blocks.
        (32, [0, 2]),

        # Two full blocks with scrambled physical order.
        (32, [3, 1]),
    ],
)
def test_paged_attention_cuda_matches_reference(
    num_tokens: int,
    block_table_values: list[int],
) -> None:
    device = torch.device("cuda")

    num_layers = 2
    num_blocks = 4
    num_kv_heads = 2
    block_size = 16
    head_dim = 64

    layer_idx = 1

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

    block_table = torch.tensor(
        block_table_values,
        device=device,
        dtype=torch.int32,
    )

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
    # Build the logical K/V sequence using PyTorch.
    #
    # This is only the correctness oracle.
    #
    # The CUDA implementation must NOT materialize K/V like this.
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

    # ------------------------------------------------------------
    # Scale:
    #
    # QK / sqrt(head_dim)
    # ------------------------------------------------------------

    scores = (
        scores
        / math.sqrt(head_dim)
    )

    # ------------------------------------------------------------
    # Softmax over tokens for each head.
    #
    # scores shape:
    # [tokens, heads]
    #
    # therefore dim=0.
    # ------------------------------------------------------------

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
    # result:
    # [heads, head_dim]
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


@pytest.mark.parametrize(
    "num_tokens",
    [
        # Partial block and longer than the old
        # 1024-thread limitation.
        1025,

        # Full Day 21 maximum benchmark length.
        4096,
    ],
)
def test_paged_attention_cuda_supports_long_sequences(
    num_tokens: int,
) -> None:
    device = torch.device("cuda")

    num_layers = 1
    num_kv_heads = 2
    block_size = 16
    head_dim = 64
    layer_idx = 0

    required_blocks = (
        num_tokens + block_size - 1
    ) // block_size

    # Leave some physical blocks unused so that
    # logical and physical allocation sizes differ.
    num_blocks = required_blocks + 4

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

    # Reverse the physical block order.
    #
    # logical block 0
    #     -> physical block required_blocks - 1
    #
    # logical block 1
    #     -> physical block required_blocks - 2
    #
    # ...
    block_table = torch.arange(
        required_blocks - 1,
        -1,
        -1,
        device=device,
        dtype=torch.int32,
    )

    output = paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        num_tokens=num_tokens,
        layer_idx=layer_idx,
    )

    # ------------------------------------------------------------
    # Build the logical K/V sequence with vectorized
    # PyTorch indexing.
    #
    # This is only the correctness oracle.
    # The custom CUDA implementation still reads
    # paged storage directly.
    # ------------------------------------------------------------

    token_indices = torch.arange(
        num_tokens,
        device=device,
        dtype=torch.long,
    )

    logical_blocks = (
        token_indices // block_size
    )

    slots = (
        token_indices % block_size
    )

    physical_blocks = block_table[
        logical_blocks
    ].long()

    logical_keys = key_cache[
        layer_idx,
        physical_blocks,
        :,
        slots,
        :,
    ].float()

    logical_values = value_cache[
        layer_idx,
        physical_blocks,
        :,
        slots,
        :,
    ].float()

    # logical_keys:
    # [num_tokens, num_kv_heads, head_dim]

    scores = (
        query[0].float().unsqueeze(0)
        * logical_keys
    ).sum(dim=-1)

    scores = (
        scores
        / math.sqrt(head_dim)
    )

    weights = torch.softmax(
        scores,
        dim=0,
    )

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