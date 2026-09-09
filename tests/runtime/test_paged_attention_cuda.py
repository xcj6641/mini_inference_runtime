import math

import torch

from app.runtime.cuda.paged_attention import (
    paged_attention,
)


def test_paged_attention_cuda_computes_attention_weights() -> None:
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
        num_tokens,
        num_kv_heads,
    )

    assert output.dtype == torch.float32

    # ------------------------------------------------------------
    # Build the reference Q·K scores in PyTorch.
    # ------------------------------------------------------------

    expected_scores = torch.empty(
        num_tokens,
        num_kv_heads,
        device=device,
        dtype=torch.float32,
    )

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

        expected_scores[token_idx] = (
            query[0].float()
            * key.float()
        ).sum(dim=-1)

    # ------------------------------------------------------------
    # Scaled dot-product attention:
    #
    # QK / sqrt(head_dim)
    # ------------------------------------------------------------

    expected_scores = (
        expected_scores
        / math.sqrt(head_dim)
    )

    # ------------------------------------------------------------
    # scores shape:
    #
    # [tokens, heads]
    #
    # We want softmax over TOKENS independently for each head.
    #
    # Therefore dim=0.
    # ------------------------------------------------------------

    expected_weights = torch.softmax(
        expected_scores,
        dim=0,
    )

    torch.testing.assert_close(
        output,
        expected_weights,
        rtol=1e-4,
        atol=1e-5,
    )

    # ------------------------------------------------------------
    # Every head's attention weights should sum to 1.
    # ------------------------------------------------------------

    weight_sums = output.sum(
        dim=0
    )

    torch.testing.assert_close(
        weight_sums,
        torch.ones_like(
            weight_sums
        ),
        rtol=1e-5,
        atol=1e-5,
    )