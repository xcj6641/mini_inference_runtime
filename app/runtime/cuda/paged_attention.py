# app/runtime/cuda/paged_attention.py
from __future__ import annotations

import torch

from app.runtime.cuda.load_paged_attention import paged_attention_cuda


def paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    layer_idx: int,
) -> torch.Tensor:
    assert query.is_cuda
    assert key_cache.is_cuda
    assert value_cache.is_cuda
    assert block_table.is_cuda

    assert query.dtype == torch.float16
    assert key_cache.dtype == torch.float16
    assert value_cache.dtype == torch.float16
    assert block_table.dtype == torch.int32

    # Day 20 frozen scope
    assert query.ndim == 3
    assert query.shape == (1, 2, 64)

    assert key_cache.ndim == 5
    assert value_cache.ndim == 5

    (
        num_layers,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
    ) = key_cache.shape

    assert value_cache.shape == key_cache.shape

    assert num_kv_heads == 2
    assert block_size == 16
    assert head_dim == 64

    assert 0 <= layer_idx < num_layers
    assert num_tokens > 0

    required_blocks = (num_tokens + block_size - 1) // block_size

    assert block_table.ndim == 1
    assert block_table.numel() >= required_blocks

    return paged_attention_cuda.forward(
        query,
        key_cache,
        value_cache,
        block_table,
        num_tokens,
        layer_idx,
    )