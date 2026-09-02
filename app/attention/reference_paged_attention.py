# app/attention/reference_paged_attention.py

from __future__ import annotations

import math

import torch


def contiguous_attention_reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
    """
    Reference attention for one query token and one attention head.

    Shapes:
        query: [head_dim]
        key:   [seq_len, head_dim]
        value: [seq_len, head_dim]

    Returns:
        [head_dim]
    """
    if query.ndim != 1:
        raise ValueError(
            "query must have shape [head_dim]"
        )

    if key.ndim != 2:
        raise ValueError(
            "key must have shape [seq_len, head_dim]"
        )

    if value.ndim != 2:
        raise ValueError(
            "value must have shape [seq_len, head_dim]"
        )

    if key.shape != value.shape:
        raise ValueError(
            "key and value must have the same shape"
        )

    if query.shape[0] != key.shape[1]:
        raise ValueError(
            "query head_dim must match key head_dim"
        )

    scores = torch.matmul(
        key,
        query,
    )

    scores = scores / math.sqrt(query.shape[0])

    probabilities = torch.softmax(
        scores,
        dim=0,
    )

    output = torch.sum(
        probabilities.unsqueeze(-1) * value,
        dim=0,
    )

    return output

def read_paged_kv_token(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: list[int],
        token_index: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

    if token_index < 0:
        raise ValueError(
            "token_index must be non-negative"
        )

    logical_block = token_index // block_size
    block_offset = token_index % block_size

    if logical_block >= len(block_table):
        raise ValueError(
            "token_index is outside the block table"
        )

    physical_block = block_table[logical_block]

    key = key_cache[
        physical_block,
        block_offset,
    ]

    value = value_cache[
        physical_block,
        block_offset,
    ]

    return key, value

def paged_attention_reference(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: list[int],
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
    """
    Reference paged attention for:
        - batch size 1
        - one decode query token
        - one attention head

    Shapes:
        query:
            [head_dim]

        key_cache/value_cache:
            [num_blocks, block_size, head_dim]

    Returns:
        [head_dim]
    """
    if seq_len <= 0:
        raise ValueError(
            "seq_len must be positive"
        )

    keys = []
    values = []

    for token_index in range(seq_len):
        key, value = read_paged_kv_token(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            token_index=token_index,
            block_size=block_size,
        )

        keys.append(key)
        values.append(value)


    # [seq_len, num_heads, head_dim]
    # For example:
    #     token 0:
    #     [
    #         K0_head0,
    #         K0_head1,
    #     ]

    #     token 1:
    #     [
    #         K1_head0,
    #         K1_head1,
    #     ]

    #     token 2:
    #     [
    #         K2_head0,
    #         K2_head1,
    #     ]

    # Stacking produces:
    #     [
    #         [K0_head0, K0_head1],
    #         [K1_head0, K1_head1],
    #         [K2_head0, K2_head1],
    #     ]

    #     shape:
    #     [seq_len, num_heads, head_dim]
    key = torch.stack(
        keys,
        dim=0,
    )

    value = torch.stack(
        values,
        dim=0,
    )

    num_attention_heads = query.shape[0]
    num_kv_heads = key.shape[1] #[seq_num, head_num, head_dim]

    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible "
            "by num_kv_heads"
        )

    num_queries_per_kv_head = num_attention_heads//num_kv_heads

    outputs = []
    for attention_head_index in range(num_attention_heads):

        kv_head_index = attention_head_index // num_queries_per_kv_head
        
        head_query = query[attention_head_index]

        head_key = key[:, kv_head_index, :]

        head_value = value[:, kv_head_index, :]

        head_output = contiguous_attention_reference(
            query=head_query,
            key=head_key,
            value=head_value,
        )
        outputs.append(head_output)
    return torch.stack(outputs, dim=0,)

def read_paged_kv_token_from_cache(
        paged_kv_cache,
        block_table: list[int],
        layer_index: int,
        token_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    if token_index < 0:
        raise ValueError(
            "token_index must be non-negative"
        )

    block_size = paged_kv_cache.block_size

    logical_block = (
        token_index // block_size
    )

    block_offset = (
        token_index % block_size
    )

    if logical_block >= len(block_table):
        raise ValueError(
            "token_index is outside the block table"
        )

    physical_block = block_table[
        logical_block
    ]

    key = paged_kv_cache.key_cache[
        layer_index,
        physical_block,
        :,
        block_offset,
        :,
    ]

    value = paged_kv_cache.value_cache[
        layer_index,
        physical_block,
        :,
        block_offset,
        :,
    ]

    return key, value

def paged_attention_reference_from_cache(
        query: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        block_table: list[int],
        layer_index: int,
        seq_len: int,
    ) -> torch.Tensor:
    """
    Reference decode attention using the real PagedKVCache.

    Shapes:
        query:
            [num_attention_heads, head_dim]

        PagedKVCache:
            [num_layers,
             num_blocks,
             num_kv_heads,
             block_size,
             head_dim]

    Returns:
        [num_attention_heads, head_dim]
    """
    if seq_len <= 0:
        raise ValueError(
            "seq_len must be positive"
        )

    keys = []
    values = []

    for token_index in range(seq_len):
        key, value = read_paged_kv_token_from_cache(
            paged_kv_cache=paged_kv_cache,
            block_table=block_table,
            layer_index=layer_index,
            token_index=token_index,
        )

        keys.append(key)
        values.append(value)

    # [seq_len, num_kv_heads, head_dim]
    key = torch.stack(
        keys,
        dim=0,
    )

    value = torch.stack(
        values,
        dim=0,
    )

    num_attention_heads = query.shape[0]
    num_kv_heads = key.shape[1]

    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible "
            "by num_kv_heads"
        )

    num_queries_per_kv_head = (
        num_attention_heads // num_kv_heads
    )

    outputs = []

    for attention_head_index in range(
        num_attention_heads
    ):
        kv_head_index = (
            attention_head_index
            // num_queries_per_kv_head
        )

        head_output = contiguous_attention_reference(
            query=query[attention_head_index],
            key=key[:, kv_head_index, :],
            value=value[:, kv_head_index, :],
        )

        outputs.append(head_output)

    return torch.stack(
        outputs,
        dim=0,
    )

