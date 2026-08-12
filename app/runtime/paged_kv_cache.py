# app/runtime/paged_kv_cache.py

import torch


class PagedKVCache:
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        cache_shape = (
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self.key_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=self.device,
        )

        self.value_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=self.device,
        )

    def get_physical_location(
        self,
        block_table: list[int],
        token_index: int,
    ) -> tuple[int, int]:
        if token_index < 0:
            raise ValueError("token_index must be non-negative")

        logical_block_index = token_index // self.block_size
        slot_index = token_index % self.block_size

        if logical_block_index >= len(block_table):
            raise IndexError("block_table does not cover token_index")

        physical_block_id = block_table[logical_block_index]

        if not 0 <= physical_block_id < self.num_blocks:
            raise ValueError("invalid physical block id")

        return physical_block_id, slot_index

    def write_request_kv(
        self,
        *,
        block_table: list[int],
        past_key_values,
        num_tokens: int,
        source_start: int = 0,
    ) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        if len(past_key_values) != self.num_layers:
            raise ValueError("unexpected number of KV layers")

        if source_start < 0:
            raise ValueError("source_start must be non-negative")
    
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            if key.shape[0] != 1 or value.shape[0] != 1:
                raise ValueError("only batch size 1 is supported")

            if key.shape[1] != self.num_kv_heads:
                raise ValueError("unexpected number of KV heads")

            if value.shape[1] != self.num_kv_heads:
                raise ValueError("unexpected number of KV heads")

            if key.shape[2] < num_tokens or value.shape[2] < num_tokens:
                raise ValueError("KV sequence is shorter than num_tokens")

            if key.shape[3] != self.head_dim:
                raise ValueError("unexpected head_dim")

            if value.shape[3] != self.head_dim:
                raise ValueError("unexpected head_dim")
            
            if source_start + num_tokens > key.shape[2]:
                raise ValueError("source_start and num_tokens exceed key shape")
                        
            if source_start + num_tokens > value.shape[2]:
                raise ValueError("source_start and num_tokens exceed value shape")
                    
            for token_idx in range(num_tokens):
                physical_block_id, slot_idx = (
                    self.get_physical_location(
                        block_table,
                        token_idx,
                    )
                )

                source_token_idx = source_start + token_idx

                self.key_cache[
                    layer_idx,
                    physical_block_id,
                    :,
                    slot_idx,
                    :,
                ] = key[
                    0,
                    :,
                    source_token_idx,
                    :,
                ]

                self.value_cache[
                    layer_idx,
                    physical_block_id,
                    :,
                    slot_idx,
                    :,
                ] = value[
                    0,
                    :,
                    source_token_idx,
                    :,
                ]

    def materialize_request_kv(
        self,
        *,
        block_table: list[int],
        num_tokens: int,
    ):
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        layers = []

        for layer_idx in range(self.num_layers):
            key = torch.empty(
                (
                    1,
                    self.num_kv_heads,
                    num_tokens,
                    self.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )

            value = torch.empty_like(key)

            for token_idx in range(num_tokens):
                physical_block_id, slot_idx = (
                    self.get_physical_location(
                        block_table,
                        token_idx,
                    )
                )

                key[
                    0,
                    :,
                    token_idx,
                    :,
                ] = self.key_cache[
                    layer_idx,
                    physical_block_id,
                    :,
                    slot_idx,
                    :,
                ]

                value[
                    0,
                    :,
                    token_idx,
                    :,
                ] = self.value_cache[
                    layer_idx,
                    physical_block_id,
                    :,
                    slot_idx,
                    :,
                ]

            layers.append((key, value))

        return tuple(layers)
