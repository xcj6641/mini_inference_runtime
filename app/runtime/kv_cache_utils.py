from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Q: [batch, num_attention_heads, seq_len, head_dim]
# K: [batch, num_kv_heads,        seq_len, head_dim]
# V: [batch, num_kv_heads,        seq_len, head_dim]
@dataclass(frozen=True)
class KVLayerInfo:
    layer_index: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    sequence_length: int
    key_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.key_bytes + self.value_bytes


@dataclass(frozen=True)
class CUDAMemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

def _to_legacy_cache(
    past_key_values: Any,
) -> Any:
    to_legacy_cache = getattr(
        past_key_values,
        "to_legacy_cache",
        None,
    )

    if callable(to_legacy_cache):
        return to_legacy_cache()

    return past_key_values

def inspect_legacy_kv_cache(
    past_key_values: Any,
) -> list[KVLayerInfo]:
    if past_key_values is None:
        raise ValueError(
            "past_key_values cannot be None"
        )

    legacy_cache = _to_legacy_cache(
        past_key_values
    )

    layer_infos: list[KVLayerInfo] = []

    for layer_index, layer_cache in enumerate(
        legacy_cache
    ):
        if len(layer_cache) != 2:
            raise ValueError(
                f"Expected key/value pair at layer {layer_index}"
            )

        key, value = layer_cache

        if not isinstance(key, torch.Tensor):
            raise TypeError(
                f"Layer {layer_index} key is not a torch.Tensor"
            )

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Layer {layer_index} value is not a torch.Tensor"
            )

        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Expected KV tensors with shape "
                "[batch, num_kv_heads, seq_len, head_dim]"
            )

        if key.shape != value.shape:
            raise ValueError(
                f"Key/value shape mismatch at layer {layer_index}: "
                f"key={tuple(key.shape)}, value={tuple(value.shape)}"
            )

        layer_infos.append(
            KVLayerInfo(
                layer_index=layer_index,
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
                sequence_length=int(key.shape[-2]),
                key_bytes=_tensor_bytes(key),
                value_bytes=_tensor_bytes(value),
            )
        )

    if not layer_infos:
        raise ValueError("past_key_values contains no layers")

    return layer_infos


def get_kv_sequence_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0

    get_seq_length = getattr(
        past_key_values,
        "get_seq_length",
        None,
    )

    if callable(get_seq_length):
        return int(get_seq_length())

    layer_infos = inspect_legacy_kv_cache(past_key_values)
    return layer_infos[0].sequence_length


def calculate_actual_kv_bytes(
    past_key_values: Any,
) -> int:
    layer_infos = inspect_legacy_kv_cache(past_key_values)

    return sum(
        layer_info.total_bytes
        for layer_info in layer_infos
    )


def estimate_kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    values = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "bytes_per_element": bytes_per_element,
    }

    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def estimate_total_kv_bytes(
    *,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length cannot be negative")

    return sequence_length * estimate_kv_bytes_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )


def get_cuda_memory_snapshot(
    device: torch.device,
) -> CUDAMemorySnapshot | None:
    if device.type != "cuda":
        return None

    torch.cuda.synchronize(device)

    return CUDAMemorySnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        max_allocated_bytes=torch.cuda.max_memory_allocated(
            device
        ),
    )