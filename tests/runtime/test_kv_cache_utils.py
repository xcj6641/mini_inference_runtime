from __future__ import annotations

from typing import Any

import pytest
import torch

from app.runtime.kv_cache_utils import (
    CUDAMemorySnapshot,
    calculate_actual_kv_bytes,
    estimate_kv_bytes_per_token,
    estimate_total_kv_bytes,
    get_cuda_memory_snapshot,
    get_kv_sequence_length,
    inspect_legacy_kv_cache,
)


def make_fake_kv_cache(
    *,
    num_layers: int = 2,
    batch_size: int = 1,
    num_kv_heads: int = 2,
    sequence_length: int = 4,
    head_dim: int = 8,
    dtype: torch.dtype = torch.float16,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    for _ in range(num_layers):
        key = torch.zeros(
            (
                batch_size,
                num_kv_heads,
                sequence_length,
                head_dim,
            ),
            dtype=dtype,
        )

        value = torch.zeros(
            (
                batch_size,
                num_kv_heads,
                sequence_length,
                head_dim,
            ),
            dtype=dtype,
        )

        layers.append((key, value))

    return tuple(layers)


class FakeModernCache:
    """
    Simulates a newer Transformers cache object.

    It exposes both get_seq_length() and to_legacy_cache().
    """

    def __init__(
        self,
        legacy_cache: Any,
    ) -> None:
        self._legacy_cache = legacy_cache

    def get_seq_length(self) -> int:
        return int(
            self._legacy_cache[0][0].shape[-2]
        )

    def to_legacy_cache(self) -> Any:
        return self._legacy_cache


def test_inspect_legacy_kv_cache_returns_layer_info() -> None:
    cache = make_fake_kv_cache(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        sequence_length=4,
        head_dim=8,
    )

    layer_infos = inspect_legacy_kv_cache(cache)

    assert len(layer_infos) == 2

    layer0 = layer_infos[0]

    assert layer0.layer_index == 0
    assert layer0.key_shape == (1, 2, 4, 8)
    assert layer0.value_shape == (1, 2, 4, 8)
    assert layer0.sequence_length == 4


def test_inspect_legacy_kv_cache_calculates_bytes() -> None:
    cache = make_fake_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        sequence_length=4,
        head_dim=8,
        dtype=torch.float16,
    )

    layer_infos = inspect_legacy_kv_cache(cache)
    layer0 = layer_infos[0]

    expected_tensor_elements = 1 * 2 * 4 * 8
    expected_tensor_bytes = expected_tensor_elements * 2

    assert layer0.key_bytes == expected_tensor_bytes
    assert layer0.value_bytes == expected_tensor_bytes
    assert (
        layer0.total_bytes
        == expected_tensor_bytes * 2
    )


def test_get_kv_sequence_length_from_legacy_cache() -> None:
    cache = make_fake_kv_cache(
        sequence_length=7,
    )

    sequence_length = get_kv_sequence_length(cache)

    assert sequence_length == 7


def test_get_kv_sequence_length_returns_zero_for_none() -> None:
    assert get_kv_sequence_length(None) == 0


def test_get_kv_sequence_length_from_modern_cache() -> None:
    legacy_cache = make_fake_kv_cache(
        sequence_length=9,
    )
    modern_cache = FakeModernCache(legacy_cache)

    sequence_length = get_kv_sequence_length(
        modern_cache
    )

    assert sequence_length == 9


def test_inspect_modern_cache_using_legacy_conversion() -> None:
    legacy_cache = make_fake_kv_cache(
        num_layers=3,
        sequence_length=5,
    )
    modern_cache = FakeModernCache(legacy_cache)

    layer_infos = inspect_legacy_kv_cache(
        modern_cache
    )

    assert len(layer_infos) == 3
    assert layer_infos[0].sequence_length == 5


def test_calculate_actual_kv_bytes_matches_tensor_shapes() -> None:
    cache = make_fake_kv_cache(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        sequence_length=4,
        head_dim=8,
        dtype=torch.float16,
    )

    actual_bytes = calculate_actual_kv_bytes(
        cache
    )

    expected_bytes = (
        2  # key and value
        * 2  # layers
        * 1  # batch size
        * 2  # KV heads
        * 4  # sequence length
        * 8  # head dimension
        * 2  # float16 bytes
    )

    assert actual_bytes == expected_bytes


def test_estimate_kv_bytes_per_token() -> None:
    bytes_per_token = estimate_kv_bytes_per_token(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        bytes_per_element=2,
    )

    expected = (
        2
        * 24
        * 2
        * 64
        * 2
    )

    assert bytes_per_token == expected


def test_estimate_total_kv_bytes() -> None:
    total_bytes = estimate_total_kv_bytes(
        sequence_length=10,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        bytes_per_element=2,
    )

    expected_bytes_per_token = (
        2
        * 24
        * 2
        * 64
        * 2
    )

    assert total_bytes == (
        10 * expected_bytes_per_token
    )


def test_estimated_bytes_match_actual_cache_bytes() -> None:
    num_layers = 3
    num_kv_heads = 2
    sequence_length = 6
    head_dim = 8

    cache = make_fake_kv_cache(
        num_layers=num_layers,
        batch_size=1,
        num_kv_heads=num_kv_heads,
        sequence_length=sequence_length,
        head_dim=head_dim,
        dtype=torch.float16,
    )

    actual_bytes = calculate_actual_kv_bytes(
        cache
    )

    estimated_bytes = estimate_total_kv_bytes(
        sequence_length=sequence_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=2,
    )

    assert estimated_bytes == actual_bytes


@pytest.mark.parametrize(
    (
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "bytes_per_element",
        "expected_message",
    ),
    [
        (
            0,
            2,
            64,
            2,
            "num_layers must be positive",
        ),
        (
            24,
            0,
            64,
            2,
            "num_kv_heads must be positive",
        ),
        (
            24,
            2,
            0,
            2,
            "head_dim must be positive",
        ),
        (
            24,
            2,
            64,
            0,
            "bytes_per_element must be positive",
        ),
    ],
)
def test_estimate_kv_bytes_per_token_rejects_invalid_values(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        estimate_kv_bytes_per_token(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bytes_per_element=bytes_per_element,
        )


def test_estimate_total_kv_bytes_rejects_negative_length() -> None:
    with pytest.raises(
        ValueError,
        match="sequence_length cannot be negative",
    ):
        estimate_total_kv_bytes(
            sequence_length=-1,
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            bytes_per_element=2,
        )


def test_zero_sequence_length_has_zero_kv_bytes() -> None:
    total_bytes = estimate_total_kv_bytes(
        sequence_length=0,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        bytes_per_element=2,
    )

    assert total_bytes == 0


def test_inspect_cache_rejects_none() -> None:
    with pytest.raises(
        ValueError,
        match="past_key_values cannot be None",
    ):
        inspect_legacy_kv_cache(None)


def test_inspect_cache_rejects_empty_cache() -> None:
    with pytest.raises(
        ValueError,
        match="past_key_values contains no layers",
    ):
        inspect_legacy_kv_cache(())


def test_inspect_cache_rejects_layer_without_key_value_pair() -> None:
    invalid_cache = (
        (
            torch.zeros((1, 2, 4, 8)),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Expected key/value pair at layer 0",
    ):
        inspect_legacy_kv_cache(invalid_cache)


def test_inspect_cache_rejects_non_tensor_key() -> None:
    invalid_cache = (
        (
            "not-a-tensor",
            torch.zeros((1, 2, 4, 8)),
        ),
    )

    with pytest.raises(
        TypeError,
        match="Layer 0 key is not a torch.Tensor",
    ):
        inspect_legacy_kv_cache(invalid_cache)


def test_inspect_cache_rejects_non_tensor_value() -> None:
    invalid_cache = (
        (
            torch.zeros((1, 2, 4, 8)),
            "not-a-tensor",
        ),
    )

    with pytest.raises(
        TypeError,
        match="Layer 0 value is not a torch.Tensor",
    ):
        inspect_legacy_kv_cache(invalid_cache)


def test_inspect_cache_rejects_non_four_dimensional_tensor() -> None:
    invalid_cache = (
        (
            torch.zeros((2, 4, 8)),
            torch.zeros((2, 4, 8)),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Expected KV tensors with shape",
    ):
        inspect_legacy_kv_cache(invalid_cache)


def test_inspect_cache_rejects_key_value_shape_mismatch() -> None:
    invalid_cache = (
        (
            torch.zeros((1, 2, 4, 8)),
            torch.zeros((1, 2, 5, 8)),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Key/value shape mismatch at layer 0",
    ):
        inspect_legacy_kv_cache(invalid_cache)


def test_cuda_memory_snapshot_returns_none_for_cpu() -> None:
    snapshot = get_cuda_memory_snapshot(
        torch.device("cpu")
    )

    assert snapshot is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_memory_snapshot_on_cuda() -> None:
    device = torch.device("cuda")

    tensor = torch.zeros(
        1024,
        device=device,
    )

    snapshot = get_cuda_memory_snapshot(device)

    assert snapshot is not None
    assert isinstance(snapshot, CUDAMemorySnapshot)

    assert snapshot.allocated_bytes > 0
    assert snapshot.reserved_bytes >= (
        snapshot.allocated_bytes
    )
    assert snapshot.max_allocated_bytes >= (
        snapshot.allocated_bytes
    )

    del tensor