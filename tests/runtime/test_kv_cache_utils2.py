import torch

from app.runtime.kv_cache_utils import (
    split_legacy_kv_cache,
    stack_legacy_kv_caches,
)


def make_past_key_values(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    sequence_length: int,
    head_dim: int,
) -> tuple:
    layers = []

    for _ in range(num_layers):
        key = torch.randn(
            batch_size,
            num_kv_heads,
            sequence_length,
            head_dim,
        )

        value = torch.randn(
            batch_size,
            num_kv_heads,
            sequence_length,
            head_dim,
        )

        layers.append((key, value))

    return tuple(layers)


def assert_past_key_values_equal(
    actual,
    expected,
) -> None:
    assert len(actual) == len(expected)

    for actual_layer, expected_layer in zip(
        actual,
        expected,
    ):
        actual_key, actual_value = actual_layer
        expected_key, expected_value = expected_layer

        assert torch.equal(
            actual_key,
            expected_key,
        )
        assert torch.equal(
            actual_value,
            expected_value,
        )


def test_split_and_stack_single_layer_round_trip() -> None:
    past_key_values = make_past_key_values(
        num_layers=1,
        batch_size=2,
        num_kv_heads=4,
        sequence_length=8,
        head_dim=16,
    )

    per_request_caches = (
        split_legacy_kv_cache(
            past_key_values
        )
    )

    assert len(per_request_caches) == 2

    restored_past_key_values = (
        stack_legacy_kv_caches(
            per_request_caches
        )
    )

    assert_past_key_values_equal(
        restored_past_key_values,
        past_key_values,
    )


def test_split_and_stack_multi_layer_round_trip() -> None:
    past_key_values = make_past_key_values(
        num_layers=3,
        batch_size=3,
        num_kv_heads=8,
        sequence_length=32,
        head_dim=16,
    )

    per_request_caches = (
        split_legacy_kv_cache(
            past_key_values
        )
    )

    assert len(per_request_caches) == 3

    for request_cache in per_request_caches:
        assert len(request_cache) == 3

    restored_past_key_values = (
        stack_legacy_kv_caches(
            per_request_caches
        )
    )

    assert_past_key_values_equal(
        restored_past_key_values,
        past_key_values,
    )


def test_split_preserves_batch_dimension() -> None:
    past_key_values = make_past_key_values(
        num_layers=2,
        batch_size=3,
        num_kv_heads=4,
        sequence_length=12,
        head_dim=8,
    )

    per_request_caches = (
        split_legacy_kv_cache(
            past_key_values
        )
    )

    assert len(per_request_caches) == 3

    for request_cache in per_request_caches:
        for key, value in request_cache:
            assert key.shape == (
                1,
                4,
                12,
                8,
            )
            assert value.shape == (
                1,
                4,
                12,
                8,
            )

def test_split_preserves_correct_batch_rows() -> None:
    batch_size = 3

    key = torch.arange(
        batch_size * 2 * 4 * 2,
        dtype=torch.float32,
    ).reshape(
        batch_size,
        2,
        4,
        2,
    )

    value = key + 1000

    past_key_values = (
        (key, value),
    )

    per_request_caches = (
        split_legacy_kv_cache(
            past_key_values
        )
    )

    for batch_index in range(batch_size):
        request_key = (
            per_request_caches[
                batch_index
            ][0][0]
        )
        request_value = (
            per_request_caches[
                batch_index
            ][0][1]
        )

        assert torch.equal(
            request_key,
            key[
                batch_index
                : batch_index + 1
            ],
        )

        assert torch.equal(
            request_value,
            value[
                batch_index
                : batch_index + 1
            ],
        )

def test_split_caches_do_not_alias_original_cache() -> None:
    past_key_values = make_past_key_values(
        num_layers=2,
        batch_size=2,
        num_kv_heads=4,
        sequence_length=8,
        head_dim=16,
    )

    original_key = (
        past_key_values[0][0].clone()
    )
    original_value = (
        past_key_values[0][1].clone()
    )

    per_request_caches = (
        split_legacy_kv_cache(
            past_key_values
        )
    )

    request_zero_key = (
        per_request_caches[0][0][0]
    )
    request_zero_value = (
        per_request_caches[0][0][1]
    )

    request_zero_key.zero_()
    request_zero_value.zero_()

    assert torch.equal(
        past_key_values[0][0],
        original_key,
    )
    assert torch.equal(
        past_key_values[0][1],
        original_value,
    )

def test_stack_combines_request_caches_on_batch_dimension() -> None:
    original = make_past_key_values(
        num_layers=2,
        batch_size=3,
        num_kv_heads=4,
        sequence_length=10,
        head_dim=8,
    )

    per_request_caches = (
        split_legacy_kv_cache(
            original
        )
    )

    stacked = stack_legacy_kv_caches(
        per_request_caches
    )

    assert len(stacked) == 2

    for key, value in stacked:
        assert key.shape == (
            3,
            4,
            10,
            8,
        )
        assert value.shape == (
            3,
            4,
            10,
            8,
        )