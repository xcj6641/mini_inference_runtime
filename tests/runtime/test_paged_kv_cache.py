import pytest
import torch

from app.runtime.paged_kv_cache import PagedKVCache


def test_constructor_creates_expected_cache_shapes() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        num_kv_heads=2,
        block_size=4,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    expected_shape = (2, 8, 2, 4, 8)

    assert cache.key_cache.shape == expected_shape
    assert cache.value_cache.shape == expected_shape


def test_constructor_uses_requested_dtype() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        num_kv_heads=2,
        block_size=4,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    assert cache.key_cache.dtype == torch.float32
    assert cache.value_cache.dtype == torch.float32


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_layers", 0),
        ("num_blocks", 0),
        ("num_kv_heads", 0),
        ("block_size", 0),
        ("head_dim", 0),
    ],
)

def test_constructor_rejects_zero_dimensions(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "num_layers": 2,
        "num_blocks": 8,
        "num_kv_heads": 2,
        "block_size": 4,
        "head_dim": 8,
        "dtype": torch.float32,
        "device": "cpu",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        PagedKVCache(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "num_layers",
        "num_blocks",
        "num_kv_heads",
        "block_size",
        "head_dim",
    ],
)
def test_constructor_rejects_negative_dimensions(
    field: str,
) -> None:
    kwargs = {
        "num_layers": 2,
        "num_blocks": 8,
        "num_kv_heads": 2,
        "block_size": 4,
        "head_dim": 8,
        "dtype": torch.float32,
        "device": "cpu",
    }

    kwargs[field] = -1

    with pytest.raises(ValueError):
        PagedKVCache(**kwargs)


def make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_blocks=8,
        num_kv_heads=2,
        block_size=4,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )


def test_get_physical_location_first_token() -> None:
    cache = make_cache()

    location = cache.get_physical_location(
        [7, 2, 5],
        0,
    )

    assert location == (7, 0)


def test_get_physical_location_last_slot_of_first_block() -> None:
    cache = make_cache()

    location = cache.get_physical_location(
        [7, 2, 5],
        3,
    )

    assert location == (7, 3)


def test_get_physical_location_crosses_block_boundary() -> None:
    cache = make_cache()

    location = cache.get_physical_location(
        [7, 2, 5],
        4,
    )

    assert location == (2, 0)


def test_get_physical_location_multiple_blocks() -> None:
    cache = make_cache()

    assert cache.get_physical_location([7, 2, 5], 5) == (2, 1)
    assert cache.get_physical_location([7, 2, 5], 8) == (5, 0)

# error cases
def test_get_physical_location_rejects_negative_token_index() -> None:
    cache = make_cache()

    with pytest.raises(ValueError):
        cache.get_physical_location(
            [7, 2],
            -1,
        )


def test_get_physical_location_rejects_short_block_table() -> None:
    cache = make_cache()

    with pytest.raises(IndexError):
        cache.get_physical_location(
            [7],
            4,
        )


@pytest.mark.parametrize(
    "block_table",
    [
        [-1],
        [8],
        [100],
    ],
)
def test_get_physical_location_rejects_invalid_block_id(
    block_table: list[int],
) -> None:
    cache = make_cache()

    with pytest.raises(ValueError):
        cache.get_physical_location(
            block_table,
            0,
        )

# write token k 
def test_write_request_kv_less_than_one_block() -> None:
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    key = torch.tensor(
        [[[
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ]]]
    )

    value = torch.tensor(
        [[[
            [110.0, 111.0],
            [120.0, 121.0],
            [130.0, 131.0],
        ]]]
    )

    past_key_values = (
        (key, value),
    )

    cache.write_request_kv(
        block_table=[2],
        past_key_values=past_key_values,
        num_tokens=3,
    )

    torch.testing.assert_close(
        cache.key_cache[0, 2, 0, 0, :],
        torch.tensor([10.0, 11.0]),
    )

    torch.testing.assert_close(
        cache.key_cache[0, 2, 0, 1, :],
        torch.tensor([20.0, 21.0]),
    )

    torch.testing.assert_close(
        cache.key_cache[0, 2, 0, 2, :],
        torch.tensor([30.0, 31.0]),
    )

    torch.testing.assert_close(
        cache.value_cache[0, 2, 0, 0, :],
        torch.tensor([110.0, 111.0])
    )

def test_write_request_kv_crosses_block_boundary() -> None:

    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    key = torch.tensor(
        [[[
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
            [4.0, 4.1],
            [5.0, 5.1],
        ]]]
    )

    value = torch.tensor(
        [[[
            [10.0, 10.1],
            [11.0, 11.1],
            [12.0, 12.1],
            [13.0, 13.1],
            [14.0, 14.1],
            [15.0, 15.1]
        ]]]
    )

    past_key_values = (
        (key, value),
    )

    cache.write_request_kv(
        block_table=[2, 6],
        past_key_values=past_key_values,
        num_tokens=6,
    )

    torch.testing.assert_close(
        cache.key_cache[0, 2, 0, 3, :],
        torch.tensor([3.0, 3.1]),
    )

    torch.testing.assert_close(
        cache.key_cache[0, 6, 0, 0, :],
        torch.tensor([4.0, 4.1]),
    )

    torch.testing.assert_close(
        cache.key_cache[0, 6, 0, 1, :],
        torch.tensor([5.0, 5.1]),
    )

    torch.testing.assert_close(
        cache.value_cache[0, 2, 0, 2, :],
        torch.tensor([12.0, 12.1]),
    )
    torch.testing.assert_close(
        cache.value_cache[0, 2, 0, 3, :],
        torch.tensor([13.0, 13.1]),
    )
    torch.testing.assert_close(
        cache.value_cache[0, 6, 0, 0, :],
        torch.tensor([14.0, 14.1]),
    )
    torch.testing.assert_close(
        cache.value_cache[0, 6, 0, 1, :],
        torch.tensor([15.0, 15.1]),
    )

def test_write_request_kv_multiple_layers() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=16,
        num_kv_heads=2,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    key1 = torch.tensor(
        [[[
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
            [4.0, 4.1],
            [5.0, 5.1],
        ],
        [
            [10.0, 10.1],
            [11.0, 11.1],
            [12.0, 12.1],
            [13.0, 13.1],
            [14.0, 14.1],
            [15.0, 15.1]
        ]]]
    )
    key2 = torch.tensor(
        [[[
            [16.0, 16.1],
            [17.0, 17.1],
            [18.0, 18.1],
            [19.0, 19.1],
            [20.0, 20.1],
            [21.0, 21.1],
        ],
        [
            [22.0, 22.1],
            [23.0, 23.1],
            [24.0, 24.1],
            [25.0, 25.1],
            [26.0, 26.1],
            [27.0, 27.1]
        ]]]
    )

    value1 = torch.tensor(
        [[[
            [100.0, 100.1],
            [101.0, 101.1],
            [102.0, 102.1],
            [103.0, 103.1],
            [104.0, 104.1],
            [105.0, 105.1]
        ],
        [
            [110.0, 110.1],
            [111.0, 111.1],
            [112.0, 112.1],
            [113.0, 113.1],
            [114.0, 114.1],
            [115.0, 115.1]
        ]]]
    )

    value2 = torch.tensor(
        [[[
            [116.0, 116.1],
            [117.0, 117.1],
            [118.0, 118.1],
            [119.0, 119.1],
            [120.0, 120.1],
            [121.0, 121.1]
        ],
        [
            [122.0, 122.1],
            [123.0, 123.1],
            [124.0, 124.1],
            [125.0, 125.1],
            [126.0, 126.1],
            [127.0, 127.1]
        ]]]
    )

    past_key_values = (
        (key1, value1),
        (key2, value2),
    )

    cache.write_request_kv(
        block_table=[2, 9],
        past_key_values=past_key_values,
        num_tokens=6,
    )

    #layer 0
    torch.testing.assert_close(
        cache.key_cache[0, 2, 0, 3, :],
        torch.tensor([3.0, 3.1]),
    )
    torch.testing.assert_close(
        cache.key_cache[0, 9, 0, 1, :],
        torch.tensor([5.0, 5.1]),
    )
    torch.testing.assert_close(
        cache.key_cache[0, 2, 1, 0, :],
        torch.tensor([10.0, 10.1]),
    )
    torch.testing.assert_close(
        cache.key_cache[0, 9, 1, 1, :],
        torch.tensor([15.0, 15.1]),
    )

    torch.testing.assert_close(
        cache.value_cache[0, 9, 1, 1, :],
        torch.tensor([115.0, 115.1]),
    )

    #layer 1
    torch.testing.assert_close(
        cache.key_cache[1, 9, 0, 1, :],
        torch.tensor([21.0, 21.1]),
    )
    

    torch.testing.assert_close(
        cache.key_cache[1, 2, 1, 2, :],
        torch.tensor([24.0, 24.1]),
    )

    torch.testing.assert_close(
        cache.value_cache[1, 9, 1, 1, :],
        torch.tensor([127.0, 127.1]),
    )

# test round trip
def test_write_then_materialize_round_trip() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=16,
        num_kv_heads=2,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    key0 = torch.randn(1, 2, 6, 2)
    value0 = torch.randn(1, 2, 6, 2)

    key1 = torch.randn(1, 2, 6, 2)
    value1 = torch.randn(1, 2, 6, 2)

    original = (
        (key0, value0),
        (key1, value1),
    )

    cache.write_request_kv(
        block_table=[2, 9],
        past_key_values=original,
        num_tokens=6,
    )

    restored = cache.materialize_request_kv(
        block_table=[2, 9],
        num_tokens=6,
    )

    assert len(restored) == 2

    for (original_k, original_v), (
        restored_k,
        restored_v,
    ) in zip(original, restored):
        torch.testing.assert_close(
            restored_k,
            original_k,
        )
        torch.testing.assert_close(
            restored_v,
            original_v,
        )

    assert restored[0][0].shape == (1, 2, 6, 2)
    assert restored[0][1].shape == (1, 2, 6, 2)

def test_multiple_requests_are_isolated() -> None:
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    # Request A: 6 tokens
    key_a = torch.ones(1, 1, 6, 2)
    value_a = torch.ones(1, 1, 6, 2) * 10

    kv_a = (
        (key_a, value_a),
    )

    # Request B: 3 tokens
    key_b = torch.ones(1, 1, 3, 2) * 9
    value_b = torch.ones(1, 1, 3, 2) * 90

    kv_b = (
        (key_b, value_b),
    )

    cache.write_request_kv(
        block_table=[0, 3],
        past_key_values=kv_a,
        num_tokens=6,
    )

    cache.write_request_kv(
        block_table=[5],
        past_key_values=kv_b,
        num_tokens=3,
    )

    restored_a = cache.materialize_request_kv(
        block_table=[0, 3],
        num_tokens=6,
    )

    restored_b = cache.materialize_request_kv(
        block_table=[5],
        num_tokens=3,
    )

    torch.testing.assert_close(
        restored_a[0][0],
        key_a,
    )

    torch.testing.assert_close(
        restored_a[0][1],
        value_a,
    )

    torch.testing.assert_close(
        restored_b[0][0],
        key_b,
    )

    torch.testing.assert_close(
        restored_b[0][1],
        value_b,
    )

    key_b_new = torch.ones(1, 1, 3, 2) * 99
    value_b_new = torch.ones(1, 1, 3, 2) * 999

    kv_b_new = (
        (key_b_new, value_b_new),
    )

    cache.write_request_kv(
        block_table=[5],
        past_key_values=kv_b_new,
        num_tokens=3,
    )
    restored_a_after_b_overwrite = (
        cache.materialize_request_kv(
            block_table=[0, 3],
            num_tokens=6,
        )
    )

    torch.testing.assert_close(
        restored_a_after_b_overwrite[0][0],
        key_a,
    )

    torch.testing.assert_close(
        restored_a_after_b_overwrite[0][1],
        value_a,
    )
    restored_b_new = cache.materialize_request_kv(
        block_table=[5],
        num_tokens=3,
    )

    torch.testing.assert_close(
        restored_b_new[0][0],
        key_b_new,
    )

    torch.testing.assert_close(
        restored_b_new[0][1],
        value_b_new,
    )