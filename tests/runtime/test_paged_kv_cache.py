import pytest
from app.runtime.batch_builder import BatchBuilder
from app.runtime.continuous_scheduler import ContinuousScheduler
from app.runtime.kv_block_manager import KVBlockManager
from app.runtime.pytorch_model_runner import PyTorchModelRunner
import torch

from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.request import Request
from tests.runtime.fake_runner import FakeRunner, FakeBatchBuilder
from app.runtime.batch import(
    DecodeBatch,
    PrefillBatch,
)
from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
)

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

# test with padding
def test_write_request_kv_skips_left_padding() -> None:
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
            [-99.0, -99.0],  # PAD
            [-99.0, -99.0],  # PAD
            [-99.0, -99.0],  # PAD
            [10.0, 10.1],    # real token 0
            [11.0, 11.1],    # real token 1
            [12.0, 12.1],    # real token 2
        ]]]
    )

    value = torch.tensor(
        [[[
            [-999.0, -999.0],
            [-999.0, -999.0],
            [-999.0, -999.0],
            [110.0, 110.1],
            [111.0, 111.1],
            [112.0, 112.1],
        ]]]
    )

    cache.write_request_kv(
        block_table=[2],
        past_key_values=((key, value),),
        num_tokens=3,
        source_start=3,
    )

    restored = cache.materialize_request_kv(
        block_table=[2],
        num_tokens=3,
    )

    expected_key = torch.tensor(
        [[[
            [10.0, 10.1],
            [11.0, 11.1],
            [12.0, 12.1],
        ]]]
    )

    expected_value = torch.tensor(
        [[[
            [110.0, 110.1],
            [111.0, 111.1],
            [112.0, 112.1],
        ]]]
    )

    torch.testing.assert_close(
        restored[0][0],
        expected_key,
    )

    torch.testing.assert_close(
        restored[0][1],
        expected_value,
    )

# test paged kv cache integration with continuous scheduler
def make_request(
    request_id: str,
    max_new_tokens: int = 4,
    input_ids: list[int] | None = None,
) -> Request:
    if input_ids is None:
        input_ids = [1, 2, 3]

    return Request(
        request_id=request_id,
        input_ids=list(input_ids),
        max_new_tokens=max_new_tokens,
    )

@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner( head_dim=2)

def test_prefill_writes_real_kv_into_paged_cache(
        fake_runner: FakeRunner,
        batch_builder: FakeBatchBuilder,
        block_manager: KVBlockManager,
) -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=16,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    scheduler = ContinuousScheduler(
        runner=fake_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
    )

    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    scheduler.step()

    assert request.kv_tokens == 4
    assert len(request.block_table) == 1

    materialized = (
        paged_kv_cache.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    reference = request.past_key_values

    assert reference is not None

    for (
        reference_layer,
        materialized_layer,
    ) in zip(reference, materialized):
        reference_key, reference_value = reference_layer
        paged_key, paged_value = materialized_layer

        torch.testing.assert_close(
            paged_key,
            reference_key,
        )

        torch.testing.assert_close(
            paged_value,
            reference_value,
        )

def test_prefill_paged_cache_excludes_left_padding(
    fake_runner: FakeRunner,
    batch_builder: FakeBatchBuilder,
    block_manager: KVBlockManager,
) -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=16,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    scheduler = ContinuousScheduler(
        runner=fake_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
    )

    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[4, 5, 6, 7, 8, 9],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    assert request_a.kv_tokens == 3
    assert request_b.kv_tokens == 6

    paged_a = paged_kv_cache.materialize_request_kv(
        block_table=request_a.block_table,
        num_tokens=request_a.kv_tokens,
    )

    paged_b = paged_kv_cache.materialize_request_kv(
        block_table=request_b.block_table,
        num_tokens=request_b.kv_tokens,
    )

    assert paged_a[0][0].shape[2] == 3
    assert paged_b[0][0].shape[2] == 6

    reference_a = request_a.past_key_values
    reference_b = request_b.past_key_values

    assert reference_a is not None
    assert reference_b is not None

    physical_length_a = reference_a[0][0].shape[2]
    physical_length_b = reference_b[0][0].shape[2]

    assert physical_length_a == 6
    assert physical_length_b == 6

    source_start_a = (
        physical_length_a - request_a.kv_tokens
    )

    source_start_b = (
        physical_length_b - request_b.kv_tokens
    )

    assert source_start_a == 3
    assert source_start_b == 0

    for layer_idx in range(len(reference_a)):
        reference_key_a, reference_value_a = (
            reference_a[layer_idx]
        )

        expected_key_a = reference_key_a[
            :,
            :,
            source_start_a:
            source_start_a + request_a.kv_tokens,
            :,
        ]

        expected_value_a = reference_value_a[
            :,
            :,
            source_start_a:
            source_start_a + request_a.kv_tokens,
            :,
        ]

        torch.testing.assert_close(
            paged_a[layer_idx][0],
            expected_key_a,
        )

        torch.testing.assert_close(
            paged_a[layer_idx][1],
            expected_value_a,
        )

    for layer_idx in range(len(reference_b)):
        reference_key_b, reference_value_b = (
            reference_b[layer_idx]
        )

        torch.testing.assert_close(
            paged_b[layer_idx][0],
            reference_key_b,
        )

        torch.testing.assert_close(
            paged_b[layer_idx][1],
            reference_value_b,
        )

# test temperoary left-padding before decoding
def test_build_decode_batch_left_pads_variable_length_kv() -> None:
    batch_builder = BatchBuilder()

    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3],
    )
    request_b = make_request(
        request_id="B",
        input_ids=[4, 5, 6, 7, 8, 9],
    )

    # Decode needs one previously generated token.
    request_a.generated_ids = [100]
    request_b.generated_ids = [200]

    # A has logical KV length 3.
    key_a = torch.tensor(
        [[[
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ]]]
    )

    value_a = torch.tensor(
        [[[
            [11.0, 11.1],
            [12.0, 12.1],
            [13.0, 13.1],
        ]]]
    )

    # B has logical KV length 6.
    key_b = torch.tensor(
        [[[
            [21.0, 21.1],
            [22.0, 22.1],
            [23.0, 23.1],
            [24.0, 24.1],
            [25.0, 25.1],
            [26.0, 26.1],
        ]]]
    )

    value_b = torch.tensor(
        [[[
            [31.0, 31.1],
            [32.0, 32.1],
            [33.0, 33.1],
            [34.0, 34.1],
            [35.0, 35.1],
            [36.0, 36.1],
        ]]]
    )

    per_request_caches = [
        ((key_a, value_a),),
        ((key_b, value_b),),
    ]

    batch = batch_builder.build_decode_batch(
        [request_a, request_b],
        per_request_caches=per_request_caches,
        device="cpu",
    )

    assert batch.input_ids.shape == (2, 1)

    torch.testing.assert_close(
        batch.input_ids,
        torch.tensor([
            [100],
            [200],
        ]),
    )

    # Past KV must now be rectangular:
    # [batch=2, heads=1, seq_len=6, head_dim=2]
    batched_key = batch.past_key_values[0][0]
    batched_value = batch.past_key_values[0][1]

    assert batched_key.shape == (2, 1, 6, 2)
    assert batched_value.shape == (2, 1, 6, 2)

    # Request A should be left-padded by 3 positions.
    torch.testing.assert_close(
        batched_key[0, 0, :, :],
        torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ]),
    )

    torch.testing.assert_close(
        batched_value[0, 0, :, :],
        torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [11.0, 11.1],
            [12.0, 12.1],
            [13.0, 13.1],
        ]),
    )

    # Request B should not be padded.
    torch.testing.assert_close(
        batched_key[1, 0, :, :],
        key_b[0, 0],
    )

    torch.testing.assert_close(
        batched_value[1, 0, :, :],
        value_b[0, 0],
    )

    # Attention mask includes historical KV + current decode token.
    #
    # A: [PAD PAD PAD A0 A1 A2 NEW]
    #     [ 0   0   0   1  1  1   1 ]
    #
    # B: [B0 B1 B2 B3 B4 B5 NEW]
    #     [1  1  1  1  1  1   1]
    torch.testing.assert_close(
        batch.attention_mask,
        torch.tensor([
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]),
    )

def test_build_decode_batch_does_not_modify_input_caches() -> None:
    batch_builder = BatchBuilder()

    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3],
    )
    request_b = make_request(
        request_id="B",
        input_ids=[4, 5, 6, 7, 8, 9],
    )

    request_a.generated_ids = [100]
    request_b.generated_ids = [200]

    key_a = torch.randn(1, 1, 3, 2)
    value_a = torch.randn(1, 1, 3, 2)

    key_b = torch.randn(1, 1, 6, 2)
    value_b = torch.randn(1, 1, 6, 2)

    cache_a = ((key_a, value_a),)
    cache_b = ((key_b, value_b),)

    batch = batch_builder.build_decode_batch(
        [request_a, request_b],
        per_request_caches=[
            cache_a,
            cache_b,
        ],
        device="cpu",
    )

    # Temporary batching must not turn the persistent/materialized
    # logical cache itself into a padded representation.
    assert key_a.shape[2] == 3
    assert value_a.shape[2] == 3

    assert key_b.shape[2] == 6
    assert value_b.shape[2] == 6

    assert batch.position_ids is not None

    torch.testing.assert_close(
        batch.position_ids,
        torch.tensor([
            [3],
            [6],
        ]),
    )

def test_decode_writes_variable_length_kv_back_to_paged_cache(
    fake_runner: FakeRunner,
    batch_builder: BatchBuilder,
    block_manager: KVBlockManager,
) -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=16,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    scheduler = ContinuousScheduler(
        runner=fake_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
    )

    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[4, 5, 6, 7, 8, 9],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # -------------------------
    # Step 1: prefill
    # -------------------------
    scheduler.step()

    assert request_a.kv_tokens == 3
    assert request_b.kv_tokens == 6

    assert len(request_a.generated_ids) == 1
    assert len(request_b.generated_ids) == 1

    paged_a_before = (
        paged_kv_cache.materialize_request_kv(
            block_table=request_a.block_table,
            num_tokens=request_a.kv_tokens,
        )
    )

    paged_b_before = (
        paged_kv_cache.materialize_request_kv(
            block_table=request_b.block_table,
            num_tokens=request_b.kv_tokens,
        )
    )

    assert paged_a_before[0][0].shape[2] == 3
    assert paged_b_before[0][0].shape[2] == 6

    # -------------------------
    # Step 2: decode
    # -------------------------
    scheduler.step()

    # The decode input token has now entered KV.
    assert request_a.kv_tokens == 4
    assert request_b.kv_tokens == 7

    paged_a_after = (
        paged_kv_cache.materialize_request_kv(
            block_table=request_a.block_table,
            num_tokens=request_a.kv_tokens,
        )
    )

    paged_b_after = (
        paged_kv_cache.materialize_request_kv(
            block_table=request_b.block_table,
            num_tokens=request_b.kv_tokens,
        )
    )

    # Persistent paged KV must contain only logical tokens.
    assert paged_a_after[0][0].shape[2] == 4
    assert paged_a_after[0][1].shape[2] == 4

    assert paged_b_after[0][0].shape[2] == 7
    assert paged_b_after[0][1].shape[2] == 7

# block-boundary decode test
# a block-boundary decode integration test with FakeRunner;
def test_decode_crossing_block_boundary_uses_new_block(
    fake_runner: FakeRunner,
    batch_builder: BatchBuilder,
    block_manager: KVBlockManager,
) -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=16,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    scheduler = ContinuousScheduler(
        runner=fake_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
    )

    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # -------------------------
    # Prefill
    # -------------------------
    scheduler.step()

    assert request.kv_tokens == 4
    assert len(request.block_table) == 1

    first_block = request.block_table[0]

    before_decode = (
        paged_kv_cache.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    assert before_decode[0][0].shape[2] == 4

    # -------------------------
    # Decode
    # -------------------------
    scheduler.step()

    # Decode input token has now entered KV.
    assert request.kv_tokens == 5

    # 5 logical KV tokens with block_size=4
    # require 2 physical blocks.
    assert len(request.block_table) == 2

    second_block = request.block_table[1]

    assert second_block != first_block

    after_decode = (
        paged_kv_cache.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    assert after_decode[0][0].shape[2] == 5
    assert after_decode[0][1].shape[2] == 5

    # Logical token 4 must map to:
    # logical block = 4 // 4 = 1
    # slot          = 4 % 4  = 0
    #
    # therefore request.block_table[1], slot 0.
    physical_block, slot = (
        paged_kv_cache.get_physical_location(
            request.block_table,
            4,
        )
    )

    assert physical_block == second_block
    assert slot == 0

    # The materialized fifth token must equal the
    # actual value stored in second_block / slot0.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            second_block,
            0,
            0,
            :,
        ],
        after_decode[0][0][
            0,
            0,
            4,
            :,
        ],
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            0,
            second_block,
            0,
            0,
            :,
        ],
        after_decode[0][1][
            0,
            0,
            4,
            :,
        ],
    )

# a real-model round-trip test with PyTorchModelRunner.
@pytest.mark.integration
def test_real_model_prefill_round_trip_through_paged_kv(
    real_runner: PyTorchModelRunner,
) -> None:

    input_ids = real_runner.encode_prompt(
        "The capital of France is"
    )

    # Adapt this call to your actual runner API.
    output = real_runner.prefill(
        input_ids
    )

    original = output.past_key_values

    assert original is not None

    first_key, _ = original[0]

    num_layers = len(original)
    num_kv_heads = first_key.shape[1]
    num_tokens = first_key.shape[2]
    head_dim = first_key.shape[3]

    block_size = 4

    num_blocks_needed = (
        num_tokens + block_size - 1
    ) // block_size

    paged_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_blocks=num_blocks_needed + 2,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=first_key.dtype,
        device=first_key.device,
    )

    block_table = list(
        range(num_blocks_needed)
    )

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=original,
        num_tokens=num_tokens,
        source_start=0,
    )

    restored = (
        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=num_tokens,
        )
    )

    assert len(restored) == len(original)

    for (
        original_layer,
        restored_layer,
    ) in zip(original, restored):
        original_key, original_value = (
            original_layer
        )

        restored_key, restored_value = (
            restored_layer
        )

        torch.testing.assert_close(
            restored_key,
            original_key,
        )

        torch.testing.assert_close(
            restored_value,
            original_value,
        )

# final real-model decode integration test.
@pytest.mark.integration
def test_real_model_decode_round_trip_through_paged_kv() -> None:
    runner = PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.float16,
    )

    # -------------------------------------------------
    # 1. Real prefill
    # -------------------------------------------------

    prompt = "The capital of France is"

    input_ids_tensor = runner.encode_prompt(
        prompt
    )

    # Adapt this section to your existing PrefillBatch API
    # if runner.prefill_batch() requires a PrefillBatch.
    attention_mask = torch.ones_like(
        input_ids_tensor,
        dtype=torch.long,
        device=runner.device,
    )

    position_ids = torch.arange(
        input_ids_tensor.shape[1],
        dtype=torch.long,
        device=runner.device,
    ).unsqueeze(0)

    prefill_batch = PrefillBatch(
        request_ids=["A"],
        input_ids=input_ids_tensor.to(
            runner.device
        ),
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    prefill_output = runner.prefill_batch(
        prefill_batch
    )

    original_prefill_kv = (
        prefill_output.past_key_values
    )

    assert original_prefill_kv is not None

    first_key = original_prefill_kv[0][0]

    num_layers = len(
        original_prefill_kv
    )
    num_kv_heads = int(
        first_key.shape[1]
    )
    prompt_kv_length = int(
        first_key.shape[2]
    )
    head_dim = int(
        first_key.shape[3]
    )

    assert prompt_kv_length == (
        input_ids_tensor.shape[1]
    )

    # -------------------------------------------------
    # 2. Create paged storage
    # -------------------------------------------------

    block_size = 4

    # Need enough capacity for prompt + one decode token.
    max_tokens_needed = (
        prompt_kv_length + 1
    )

    num_blocks_needed = (
        max_tokens_needed
        + block_size
        - 1
    ) // block_size

    paged_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_blocks=num_blocks_needed + 2,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=first_key.dtype,
        device=first_key.device,
    )

    block_table = list(
        range(num_blocks_needed)
    )

    # -------------------------------------------------
    # 3. Persist prefill KV
    # -------------------------------------------------

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=original_prefill_kv,
        num_tokens=prompt_kv_length,
        source_start=0,
    )

    materialized_prefill_kv = (
        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=prompt_kv_length,
        )
    )

    for (
        original_layer,
        materialized_layer,
    ) in zip(
        original_prefill_kv,
        materialized_prefill_kv,
    ):
        original_key, original_value = (
            original_layer
        )

        materialized_key, materialized_value = (
            materialized_layer
        )

        torch.testing.assert_close(
            materialized_key,
            original_key,
        )

        torch.testing.assert_close(
            materialized_value,
            original_value,
        )

    # -------------------------------------------------
    # 4. Real decode using MATERIALIZED paged KV
    # -------------------------------------------------

    decode_input_id = int(
        prefill_output.next_token_ids[0]
    )

    decode_input_ids = torch.tensor(
        [[decode_input_id]],
        dtype=torch.long,
        device=runner.device,
    )

    # Past KV contains N valid tokens.
    # Decode input is token N.
    decode_position_ids = torch.tensor(
        [[prompt_kv_length]],
        dtype=torch.long,
        device=runner.device,
    )

    decode_attention_mask = torch.ones(
        (
            1,
            prompt_kv_length + 1,
        ),
        dtype=torch.long,
        device=runner.device,
    )

    decode_batch = DecodeBatch(
        request_ids=["A"],
        input_ids=decode_input_ids,
        past_key_values=materialized_prefill_kv,
        attention_mask=decode_attention_mask,
        position_ids=decode_position_ids,
    )

    decode_output = runner.decode_batch(
        decode_batch
    )

    updated_kv = (
        decode_output.past_key_values
    )

    assert updated_kv is not None

    new_kv_length = (
        get_kv_sequence_length(
            updated_kv
        )
    )

    assert new_kv_length == (
        prompt_kv_length + 1
    )

    # -------------------------------------------------
    # 5. Write decode result back into paged storage
    # -------------------------------------------------

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=updated_kv,
        num_tokens=new_kv_length,
        source_start=0,
    )

    restored_after_decode = (
        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=new_kv_length,
        )
    )

    # -------------------------------------------------
    # 6. Paged representation must match real model KV
    # -------------------------------------------------

    assert len(
        restored_after_decode
    ) == len(updated_kv)

    for (
        model_layer,
        paged_layer,
    ) in zip(
        updated_kv,
        restored_after_decode,
    ):
        model_key, model_value = (
            model_layer
        )

        paged_key, paged_value = (
            paged_layer
        )

        torch.testing.assert_close(
            paged_key,
            model_key,
        )

        torch.testing.assert_close(
            paged_value,
            model_value,
        )
