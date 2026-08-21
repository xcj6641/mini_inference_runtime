import pytest

from app.runtime.batch_builder import BatchBuilder
from app.runtime.continuous_scheduler import (
    ContinuousScheduler,
)
from app.runtime.kv_block_manager import KVBlockManager
from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
)
from app.runtime.pytorch_model_runner import (
    PyTorchModelRunner,
)
from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.prefix_cache import PrefixCache
from app.runtime.request import (
    Request,
    RequestState,
)


@pytest.mark.integration
def test_scheduler_decode_updates_real_kv_cache(
    real_runner: PyTorchModelRunner,
    paged_kv_cache_qwen: PagedKVCache,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache_qwen,
        prefix_cache=prefix_cache,
    )

    input_ids = (
        real_runner.encode_prompt(
            "The capital of France is"
        )
        .squeeze(0)
        .tolist()
    )

    request = Request(
        request_id="request-a",
        input_ids=input_ids,
        max_new_tokens=4,
    )

    scheduler.add_request(request)

    # -------------------------
    # Step 1: prefill
    # -------------------------
    prefill_result = scheduler.step()

    assert prefill_result.prefetched_request_ids == [
        "request-a"
    ]

    assert (
        prefill_result.decoded_request_ids
        == []
    )

    assert request.state in {
        RequestState.DECODING,
        RequestState.FINISHED,
    }

    assert len(request.generated_ids) == 1

    assert (
        request.generated_tokens_count
        == 1
    )

    # Request no longer owns past_key_values.
    # Materialize KV from PagedKVCache.
    kv_after_prefill = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    kv_length_after_prefill = (
        get_kv_sequence_length(
            kv_after_prefill
        )
    )

    assert kv_length_after_prefill == len(
        request.input_ids
    )

    assert request.kv_tokens == len(
        request.input_ids
    )

    first_generated_token = (
        request.generated_ids[0]
    )

    # -------------------------
    # Step 2: decode
    # -------------------------

    # Before decode, scheduler must have enough
    # physical blocks for one additional KV token.
    required_blocks_after_decode = (
        request.kv_tokens
        + 1
        + block_manager.block_size
        - 1
    ) // block_manager.block_size

    # This does not require capacity to have been
    # allocated yet; scheduler.step() may allocate
    # the extra block during decode selection.
    assert (
        len(request.block_table)
        <= required_blocks_after_decode
    )

    decode_result = scheduler.step()

    assert (
        decode_result.prefetched_request_ids
        == []
    )

    assert decode_result.decoded_request_ids == [
        "request-a"
    ]

    assert request.state in {
        RequestState.DECODING,
        RequestState.FINISHED,
    }

    assert len(request.generated_ids) == 2

    assert (
        request.generated_tokens_count
        == 2
    )

    assert (
        request.generated_ids[0]
        == first_generated_token
    )

    kv_after_decode = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    kv_length_after_decode = (
        get_kv_sequence_length(
            kv_after_decode
        )
    )

    assert kv_length_after_decode == (
        kv_length_after_prefill + 1
    )

    first_key, first_value = (
        kv_after_decode[0]
    )

    assert first_key.shape[0] == 1
    assert first_value.shape[0] == 1

    assert (
        first_key.shape
        == first_value.shape
    )


@pytest.mark.integration
def test_scheduler_real_batched_decode_two_requests(
    real_runner: PyTorchModelRunner,
    paged_kv_cache_qwen: PagedKVCache,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache_qwen,
        prefix_cache=prefix_cache,
    )

    input_ids_a = (
        real_runner.encode_prompt(
            "The capital of France is"
        )
        .squeeze(0)
        .tolist()
    )

    input_ids_b = (
        real_runner.encode_prompt(
            "The capital of Germany is"
        )
        .squeeze(0)
        .tolist()
    )

    assert len(input_ids_a) == len(
        input_ids_b
    )

    request_a = Request(
        request_id="request-a",
        input_ids=input_ids_a,
        max_new_tokens=4,
    )

    request_b = Request(
        request_id="request-b",
        input_ids=input_ids_b,
        max_new_tokens=4,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # -------------------------
    # Step 1: batched prefill
    # -------------------------
    prefill_result = scheduler.step()

    assert set(
        prefill_result.prefetched_request_ids
    ) == {
        "request-a",
        "request-b",
    }

    assert request_a.state in {
        RequestState.DECODING,
        RequestState.FINISHED,
    }

    assert request_b.state in {
        RequestState.DECODING,
        RequestState.FINISHED,
    }

    assert len(request_a.generated_ids) == 1
    assert len(request_b.generated_ids) == 1

    kv_a_after_prefill = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request_a.block_table,
            num_tokens=request_a.kv_tokens,
        )
    )

    kv_b_after_prefill = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request_b.block_table,
            num_tokens=request_b.kv_tokens,
        )
    )

    kv_length_a_after_prefill = (
        get_kv_sequence_length(
            kv_a_after_prefill
        )
    )

    kv_length_b_after_prefill = (
        get_kv_sequence_length(
            kv_b_after_prefill
        )
    )

    assert (
        kv_length_a_after_prefill
        == len(request_a.input_ids)
    )

    assert (
        kv_length_b_after_prefill
        == len(request_b.input_ids)
    )

    assert (
        kv_length_a_after_prefill
        == kv_length_b_after_prefill
    )

    # -------------------------
    # Step 2: batched decode
    # -------------------------
    decode_result = scheduler.step()

    assert set(
        decode_result.decoded_request_ids
    ) == {
        "request-a",
        "request-b",
    }

    assert len(request_a.generated_ids) == 2
    assert len(request_b.generated_ids) == 2

    assert (
        request_a.generated_tokens_count
        == 2
    )

    assert (
        request_b.generated_tokens_count
        == 2
    )

    kv_a_after_decode = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request_a.block_table,
            num_tokens=request_a.kv_tokens,
        )
    )

    kv_b_after_decode = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request_b.block_table,
            num_tokens=request_b.kv_tokens,
        )
    )

    kv_length_a_after_decode = (
        get_kv_sequence_length(
            kv_a_after_decode
        )
    )

    kv_length_b_after_decode = (
        get_kv_sequence_length(
            kv_b_after_decode
        )
    )

    assert kv_length_a_after_decode == (
        kv_length_a_after_prefill + 1
    )

    assert kv_length_b_after_decode == (
        kv_length_b_after_prefill + 1
    )

    first_key_a, first_value_a = (
        kv_a_after_decode[0]
    )

    first_key_b, first_value_b = (
        kv_b_after_decode[0]
    )

    assert first_key_a.shape[0] == 1
    assert first_value_a.shape[0] == 1

    assert first_key_b.shape[0] == 1
    assert first_value_b.shape[0] == 1

    assert (
        first_key_a.shape
        == first_key_b.shape
    )

    assert (
        first_value_a.shape
        == first_value_b.shape
    )


@pytest.mark.integration
def test_scheduler_real_request_finishes_by_length(
    real_runner: PyTorchModelRunner,
    paged_kv_cache_qwen: PagedKVCache,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache_qwen,
        prefix_cache=prefix_cache,
    )

    input_ids = (
        real_runner.encode_prompt(
            "The capital of France is"
        )
        .squeeze(0)
        .tolist()
    )

    request = Request(
        request_id="request-a",
        input_ids=input_ids,
        max_new_tokens=2,
    )

    scheduler.add_request(request)

    # -------------------------
    # Step 1: prefill
    # -------------------------
    prefill_result = scheduler.step()

    assert (
        request.state
        == RequestState.DECODING
    )

    assert (
        request.generated_tokens_count
        == 1
    )

    assert (
        request.request_id
        in scheduler.active
    )

    assert (
        prefill_result.finished_request_ids
        == []
    )

    # Validate paged-KV state after prefill.
    kv_after_prefill = (
        paged_kv_cache_qwen.materialize_request_kv(
            block_table=request.block_table,
            num_tokens=request.kv_tokens,
        )
    )

    assert (
        get_kv_sequence_length(
            kv_after_prefill
        )
        == request.kv_tokens
    )

    # Debug/invariant information for the
    # decode capacity issue.
    print(
        "before decode:",
        {
            "kv_tokens": request.kv_tokens,
            "block_table": request.block_table,
            "num_free_blocks":
                block_manager.num_free_blocks,
            "block_size":
                block_manager.block_size,
        },
    )

    # -------------------------
    # Step 2: decode
    # -------------------------
    decode_result = scheduler.step()

    assert (
        request.state
        == RequestState.FINISHED
    )

    assert (
        request.finish_reason
        == "length"
    )

    assert (
        request.generated_tokens_count
        == 2
    )

    assert (
        request.request_id
        not in scheduler.active
    )

    assert (
        request.request_id
        in scheduler.completed
    )

    assert decode_result.decoded_request_ids == [
        "request-a"
    ]

    assert decode_result.finished_request_ids == [
        "request-a"
    ]

    assert (
        scheduler.has_pending_work()
        is False
    )


@pytest.mark.integration
def test_real_batched_decode_one_finishes_other_continues(
    real_runner: PyTorchModelRunner,
    paged_kv_cache_qwen: PagedKVCache,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache_qwen,
        prefix_cache=prefix_cache,
    )

    input_ids_a = (
        real_runner.encode_prompt(
            "The capital of France is"
        )
        .squeeze(0)
        .tolist()
    )

    input_ids_b = (
        real_runner.encode_prompt(
            "The capital of Germany is"
        )
        .squeeze(0)
        .tolist()
    )

    assert len(input_ids_a) == len(
        input_ids_b
    )

    request_a = Request(
        request_id="request-a",
        input_ids=input_ids_a,
        max_new_tokens=2,
    )

    request_b = Request(
        request_id="request-b",
        input_ids=input_ids_b,
        max_new_tokens=4,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # -------------------------
    # Step 1: prefill
    # -------------------------
    scheduler.step()

    assert (
        request_a.generated_tokens_count
        == 1
    )

    assert (
        request_b.generated_tokens_count
        == 1
    )

    print(
        "before batched decode:",
        {
            "A": {
                "kv_tokens":
                    request_a.kv_tokens,
                "block_table":
                    request_a.block_table,
            },
            "B": {
                "kv_tokens":
                    request_b.kv_tokens,
                "block_table":
                    request_b.block_table,
            },
            "num_free_blocks":
                block_manager.num_free_blocks,
        },
    )

    # -------------------------
    # Step 2: batched decode
    # -------------------------
    result = scheduler.step()

    assert (
        request_a.state
        == RequestState.FINISHED
    )

    assert (
        request_a.finish_reason
        == "length"
    )

    assert (
        request_a.request_id
        in scheduler.completed
    )

    assert (
        request_a.request_id
        not in scheduler.active
    )

    assert (
        request_b.state
        == RequestState.DECODING
    )

    assert (
        request_b.request_id
        in scheduler.active
    )

    assert (
        request_b.request_id
        not in scheduler.completed
    )

    assert result.decoded_request_ids == [
        "request-a",
        "request-b",
    ]

    assert result.finished_request_ids == [
        "request-a"
    ]

    # -------------------------
    # Step 3: decode B only
    # -------------------------
    next_result = scheduler.step()

    assert next_result.decoded_request_ids == [
        "request-b"
    ]

    assert (
        request_b.generated_tokens_count
        == 3
    )