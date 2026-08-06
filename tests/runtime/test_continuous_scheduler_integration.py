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
from app.runtime.request import (
    Request,
    RequestState,
)


@pytest.mark.integration
def test_scheduler_decode_updates_real_kv_cache(
    real_runner: PyTorchModelRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=block_manager,
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

    # Step 1: prefill.
    prefill_result = scheduler.step()

    assert prefill_result.prefetched_request_ids == [
        "request-a"
    ]
    assert prefill_result.decoded_request_ids == []
    assert request.state in {RequestState.DECODING, RequestState.FINISHED}
    assert request.past_key_values is not None
    assert len(request.generated_ids) == 1
    assert request.generated_tokens_count == 1
    

    kv_length_after_prefill = (
        get_kv_sequence_length(
            request.past_key_values
        )
    )

    assert kv_length_after_prefill == len(
        request.input_ids
    )

    first_generated_token = (
        request.generated_ids[0]
    )

    # Step 2: decode the first generated token.
    decode_result = scheduler.step()
    assert decode_result.prefetched_request_ids == []
    assert decode_result.decoded_request_ids == [
        "request-a"
    ]
    assert request.state in {RequestState.DECODING, RequestState.FINISHED}
    assert request.past_key_values is not None

    assert len(request.generated_ids) == 2
    assert request.generated_tokens_count == 2

    assert request.generated_ids[0] == (
        first_generated_token
    )

    kv_length_after_decode = (
        get_kv_sequence_length(
            request.past_key_values
        )
    )

    assert kv_length_after_decode == (
        kv_length_after_prefill + 1
    )

    first_key, first_value = (
        request.past_key_values[0]
    )

    assert first_key.shape[0] == 1
    assert first_value.shape[0] == 1
    assert first_key.shape == first_value.shape    


@pytest.mark.integration
def test_scheduler_real_batched_decode_two_requests(
    real_runner: PyTorchModelRunner,
) -> None:
    block_manager = KVBlockManager(
            num_blocks=8,
            block_size=4,
        )
    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
        block_manager=block_manager,
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

    assert len(input_ids_a) == len(input_ids_b)

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

    # Step 1: both requests are prefetched together.
    scheduler.step()

    assert request_a.state in {RequestState.DECODING, RequestState.FINISHED}
    assert request_b.state in {RequestState.DECODING, RequestState.FINISHED}

    assert request_a.past_key_values is not None
    assert request_b.past_key_values is not None

    assert len(request_a.generated_ids) == 1
    assert len(request_b.generated_ids) == 1

    kv_length_a_after_prefill = (
        get_kv_sequence_length(
            request_a.past_key_values
        )
    )

    kv_length_b_after_prefill = (
        get_kv_sequence_length(
            request_b.past_key_values
        )
    )

    assert kv_length_a_after_prefill == (
        kv_length_b_after_prefill
    )

    # Step 2: both requests are decoded together.
    scheduler.step()

    assert request_a.state in {RequestState.DECODING, RequestState.FINISHED}
    assert request_b.state in {RequestState.DECODING, RequestState.FINISHED}

    assert request_a.past_key_values is not None
    assert request_b.past_key_values is not None

    assert len(request_a.generated_ids) == 2
    assert len(request_b.generated_ids) == 2

    assert request_a.generated_tokens_count == 2
    assert request_b.generated_tokens_count == 2

    kv_length_a_after_decode = (
        get_kv_sequence_length(
            request_a.past_key_values
        )
    )

    kv_length_b_after_decode = (
        get_kv_sequence_length(
            request_b.past_key_values
        )
    )

    assert kv_length_a_after_decode == (
        kv_length_a_after_prefill + 1
    )

    assert kv_length_b_after_decode == (
        kv_length_b_after_prefill + 1
    )

    first_key_a, first_value_a = (
        request_a.past_key_values[0]
    )

    first_key_b, first_value_b = (
        request_b.past_key_values[0]
    )

    # The batched KV cache must be split back into
    # request-local caches.
    assert first_key_a.shape[0] == 1
    assert first_value_a.shape[0] == 1
    assert first_key_b.shape[0] == 1
    assert first_value_b.shape[0] == 1

    assert first_key_a.shape == first_key_b.shape
    assert first_value_a.shape == first_value_b.shape

    # Requests must not share the same tensor storage.
    assert first_key_a.data_ptr() != (
        first_key_b.data_ptr()
    )

    assert first_value_a.data_ptr() != (
        first_value_b.data_ptr()
    )

@pytest.mark.integration
def test_scheduler_real_request_finishes_by_length(
    real_runner: PyTorchModelRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=block_manager,
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

    prefill_result = scheduler.step()

    assert request.state == RequestState.DECODING
    assert request.generated_tokens_count == 1
    assert request.request_id in scheduler.active
    assert request.past_key_values is not None
    assert prefill_result.finished_request_ids == []

    decode_result = scheduler.step()

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"
    assert request.generated_tokens_count == 2

    assert request.request_id not in scheduler.active
    assert request.request_id in scheduler.completed
    assert request.past_key_values is None

    assert decode_result.decoded_request_ids == [
        "request-a"
    ]

    assert decode_result.finished_request_ids == [
        "request-a"
    ]

    assert scheduler.has_pending_work() is False

@pytest.mark.integration
def test_real_batched_decode_one_finishes_other_continues(
    real_runner: PyTorchModelRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )
    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
        block_manager=block_manager,
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

    assert len(input_ids_a) == len(input_ids_b)

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

    # Prefill produces token 1 for both.
    scheduler.step()

    assert request_a.generated_tokens_count == 1
    assert request_b.generated_tokens_count == 1

    # Decode produces token 2 for both.
    # A reaches max_new_tokens, B continues.
    result = scheduler.step()

    assert request_a.state == RequestState.FINISHED
    assert request_a.finish_reason == "length"
    assert request_a.past_key_values is None
    assert request_a.request_id in scheduler.completed
    assert request_a.request_id not in scheduler.active

    assert request_b.state == RequestState.DECODING
    assert request_b.past_key_values is not None
    assert request_b.request_id in scheduler.active
    assert request_b.request_id not in scheduler.completed

    assert result.decoded_request_ids == [
        "request-a",
        "request-b",
    ]

    assert result.finished_request_ids == [
        "request-a"
    ]

    # Next step should decode only B.
    next_result = scheduler.step()

    assert next_result.decoded_request_ids == [
        "request-b"
    ]

    assert request_b.generated_tokens_count == 3