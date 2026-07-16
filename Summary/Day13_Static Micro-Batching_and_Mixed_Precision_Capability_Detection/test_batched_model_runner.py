from itertools import batched

import pytest
from app.runtime.batch import DecodeBatch
import torch

from app.runtime.batch_builder import BatchBuilder
from test_batch_builder import make_request


from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
    split_batched_legacy_kv_cache,
    stack_legacy_kv_caches,
)

@pytest.mark.integration
def test_equal_length_batched_prefill_matches_single_prefill(
    real_runner,
) -> None:
    prompt_a = "The capital of France is"
    prompt_b = "The capital of Germany is"

    input_ids_a_tensor = real_runner.encode_prompt(prompt_a)# 2d tensor
    input_ids_b_tensor = real_runner.encode_prompt(prompt_b)


    #1d list, fit Class Request
    input_ids_a = (
        input_ids_a_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    input_ids_b = (
        input_ids_b_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    assert len(input_ids_a) == len(input_ids_b)

    request_a = make_request("A", input_ids_a)
    request_b = make_request("B", input_ids_b)

    single_input_ids_a = torch.tensor([input_ids_a], dtype=torch.long)
    single_input_ids_b = torch.tensor([input_ids_b], dtype=torch.long)

    single_attention_mask_a = torch.ones_like(single_input_ids_a, dtype=torch.long)
    single_attention_mask_b = torch.ones_like(single_input_ids_b, dtype=torch.long)

    single_a = real_runner.prefill(
        input_ids=single_input_ids_a,
        attention_mask=single_attention_mask_a,
    )

    single_b = real_runner.prefill(
        input_ids=single_input_ids_b,
        attention_mask=single_attention_mask_b,
    )

    list_of_requests = [request_a, request_b]
    batch_requests = BatchBuilder().build_equal_length_prefill_batch(
        list_of_requests
    )

    batched_output = real_runner.prefill_batch(batch_requests)

    assert batched_output.next_token_ids == [
        single_a.next_token_id,
        single_b.next_token_id,
    ]

    assert batched_output.logits.shape[0] == 2
    assert len(batched_output.next_token_ids) == 2
    assert batched_output.past_key_values is not None
    assert torch.isfinite(batched_output.logits).all()

    request_caches = split_batched_legacy_kv_cache(
        batched_output.past_key_values
    )

    if len(list_of_requests) != len(batched_output.next_token_ids):
        raise RuntimeError(
            "Number of next tokens does not match requests"
        )

    if len(list_of_requests) != len(request_caches):
        raise RuntimeError(
            "Number of KV caches does not match requests"
        )

    for request, next_token_id, request_cache in zip(
        list_of_requests,
        batched_output.next_token_ids,
        request_caches,
        strict=True,
    ):
        request.attach_kv_cache(request_cache)
        request.append_generated_token(next_token_id)

    assert request_a.past_key_values is not None
    assert request_b.past_key_values is not None

    assert request_a.generated_ids == [
        batched_output.next_token_ids[0]
    ]

    assert request_b.generated_ids == [
        batched_output.next_token_ids[1]
    ]
    assert get_kv_sequence_length(
        request_a.past_key_values
    ) == len(request_a.input_ids)

    assert get_kv_sequence_length(
        request_b.past_key_values
    ) == len(request_b.input_ids)



@pytest.mark.integration
def test_split_batched_legacy_kv_cache() -> None:
    key = torch.tensor(
        [
            [[[10.0], [11.0]]],
            [[[20.0], [21.0]]],
        ]
    )

    value = torch.tensor(
        [
            [[[100.0], [101.0]]],
            [[[200.0], [201.0]]],
        ]
    )

    batched_cache = (
        (key, value),
    )

    request_caches = (
        split_batched_legacy_kv_cache(
            batched_cache
        )
    )

    assert len(request_caches) == 2

    request_a_key, request_a_value = (
        request_caches[0][0]
    )
    request_b_key, request_b_value = (
        request_caches[1][0]
    )

    assert request_a_key.shape[0] == 1
    assert request_b_key.shape[0] == 1

    assert torch.equal(
        request_a_key,
        key[0:1],
    )
    assert torch.equal(
        request_a_value,
        value[0:1],
    )

    assert torch.equal(
        request_b_key,
        key[1:2],
    )
    assert torch.equal(
        request_b_value,
        value[1:2],
    )

@pytest.mark.integration
def test_build_variable_length_prefill_batch() -> None:
    request_a = make_request("A", [11, 12])
    request_b = make_request("B", [21, 22, 23, 24])
    requests = [
        request_a,
        request_b,
    ]

    batch = BatchBuilder().build_prefill_batch(
        requests,
        pad_token_id=0,
    )

    assert torch.equal(
        batch.input_ids,
        torch.tensor(
            [
                [0, 0, 11, 12],
                [21, 22, 23, 24],
            ]
        ),
    )

    assert torch.equal(
        batch.attention_mask,
        torch.tensor(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ]
        ),
    )

    assert torch.equal(
        batch.position_ids,
        torch.tensor(
            [
                [0, 0, 0, 1],
                [0, 1, 2, 3],
            ]
        ),
    )

@pytest.mark.integration
def test_variable_length_batched_prefill_kv_length(
    real_runner,
) -> None:
    input_ids_a = (
        real_runner.encode_prompt("Hello")
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    input_ids_b = (
        real_runner.encode_prompt(
            "The capital of France is"
        )
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    assert len(input_ids_a) != len(input_ids_b)

    request_a = make_request("A", input_ids_a)
    request_b = make_request("B", input_ids_b)

    requests = [request_a, request_b]

    batch = BatchBuilder().build_prefill_batch(
        requests,
        pad_token_id=real_runner.pad_token_id,
    )

    output = real_runner.prefill_batch(batch)

    request_caches = split_batched_legacy_kv_cache(
        output.past_key_values
    )

    for request, next_token_id, request_cache in zip(
        requests,
        output.next_token_ids,
        request_caches,
        strict=True,
    ):
        request.attach_kv_cache(request_cache)
        request.append_generated_token(next_token_id)

    print(
        "A logical prompt length:",
        len(request_a.input_ids),
    )
    print(
        "A physical KV length:",
        get_kv_sequence_length(
            request_a.past_key_values
        ),
    )

    print(
        "B logical prompt length:",
        len(request_b.input_ids),
    )
    print(
        "B physical KV length:",
        get_kv_sequence_length(
            request_b.past_key_values
        ),
    )

@pytest.mark.integration
def test_stack_and_split_cache_round_trip() -> None:
    cache_a = (
        (
            torch.tensor(
                [[[[10.0], [11.0]]]]
            ),
            torch.tensor(
                [[[[100.0], [101.0]]]]
            ),
        ),
    )

    cache_b = (
        (
            torch.tensor(
                [[[[20.0], [21.0]]]]
            ),
            torch.tensor(
                [[[[200.0], [201.0]]]]
            ),
        ),
    )

    batched_cache = stack_legacy_kv_caches(
        [cache_a, cache_b]
    )

    batched_key, batched_value = (
        batched_cache[0]
    )

    assert batched_key.shape[0] == 2
    assert batched_value.shape[0] == 2

    split_caches = (
        split_batched_legacy_kv_cache(
            batched_cache
        )
    )

    assert len(split_caches) == 2

    assert torch.equal(
        split_caches[0][0][0],
        cache_a[0][0],
    )
    assert torch.equal(
        split_caches[0][0][1],
        cache_a[0][1],
    )

    assert torch.equal(
        split_caches[1][0][0],
        cache_b[0][0],
    )
    assert torch.equal(
        split_caches[1][0][1],
        cache_b[0][1],
    )

# @pytest.mark.integration
# def test_stack_rejects_different_kv_lengths() -> None:
#     cache_a = ...
#     cache_b = ...

#     with pytest.raises(
#         ValueError,
#         match="non-batch dimensions",
#     ):
#         stack_legacy_kv_caches(
#             [cache_a, cache_b]
#         )


@pytest.mark.integration
def test_equal_length_batched_decode_matches_single_prefill(
    real_runner,
) -> None:
    prompt_a = "The capital of France is"
    prompt_b = "The capital of Germany is"

    input_ids_a_tensor = real_runner.encode_prompt(prompt_a)# 2d tensor
    input_ids_b_tensor = real_runner.encode_prompt(prompt_b)

    #1d list, fit Class Request
    input_ids_a = (
        input_ids_a_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    input_ids_b = (
        input_ids_b_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    assert len(input_ids_a) == len(input_ids_b)

    request_a = make_request("A", input_ids_a)
    request_b = make_request("B", input_ids_b)

    list_of_requests = [request_a, request_b]
    batch_requests = BatchBuilder().build_equal_length_prefill_batch(
        list_of_requests
    )

    batched_prefill_output = real_runner.prefill_batch(batch_requests)

    # decode
    # decode input id are the new token id from prefill.
    first_token_a = batched_prefill_output.next_token_ids[0]
    first_token_b = batched_prefill_output.next_token_ids[1]
    # past_key_values should be splited from batched_prefill_output.past_key_values
    request_past_key_values = split_batched_legacy_kv_cache(
        batched_prefill_output.past_key_values
    )
    # get mask
    kv_length = get_kv_sequence_length(
        request_past_key_values[0]
    )


    batched_attention_mask = torch.ones(
        (2, kv_length + 1),
        dtype=torch.long,
    )

    single_decode_input_a = torch.tensor([[first_token_a]], dtype=torch.long)
    single_decode_input_b = torch.tensor([[first_token_b]], dtype=torch.long)
    single_attention_mask_a = torch.ones(
        (1, kv_length + 1),
        dtype=torch.long,
    )

    single_attention_mask_b = torch.ones(
        (1, kv_length + 1),
        dtype=torch.long,
    )

    single_decode_output_a = real_runner.decode(
        input_ids=single_decode_input_a,
        past_key_values=request_past_key_values[0],
        attention_mask=single_attention_mask_a,
    )
    single_decode_output_b = real_runner.decode(
        input_ids=single_decode_input_b,
        past_key_values=request_past_key_values[1],
        attention_mask=single_attention_mask_b,
    )

    decode_batch = DecodeBatch(
        request_ids=["A", "B"],
        input_ids=torch.tensor(
            [
                [first_token_a],
                [first_token_b],
            ],
            dtype=torch.long,
        ),
        past_key_values=batched_prefill_output.past_key_values,
        attention_mask=batched_attention_mask,
    )
    batched_decode_output = real_runner.decode_batch(
        decode_batch,
    )

    assert batched_decode_output.next_token_ids == [
        single_decode_output_a.next_token_id,
        single_decode_output_b.next_token_id,
    ]

    assert batched_decode_output.logits.shape[0] == 2
    assert batched_decode_output.logits.shape[1] == 1
    assert torch.isfinite(
        batched_decode_output.logits
    ).all()