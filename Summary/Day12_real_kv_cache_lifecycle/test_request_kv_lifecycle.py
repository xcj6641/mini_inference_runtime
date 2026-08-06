from __future__ import annotations

import pytest

from app.runtime.request import Request
from app.runtime.types import RequestState


def make_request() -> Request:
    return Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=8,
    )


def test_request_initial_state() -> None:
    request = make_request()

    assert request.request_id == "request-a"
    assert request.input_ids == [1, 2, 3]
    assert request.max_new_tokens == 8

    assert request.prompt_tokens == 3
    assert request.generated_ids == []
    assert request.generated_tokens_count == 0
    assert request.kv_tokens == 0
    assert request.total_sequence_length == 3

    assert request.state == RequestState.WAITING
    assert request.finish_reason is None
    assert request.error_message is None

    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_request_attaches_kv_cache() -> None:
    request = make_request()
    fake_cache = object()

    request.attach_kv_cache(fake_cache)

    assert request.has_kv_cache is True
    assert request.past_key_values is fake_cache


def test_request_rejects_none_kv_cache() -> None:
    request = make_request()

    with pytest.raises(
        ValueError,
        match="past_key_values cannot be None",
    ):
        request.attach_kv_cache(None)


def test_append_generated_token_updates_state() -> None:
    request = make_request()

    request.append_generated_token(100)
    request.append_generated_token(101)

    assert request.generated_ids == [100, 101]
    assert request.generated_tokens_count == 2
    assert request.total_sequence_length == 5


def test_release_kv_cache_removes_reference() -> None:
    request = make_request()
    fake_cache = object()

    request.attach_kv_cache(fake_cache)

    assert request.has_kv_cache is True

    request.release_kv_cache()

    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_finished_request_releases_kv_reference() -> None:
    request = make_request()
    fake_cache = object()

    request.attach_kv_cache(fake_cache)
    request.append_generated_token(100)

    request.mark_finished("eos")

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "eos"
    assert request.error_message is None

    assert request.generated_ids == [100]
    assert request.generated_tokens_count == 1

    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_length_finished_request_releases_kv_reference() -> None:
    request = make_request()
    request.attach_kv_cache(object())

    request.mark_finished("length")

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"
    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_cancelled_request_releases_kv_reference() -> None:
    request = make_request()
    request.attach_kv_cache(object())

    request.cancel()

    assert request.state == RequestState.CANCELLED
    assert request.finish_reason == "cancelled"
    assert request.error_message is None

    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_failed_request_releases_kv_reference() -> None:
    request = make_request()
    request.attach_kv_cache(object())

    error = RuntimeError("decode failed")

    request.mark_failed(error)

    assert request.state == RequestState.FAILED
    assert request.finish_reason == "error"
    assert request.error_message == "decode failed"

    assert request.past_key_values is None
    assert request.has_kv_cache is False


def test_generated_tokens_are_preserved_after_finish() -> None:
    request = make_request()
    request.attach_kv_cache(object())

    request.append_generated_token(100)
    request.append_generated_token(101)

    request.mark_finished("length")

    assert request.generated_ids == [100, 101]
    assert request.generated_tokens_count == 2
    assert request.total_sequence_length == 5

    assert request.past_key_values is None

def test_request_rejects_empty_request_id() -> None:
    with pytest.raises(
        ValueError,
        match="request_id cannot be empty",
    ):
        Request(
            request_id="",
            input_ids=[1, 2, 3],
            max_new_tokens=8,
        )


def test_request_rejects_empty_input_ids() -> None:
    with pytest.raises(
        ValueError,
        match="input_ids cannot be empty",
    ):
        Request(
            request_id="request-a",
            input_ids=[],
            max_new_tokens=8,
        )


def test_request_rejects_non_positive_max_new_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        Request(
            request_id="request-a",
            input_ids=[1, 2, 3],
            max_new_tokens=0,
        )