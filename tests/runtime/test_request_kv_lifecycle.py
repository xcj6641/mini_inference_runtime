from __future__ import annotations

import pytest

from app.runtime.request import (
    Request,
    RequestState,
)


def make_request() -> Request:
    return Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=8,
    )


# ============================================================
# Initial state
# ============================================================


def test_request_initial_state() -> None:
    request = make_request()

    assert request.request_id == "request-a"
    assert request.input_ids == [1, 2, 3]
    assert request.max_new_tokens == 8

    assert request.prompt_tokens == 3

    assert request.generated_ids == []
    assert request.generated_tokens_count == 0

    assert request.total_sequence_length == 3

    assert request.state == RequestState.WAITING

    assert request.finish_reason is None
    assert request.error_message is None

    # Current paged-KV logical state.
    assert request.kv_tokens == 0
    assert request.block_table == []


# ============================================================
# Generated-token state
# ============================================================


def test_append_generated_token_updates_state() -> None:
    request = make_request()

    request.append_generated_token(100)
    request.append_generated_token(101)

    assert request.generated_ids == [
        100,
        101,
    ]

    assert request.generated_tokens_count == 2

    assert request.total_sequence_length == 5


def test_generated_tokens_preserve_order() -> None:
    request = make_request()

    request.append_generated_token(100)
    request.append_generated_token(101)
    request.append_generated_token(102)

    assert request.generated_ids == [
        100,
        101,
        102,
    ]

    assert request.generated_tokens_count == 3


# ============================================================
# Logical KV state
# ============================================================


def test_request_starts_without_kv_tokens() -> None:
    request = make_request()

    assert request.kv_tokens == 0
    assert request.block_table == []


def test_request_can_track_prompt_kv_tokens() -> None:
    request = make_request()

    request.set_kv_tokens_from_prompt()

    assert request.kv_tokens == 3


def test_setting_prompt_kv_tokens_does_not_change_prompt() -> None:
    request = make_request()

    original_input_ids = list(
        request.input_ids
    )

    request.set_kv_tokens_from_prompt()

    assert request.input_ids == original_input_ids
    assert request.kv_tokens == 3


def test_block_table_can_track_physical_blocks() -> None:
    request = make_request()

    request.block_table.extend(
        [2, 7]
    )

    assert request.block_table == [
        2,
        7,
    ]


# ============================================================
# Finish lifecycle
# ============================================================


def test_finished_request_updates_terminal_state() -> None:
    request = make_request()

    request.append_generated_token(100)

    request.mark_finished("eos")

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "eos"
    assert request.error_message is None

    assert request.generated_ids == [100]
    assert request.generated_tokens_count == 1


def test_length_finished_request_updates_terminal_state() -> None:
    request = make_request()

    request.mark_finished("length")

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"
    assert request.error_message is None


def test_generated_tokens_are_preserved_after_finish() -> None:
    request = make_request()

    request.append_generated_token(100)
    request.append_generated_token(101)

    request.mark_finished("length")

    assert request.state == RequestState.FINISHED

    assert request.generated_ids == [
        100,
        101,
    ]

    assert request.generated_tokens_count == 2

    assert request.total_sequence_length == 5


# ============================================================
# Cancel lifecycle
# ============================================================


def test_cancelled_request_updates_terminal_state() -> None:
    request = make_request()

    request.cancel()

    assert request.state == RequestState.CANCELLED
    assert request.finish_reason == "cancelled"
    assert request.error_message is None


# ============================================================
# Failure lifecycle
# ============================================================


def test_failed_request_updates_terminal_state() -> None:
    request = make_request()

    error = RuntimeError(
        "decode failed"
    )

    request.mark_failed(error)

    assert request.state == RequestState.FAILED
    assert request.finish_reason == "error"
    assert request.error_message == "decode failed"


def test_generated_tokens_are_preserved_after_failure() -> None:
    request = make_request()

    request.append_generated_token(100)

    request.mark_failed(
        RuntimeError("decode failed")
    )

    assert request.state == RequestState.FAILED

    assert request.generated_ids == [100]
    assert request.generated_tokens_count == 1


# ============================================================
# Validation
# ============================================================


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
        match=(
            "max_new_tokens must be greater than zero"
        ),
    ):
        Request(
            request_id="request-a",
            input_ids=[1, 2, 3],
            max_new_tokens=0,
        )