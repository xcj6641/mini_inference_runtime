import pytest

from app.runtime.continuous_scheduler import (
    ContinuousScheduler,
)
from app.runtime.request import Request, RequestState


class DummyRunner:
    pass


class DummyBatchBuilder:
    pass


@pytest.fixture
def scheduler() -> ContinuousScheduler:
    return ContinuousScheduler(
        runner=DummyRunner(),
        batch_builder=DummyBatchBuilder(),
        max_prefill_batch_size=2,
    )


def make_request(
    request_id: str,
) -> Request:
    return Request(
        request_id=request_id,
        input_ids=[1, 2, 3],
        generated_ids=[],
        past_key_values=None,
        prompt_tokens=3,
        generated_tokens_count=0,
        state=RequestState.WAITING,
        finish_reason=None,
        error_message=None,
        max_new_tokens=4,
    )


def test_add_request_places_it_in_waiting(
    scheduler,
) -> None:
    request = make_request("request-a")

    scheduler.add_request(request)

    assert list(scheduler.waiting) == [request]
    assert scheduler.has_pending_work()


def test_duplicate_request_id_is_rejected(
    scheduler,
) -> None:
    scheduler.add_request(
        make_request("request-a")
    )

    with pytest.raises(
        ValueError,
        match="Duplicate request ID",
    ):
        scheduler.add_request(
            make_request("request-a")
        )


def test_get_waiting_request(
    scheduler,
) -> None:
    request = make_request("request-a")
    scheduler.add_request(request)

    result = scheduler.get_request("request-a")

    assert result is request


def test_empty_step_is_no_op(
    scheduler,
) -> None:
    result = scheduler.step()

    assert result.prefetched_request_ids == []
    assert result.decoded_request_ids == []
    assert result.finished_request_ids == []
    assert result.generated_token_ids == {}

    assert not scheduler.has_pending_work()
    assert scheduler.tick_id == 1

# test scheduler