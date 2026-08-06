import pytest
import torch

from app.runtime.continuous_scheduler import (
    ContinuousScheduler,
)
from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
)
from app.runtime.request import (
    Request,
    RequestState,
)
from app.runtime.types import (
    BatchedDecodeOutput,
    BatchedPrefillOutput,
)


class FakeRunner:
    def __init__(self) -> None:
        self.prefill_calls: list[list[str]] = []
        self.decode_calls: list[list[str]] = []

        self.eos_token_ids: set[int] = {9999}
        self.eos_on_decode_for: set[str] = set()

        self.pad_token_id: int = 0
        self.device = torch.device("cpu")

        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 1
        self.vocab_size = 10000

    def _make_batched_kv_cache(
        self,
        *,
        batch_size: int,
        sequence_length: int,
    ):
        layers = []

        for _ in range(self.num_layers):
            shape = (
                batch_size,
                self.num_kv_heads,
                sequence_length,
                self.head_dim,
            )

            key = torch.zeros(
                shape,
                dtype=torch.float32,
            )
            value = torch.zeros(
                shape,
                dtype=torch.float32,
            )

            layers.append(
                (
                    key,
                    value,
                )
            )

        return tuple(layers)

    def _make_logits(
        self,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return torch.zeros(
            (
                batch_size,
                sequence_length,
                self.vocab_size,
            ),
            dtype=torch.float32,
        )

    def prefill_batch(
        self,
        requests: list[Request],
    ) -> BatchedPrefillOutput:
        self.prefill_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        batch_size = len(requests)

        # Mimic padded batched prefill:
        # physical KV length equals the longest prompt
        # in the batch, while each Request.kv_tokens
        # still tracks its own logical prompt length.
        physical_kv_length = max(
            len(request.input_ids)
            for request in requests
        )

        next_token_ids = [
            1000 + index
            for index in range(batch_size)
        ]

        return BatchedPrefillOutput(
            next_token_ids=next_token_ids,
            past_key_values=self._make_batched_kv_cache(
                batch_size=batch_size,
                sequence_length=physical_kv_length,
            ),
            logits=self._make_logits(
                batch_size=batch_size,
                sequence_length=physical_kv_length,
            ),
        )

    def decode_batch(
        self,
        requests: list[Request],
    ) -> BatchedDecodeOutput:
        self.decode_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        batch_size = len(requests)

        old_physical_kv_length = (
            get_kv_sequence_length(
                requests[0].past_key_values
            )
        )

        next_token_ids: list[int] = []

        for request in requests:
            if (
                request.request_id
                in self.eos_on_decode_for
            ):
                next_token_id = 9999
            else:
                next_token_id = (
                    2000
                    + request.generated_tokens_count
                )

            next_token_ids.append(next_token_id)

        return BatchedDecodeOutput(
            next_token_ids=next_token_ids,
            past_key_values=self._make_batched_kv_cache(
                batch_size=batch_size,
                sequence_length=(
                    old_physical_kv_length + 1
                ),
            ),
            logits=self._make_logits(
                batch_size=batch_size,
                sequence_length=1,
            ),
        )


class FakeBatchBuilder:
    def __init__(self) -> None:
        self.prefill_calls: list[list[str]] = []
        self.decode_calls: list[list[str]] = []

    def build_prefill_batch(
        self,
        requests: list[Request],
        pad_token_id: int | None,
        device: torch.device | None,
    ) -> list[Request]:
        self.prefill_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        return requests

    def build_equal_length_decode_batch(
        self,
        requests: list[Request],
        device: torch.device | None,
    ) -> list[Request]:
        self.decode_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        return requests


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def fake_batch_builder() -> FakeBatchBuilder:
    return FakeBatchBuilder()


@pytest.fixture
def scheduler(
    fake_runner,
    fake_batch_builder,
    block_manager,
) -> ContinuousScheduler:
    return ContinuousScheduler(
        runner=fake_runner,
        batch_builder=fake_batch_builder,
        block_manager=block_manager,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )


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


def test_step_prefills_waiting_request(
    scheduler,
) -> None:
    request = make_request("request-a")
    scheduler.add_request(request)

    result = scheduler.step()

    assert result.prefetched_request_ids == [
        "request-a"
    ]
    assert result.decoded_request_ids == []
    assert result.finished_request_ids == []

    assert result.generated_token_ids == {
        "request-a": 1000
    }

    assert len(scheduler.waiting) == 0
    assert "request-a" in scheduler.active

    assert request.state == RequestState.DECODING
    assert request.generated_ids == [1000]
    assert request.generated_tokens_count == 1

    assert request.past_key_values is not None
    assert request.kv_tokens == 3
    assert len(request.block_table) == 1


def test_prefill_respects_batch_size_limit(
    scheduler,
) -> None:
    request_a = make_request("request-a")
    request_b = make_request("request-b")
    request_c = make_request("request-c")

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    first_result = scheduler.step()

    assert first_result.prefetched_request_ids == [
        "request-a",
        "request-b",
    ]

    assert list(scheduler.waiting) == [
        request_c
    ]

    assert set(scheduler.active) == {
        "request-a",
        "request-b",
    }

    second_result = scheduler.step()

    assert second_result.prefetched_request_ids == [
        "request-c"
    ]

    assert len(scheduler.waiting) == 0
    assert set(scheduler.active) == {
        "request-a",
        "request-b",
        "request-c",
    }


def test_prefill_builds_one_batch(
    scheduler,
    fake_runner,
    fake_batch_builder,
) -> None:
    scheduler.add_request(
        make_request("request-a")
    )
    scheduler.add_request(
        make_request("request-b")
    )

    scheduler.step()

    assert fake_batch_builder.prefill_calls == [
        ["request-a", "request-b"]
    ]

    assert fake_runner.prefill_calls == [
        ["request-a", "request-b"]
    ]


def test_request_can_finish_during_prefill(
    scheduler,
) -> None:
    request = make_request(
        "request-a",
        max_new_tokens=1,
    )

    scheduler.add_request(request)

    result = scheduler.step()

    assert result.prefetched_request_ids == [
        "request-a"
    ]
    assert result.finished_request_ids == [
        "request-a"
    ]

    assert request.generated_ids == [1000]
    assert request.generated_tokens_count == 1
    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"

    assert "request-a" not in scheduler.active
    assert "request-a" in scheduler.completed

    assert request.past_key_values is None
    assert request.kv_tokens == 0
    assert request.block_table == []


def test_active_request_is_decoded_on_next_step(
    scheduler,
    fake_runner,
    fake_batch_builder,
) -> None:
    request = make_request(
        "request-a",
        max_new_tokens=3,
    )
    scheduler.add_request(request)

    first_result = scheduler.step()

    assert first_result.prefetched_request_ids == [
        "request-a"
    ]
    assert first_result.decoded_request_ids == []
    assert request.generated_ids == [1000]
    assert request.kv_tokens == 3

    second_result = scheduler.step()

    assert second_result.prefetched_request_ids == []
    assert second_result.decoded_request_ids == [
        "request-a"
    ]
    assert second_result.generated_token_ids == {
        "request-a": 2001
    }

    assert request.generated_ids == [
        1000,
        2001,
    ]
    assert request.generated_tokens_count == 2
    assert request.kv_tokens == 4
    assert request.state == RequestState.DECODING

    assert fake_batch_builder.decode_calls == [
        ["request-a"]
    ]
    assert fake_runner.decode_calls == [
        ["request-a"]
    ]


def test_request_finishes_during_decode(
    scheduler,
) -> None:
    request = make_request(
        "request-a",
        max_new_tokens=2,
    )
    scheduler.add_request(request)

    first_result = scheduler.step()

    assert first_result.prefetched_request_ids == [
        "request-a"
    ]
    assert request.generated_ids == [1000]
    assert request.state == RequestState.DECODING

    second_result = scheduler.step()

    assert second_result.decoded_request_ids == [
        "request-a"
    ]
    assert second_result.finished_request_ids == [
        "request-a"
    ]

    assert request.generated_ids == [
        1000,
        2001,
    ]
    assert request.generated_tokens_count == 2
    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"

    assert "request-a" not in scheduler.active
    assert "request-a" in scheduler.completed

    assert request.past_key_values is None
    assert request.kv_tokens == 0
    assert request.block_table == []
    assert not scheduler.has_pending_work()


def test_new_request_joins_while_existing_request_decodes(
    scheduler,
    fake_runner,
) -> None:
    request_a = make_request(
        "request-a",
        max_new_tokens=4,
    )
    scheduler.add_request(request_a)

    tick_1 = scheduler.step()

    assert tick_1.prefetched_request_ids == [
        "request-a"
    ]
    assert tick_1.decoded_request_ids == []

    request_b = make_request(
        "request-b",
        max_new_tokens=4,
    )
    scheduler.add_request(request_b)

    tick_2 = scheduler.step()

    assert tick_2.prefetched_request_ids == [
        "request-b"
    ]
    assert tick_2.decoded_request_ids == [
        "request-a"
    ]

    assert request_a.generated_ids == [
        1000,
        2001,
    ]
    assert request_b.generated_ids == [1000]

    tick_3 = scheduler.step()

    assert tick_3.prefetched_request_ids == []
    assert tick_3.decoded_request_ids == [
        "request-a"
    ]

    assert fake_runner.decode_calls == [
        ["request-a"],
        ["request-a"],
    ]


def test_short_request_finishes_while_long_request_continues(
    scheduler,
) -> None:
    request_a = make_request(
        "request-a",
        max_new_tokens=2,
    )
    request_b = make_request(
        "request-b",
        max_new_tokens=4,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    assert set(scheduler.active) == {
        "request-a",
        "request-b",
    }

    second_result = scheduler.step()

    assert second_result.finished_request_ids == [
        "request-a"
    ]

    assert "request-a" in scheduler.completed
    assert "request-a" not in scheduler.active
    assert "request-b" in scheduler.active

    assert request_a.block_table == []
    assert request_a.kv_tokens == 0
    assert request_b.block_table != []

    third_result = scheduler.step()

    assert third_result.decoded_request_ids == [
        "request-b"
    ]


def test_eos_finishes_and_cleans_up_request(
    scheduler,
    fake_runner,
) -> None:
    request = make_request(
        "request-a",
        max_new_tokens=10,
    )

    scheduler.add_request(request)

    first_result = scheduler.step()

    assert first_result.prefetched_request_ids == [
        "request-a"
    ]
    assert request.generated_ids == [1000]
    assert request.state == RequestState.DECODING
    assert request.past_key_values is not None

    fake_runner.eos_on_decode_for.add(
        "request-a"
    )

    second_result = scheduler.step()

    assert second_result.decoded_request_ids == [
        "request-a"
    ]
    assert second_result.finished_request_ids == [
        "request-a"
    ]
    assert second_result.generated_token_ids == {
        "request-a": 9999
    }

    assert request.generated_ids == [
        1000,
        9999,
    ]
    assert request.generated_tokens_count == 2

    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "eos"

    assert "request-a" not in scheduler.active
    assert "request-a" in scheduler.completed

    assert request.past_key_values is None
    assert request.kv_tokens == 0
    assert request.block_table == []
    assert not scheduler.has_pending_work()


def test_eos_takes_precedence_over_length(
    scheduler,
    fake_runner,
) -> None:
    request = make_request(
        "request-a",
        max_new_tokens=2,
    )

    scheduler.add_request(request)
    scheduler.step()

    assert request.generated_tokens_count == 1

    fake_runner.eos_on_decode_for.add(
        "request-a"
    )

    scheduler.step()

    assert request.generated_tokens_count == 2
    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "eos"


def test_one_request_hits_eos_while_other_continues(
    scheduler,
    fake_runner,
) -> None:
    request_a = make_request(
        "request-a",
        max_new_tokens=10,
    )
    request_b = make_request(
        "request-b",
        max_new_tokens=10,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    assert set(scheduler.active) == {
        "request-a",
        "request-b",
    }

    fake_runner.eos_on_decode_for.add(
        "request-a"
    )

    second_result = scheduler.step()

    assert second_result.decoded_request_ids == [
        "request-a",
        "request-b",
    ]
    assert second_result.finished_request_ids == [
        "request-a"
    ]

    assert request_a.state == RequestState.FINISHED
    assert request_a.finish_reason == "eos"
    assert request_a.past_key_values is None
    assert request_a.kv_tokens == 0
    assert request_a.block_table == []

    assert request_b.state == RequestState.DECODING
    assert request_b.past_key_values is not None
    assert request_b.kv_tokens > 0
    assert request_b.block_table != []

    assert set(scheduler.active) == {
        "request-b"
    }
    assert "request-a" in scheduler.completed

    third_result = scheduler.step()

    assert third_result.decoded_request_ids == [
        "request-b"
    ]

    assert fake_runner.decode_calls == [
        ["request-a", "request-b"],
        ["request-b"],
    ]