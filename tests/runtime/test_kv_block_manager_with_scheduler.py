
from __future__ import annotations
import pytest

from app.runtime.kv_block_manager import (
    KVBlockManager,
    BlockAllocationError
)
from app.runtime.continuous_scheduler import (
    ContinuousScheduler
)
from app.runtime.batch_builder import BatchBuilder
from app.runtime.request import Request

import torch

from app.runtime.request import (
    Request,
    RequestState,
)

from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
)

from typing import Any, TypeAlias


from app.runtime.types import (
    BatchedDecodeOutput,
    BatchedPrefillOutput,
    DecodeOutput,
    PrefillOutput,
)


LegacyKVCache: TypeAlias = tuple[
    tuple[torch.Tensor, torch.Tensor],
    ...,
]


class FakeRunner:
    """
    CPU-only runner for ContinuousScheduler unit tests.

    Supports both:
    - single-request prefill/decode;
    - batched prefill/decode.
    """

    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        vocab_size: int = 32,
        pad_token_id: int = 0,
        eos_token_id: int = 31,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        if num_kv_heads <= 0:
            raise ValueError(
                "num_kv_heads must be positive"
            )

        if head_dim <= 0:
            raise ValueError(
                "head_dim must be positive"
            )

        if vocab_size <= 2:
            raise ValueError(
                "vocab_size must be greater than 2"
            )

        if not 0 <= pad_token_id < vocab_size:
            raise ValueError(
                "pad_token_id must be inside vocabulary"
            )

        if not 0 <= eos_token_id < vocab_size:
            raise ValueError(
                "eos_token_id must be inside vocabulary"
            )

        if pad_token_id == eos_token_id:
            raise ValueError(
                "pad_token_id and eos_token_id "
                "must be different"
            )

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.device = torch.device("cpu")

        self.prefill_call_count = 0
        self.decode_call_count = 0

        self.prefill_batch_call_count = 0
        self.decode_batch_call_count = 0

        self.last_prefill_input_ids: (
            torch.Tensor | None
        ) = None

        self.last_decode_input_ids: (
            torch.Tensor | None
        ) = None

        self.last_prefill_batch: Any | None = None
        self.last_decode_batch: Any | None = None

        self._next_token_id = 1

    def _make_next_token_ids(
        self,
        batch_size: int,
    ) -> list[int]:
        token_ids: list[int] = []

        while len(token_ids) < batch_size:
            token_id = (
                self._next_token_id
                % self.vocab_size
            )

            self._next_token_id += 1

            if token_id in {
                self.pad_token_id,
                self.eos_token_id,
            }:
                continue

            token_ids.append(token_id)

        return token_ids

    def _make_logits(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        next_token_ids: list[int],
    ) -> torch.Tensor:
        logits = torch.zeros(
            (
                batch_size,
                sequence_length,
                self.vocab_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for batch_index, token_id in enumerate(
            next_token_ids
        ):
            logits[
                batch_index,
                -1,
                token_id,
            ] = 1.0

        return logits

    def _make_kv_cache(
        self,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> LegacyKVCache:
        layers: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

        for layer_index in range(
            self.num_layers
        ):
            shape = (
                batch_size,
                self.num_kv_heads,
                sequence_length,
                self.head_dim,
            )

            key = torch.full(
                shape,
                fill_value=float(
                    layer_index + 1
                ),
                dtype=torch.float32,
                device=self.device,
            )

            value = torch.full(
                shape,
                fill_value=float(
                    -(layer_index + 1)
                ),
                dtype=torch.float32,
                device=self.device,
            )

            layers.append(
                (
                    key,
                    value,
                )
            )

        return tuple(layers)

    def _extend_kv_cache(
        self,
        past_key_values: LegacyKVCache,
    ) -> LegacyKVCache:
        updated_layers: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

        for key, value in past_key_values:
            if key.ndim != 4:
                raise ValueError(
                    "KV key must have shape "
                    "[batch, heads, sequence, head_dim]"
                )

            if value.shape != key.shape:
                raise ValueError(
                    "KV key and value shapes must match"
                )

            extension_shape = (
                key.shape[0],
                key.shape[1],
                1,
                key.shape[3],
            )

            updated_key = torch.cat(
                [
                    key,
                    torch.zeros(
                        extension_shape,
                        dtype=key.dtype,
                        device=key.device,
                    ),
                ],
                dim=2,
            )

            updated_value = torch.cat(
                [
                    value,
                    torch.zeros(
                        extension_shape,
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ],
                dim=2,
            )

            updated_layers.append(
                (
                    updated_key,
                    updated_value,
                )
            )

        return tuple(updated_layers)

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: (
            torch.Tensor | None
        ) = None,
    ) -> PrefillOutput:
        self.prefill_call_count += 1
        self.last_prefill_input_ids = (
            input_ids.clone()
        )

        if input_ids.ndim != 2:
            raise ValueError(
                "Prefill input_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if input_ids.shape[0] != 1:
            raise ValueError(
                "Single-request prefill requires "
                "batch size 1"
            )

        sequence_length = input_ids.shape[1]

        next_token_id = (
            self._make_next_token_ids(1)[0]
        )

        return PrefillOutput(
            next_token_id=next_token_id,
            past_key_values=self._make_kv_cache(
                batch_size=1,
                sequence_length=sequence_length,
            ),
            logits=self._make_logits(
                batch_size=1,
                sequence_length=sequence_length,
                next_token_ids=[
                    next_token_id
                ],
            ),
        )

    def decode(
        self,
        input_ids: torch.Tensor,
        past_key_values: LegacyKVCache,
        attention_mask: (
            torch.Tensor | None
        ) = None,
    ) -> DecodeOutput:
        self.decode_call_count += 1
        self.last_decode_input_ids = (
            input_ids.clone()
        )

        if input_ids.ndim != 2:
            raise ValueError(
                "Decode input_ids must have shape "
                "[batch_size, 1]"
            )

        if input_ids.shape != (1, 1):
            raise ValueError(
                "Single-request decode requires "
                "shape [1, 1]"
            )

        next_token_id = (
            self._make_next_token_ids(1)[0]
        )

        return DecodeOutput(
            next_token_id=next_token_id,
            past_key_values=(
                self._extend_kv_cache(
                    past_key_values
                )
            ),
            logits=self._make_logits(
                batch_size=1,
                sequence_length=1,
                next_token_ids=[
                    next_token_id
                ],
            ),
        )

    def prefill_batch(
        self,
        batch,
    ) -> BatchedPrefillOutput:
        self.prefill_batch_call_count += 1
        self.prefill_call_count += 1
        self.last_prefill_batch = batch

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask

        if input_ids.ndim != 2:
            raise ValueError(
                "Batched prefill input_ids must "
                "have shape [batch, sequence]"
            )

        if (
            attention_mask is not None
            and attention_mask.shape
            != input_ids.shape
        ):
            raise ValueError(
                "attention_mask must match "
                "input_ids shape"
            )

        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        next_token_ids = (
            self._make_next_token_ids(
                batch_size
            )
        )

        return BatchedPrefillOutput(
            next_token_ids=next_token_ids,
            past_key_values=self._make_kv_cache(
                batch_size=batch_size,
                sequence_length=sequence_length,
            ),
            logits=self._make_logits(
                batch_size=batch_size,
                sequence_length=sequence_length,
                next_token_ids=next_token_ids,
            ),
        )

    def decode_batch(
        self,
        batch,
    ) -> BatchedDecodeOutput:
        self.decode_batch_call_count += 1
        self.decode_call_count += 1
        self.last_decode_batch = batch

        input_ids = batch.input_ids
        past_key_values = (
            batch.past_key_values
        )

        if input_ids.ndim != 2:
            raise ValueError(
                "Batched decode input_ids must "
                "have shape [batch, 1]"
            )

        if input_ids.shape[1] != 1:
            raise ValueError(
                "Batched decode must contain "
                "one token per request"
            )

        if not past_key_values:
            raise ValueError(
                "Batched decode requires "
                "past_key_values"
            )

        batch_size = input_ids.shape[0]

        first_key = past_key_values[0][0]

        if first_key.shape[0] != batch_size:
            raise ValueError(
                "KV batch size must match "
                "input batch size"
            )

        next_token_ids = (
            self._make_next_token_ids(
                batch_size
            )
        )

        return BatchedDecodeOutput(
            next_token_ids=next_token_ids,
            past_key_values=(
                self._extend_kv_cache(
                    past_key_values
                )
            ),
            logits=self._make_logits(
                batch_size=batch_size,
                sequence_length=1,
                next_token_ids=next_token_ids,
            ),
        )

def make_request( 
        request_id:str = "request-a",
        input_ids:list[int] | None = None,
        ) -> Request:
    return Request(
        request_id=request_id,
        input_ids=list(input_ids),
        max_new_tokens=8,
    )

def make_scheduler(
    *,
    runner=None,
    batch_builder=None,
    block_manager=None,
    max_prefill_batch_size: int = 4,
    max_decode_batch_size: int = 4,
) -> ContinuousScheduler:
    if runner is None:
        runner = FakeRunner()

    if batch_builder is None:
        batch_builder = BatchBuilder()

    if block_manager is None:
        block_manager = KVBlockManager(
            num_blocks=32,
            block_size=4,
        )

    return ContinuousScheduler(
        runner=runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        max_prefill_batch_size=(
            max_prefill_batch_size
        ),
        max_decode_batch_size=(
            max_decode_batch_size
        ),
    )



def test_prefill_reserves_prompt_blocks() -> None:

    runner = FakeRunner()
    assert hasattr(
        runner,
        "prefill_batch",
    )

    assert callable(
        runner.prefill_batch
    )

    print(
        "FakeRunner module:",
        FakeRunner.__module__,
    )

    print(
        "FakeRunner methods:",
        [
            name
            for name in dir(runner)
            if "prefill" in name
        ],
    )

    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )

    scheduler.add_request(request)
    scheduler.step()

    assert runner.prefill_call_count == 1

    assert request.block_table == [0, 1]
    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 6

def test_prefill_reserves_blocks_for_entire_batch() -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )
    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
        max_prefill_batch_size=2,
    )

    request_a = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )
    request_b = make_request(
        request_id="request-b",
        input_ids=[6, 7, 8],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    assert runner.prefill_call_count == 1

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == [2]

    assert block_manager.num_allocated_blocks == 3
    assert block_manager.num_free_blocks == 5

# for senario where a batch cannot fit in the available KV blocks, the scheduler should not run prefill for all requests in that batchand should raise an error
# with old version ContinuousScheduler::_select_prefill_requests
# def test_prefill_does_not_run_when_batch_cannot_fit() -> None:
#     block_manager = KVBlockManager(
#         num_blocks=2,
#         block_size=4,
#     )
#     runner = FakeRunner()

#     scheduler = make_scheduler(
#         runner=runner,
#         block_manager=block_manager,
#         max_prefill_batch_size=2,
#     )

#     request_a = make_request(
#         request_id="request-a",
#         input_ids=[1, 2, 3, 4, 5],
#     )
#     request_b = make_request(
#         request_id="request-b",
#         input_ids=[6, 7, 8, 9, 10],
#     )

#     scheduler.add_request(request_a)
#     scheduler.add_request(request_b)

#     with pytest.raises(
#         BlockAllocationError,
#         match="Insufficient KV blocks for batch",
#     ):
#         scheduler.step()

#     assert runner.prefill_call_count == 0
#     assert runner.decode_call_count == 0

#     assert request_a.block_table == []
#     assert request_b.block_table == []

#     assert block_manager.num_free_blocks == 2
#     assert block_manager.num_allocated_blocks == 0

#     assert request_a.state == RequestState.WAITING
#     assert request_b.state == RequestState.WAITING

# capacity aware prefill
def test_prefill_selects_only_requests_that_fit() -> None:
    block_manager = KVBlockManager(
        num_blocks=3,
        block_size=4,
    )
    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
        max_prefill_batch_size=2,
    )

    request_a = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )

    request_b = make_request(
        request_id="request-b",
        input_ids=[6, 7, 8, 9, 10],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    assert runner.prefill_batch_call_count == 1

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == []

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.WAITING

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 1

def test_prefill_does_nothing_when_no_request_fits() -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )
    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )

    scheduler.add_request(request)

    scheduler.step()

    assert runner.prefill_batch_call_count == 0

    assert request.state == RequestState.WAITING
    assert request.block_table == []

    assert block_manager.num_allocated_blocks == 0
    assert block_manager.num_free_blocks == 1

def test_prefill_can_skip_large_request_and_run_smaller_one() -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )
    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
        max_prefill_batch_size=2,
    )

    large_request = make_request(
        request_id="large",
        input_ids=[1, 2, 3, 4, 5],
    )

    small_request = make_request(
        request_id="small",
        input_ids=[6, 7, 8],
    )

    scheduler.add_request(large_request)
    scheduler.add_request(small_request)

    scheduler.step()

    assert large_request.state == RequestState.WAITING

    assert small_request.state == RequestState.DECODING
    assert small_request.block_table == [0]


# test decode
def test_decode_allocates_new_block_when_crossing_boundary() -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # Tick 1: prefill.
    scheduler.step()

    assert request.block_table == [0]

    # At this point:
    # prompt = 4
    # first generated token already exists after prefill.
    #
    # Total KV requirement before next decode
    # depends on your exact generated-token semantics.

    scheduler.step()

    assert len(request.block_table) == 2

def test_decode_allocates_new_block_when_crossing_boundary(
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # Tick 1: prefill.
    scheduler.step()

    assert request.state == RequestState.DECODING

    assert request.block_table == [0]
    assert block_manager.num_allocated_blocks == 1

    assert runner.prefill_batch_call_count == 1
    assert runner.decode_batch_call_count == 0

    # KV currently contains only the 4 prompt tokens.
    assert (
        get_kv_sequence_length(
            request.past_key_values
        )
        == 4
    )

    # Tick 2:
    #
    # decode the first generated token.
    #
    # KV: 4 -> 5
    # blocks: 1 -> 2
    scheduler.step()

    assert request.block_table == [0, 1]

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 2

    assert runner.decode_batch_call_count == 1

    assert (
        get_kv_sequence_length(
            request.past_key_values
        )
        == 5
    )

def test_decode_does_not_allocate_block_when_capacity_exists(
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3],
    )

    scheduler.add_request(request)

    # Tick 1: prefill.
    scheduler.step()

    assert request.block_table == [0]
    assert block_manager.num_allocated_blocks == 1

    assert (
        get_kv_sequence_length(
            request.past_key_values
        )
        == 3
    )

    # Tick 2:
    #
    # KV: 3 -> 4
    #
    # Still fits inside block 0.
    scheduler.step()

    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 3

    assert runner.decode_batch_call_count == 1

    assert (
        get_kv_sequence_length(
            request.past_key_values
        )
        == 4
    )

def test_decode_does_not_run_when_new_block_unavailable(
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )

    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # Tick 1: prefill consumes the only block.
    scheduler.step()

    assert request.state == RequestState.DECODING

    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 0

    assert runner.decode_batch_call_count == 0

    # Tick 2:
    scheduler.step()

    assert runner.decode_batch_call_count == 0
    assert request.state == RequestState.DECODING
    assert request.block_table == [0]



def test_decode_runs_requests_that_fit_and_skips_blocked_ones(
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    runner = FakeRunner()

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
        max_decode_batch_size=2,
    )

    request_a = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3],
    )

    request_b = make_request(
        request_id="request-b",
        input_ids=[4, 5, 6, 7],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # Tick 1: prefill.
    scheduler.step()

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.DECODING

    # A owns one block and has KV len 3.
    # B owns one block and has KV len 4.
    #
    # All blocks are now allocated.
    assert block_manager.num_free_blocks == 0

    old_a_kv_length = get_kv_sequence_length(
        request_a.past_key_values
    )

    old_b_kv_length = get_kv_sequence_length(
        request_b.past_key_values
    )

    # Tick 2:
    #
    # A: 3 -> 4, no new block required
    # B: 4 -> 5, needs a new block
    #
    # Only A should decode.
    scheduler.step()

    assert (
        get_kv_sequence_length(
            request_a.past_key_values
        )
        == old_a_kv_length + 1
    )

    assert (
        get_kv_sequence_length(
            request_b.past_key_values
        )
        == old_b_kv_length
    )

    assert runner.decode_batch_call_count == 1
