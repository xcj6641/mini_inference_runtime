
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

from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.prefix_cache import PrefixCache

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
        paged_kv_cache: PagedKVCache | None = None,
        prefix_cache: PrefixCache | None = None,
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

    if prefix_cache is None:
        prefix_cache = PrefixCache(
            block_manager=block_manager,
        )

    if paged_kv_cache is None:
        paged_kv_cache = PagedKVCache(
            num_layers=1,
            num_blocks=32,
            num_kv_heads=1,
            block_size=4,
            head_dim=2,
            dtype=torch.float32,
            device="cpu",
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
        paged_kv_cache=paged_kv_cache,
        prefix_cache=prefix_cache,
    )



# ============================================================
# Prefill + KV block manager
# ============================================================


def test_prefill_reserves_prompt_blocks(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )

    scheduler.add_request(request)
    scheduler.step()

    assert request.state == RequestState.DECODING

    # 5 KV tokens with block_size=4 require 2 blocks.
    assert request.kv_tokens == 5
    assert request.block_table == [0, 1]

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 6


def test_prefill_reserves_blocks_for_entire_batch(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        max_prefill_batch_size=2,
        prefix_cache=prefix_cache,
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

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.DECODING

    assert request_a.kv_tokens == 5
    assert request_b.kv_tokens == 3

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == [2]

    assert block_manager.num_allocated_blocks == 3
    assert block_manager.num_free_blocks == 5


def test_prefill_selects_only_requests_that_fit(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=3,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        max_prefill_batch_size=2,
        prefix_cache=prefix_cache,
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

    # Only A fits:
    #
    # A needs 2 blocks.
    # B would need another 2.
    # Only 3 exist.
    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.WAITING

    assert request_a.kv_tokens == 5
    assert request_b.kv_tokens == 0

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == []

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 1


def test_prefill_does_nothing_when_no_request_fits(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4, 5],
    )

    scheduler.add_request(request)

    scheduler.step()

    # Prompt requires 2 blocks but only 1 exists.
    assert request.state == RequestState.WAITING

    assert request.kv_tokens == 0
    assert request.block_table == []

    assert block_manager.num_allocated_blocks == 0
    assert block_manager.num_free_blocks == 1


def test_prefill_can_skip_large_request_and_run_smaller_one(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )
    runner = fake_runner_all_dim

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
# ============================================================
# Decode + KV block manager
# ============================================================


def test_decode_allocates_new_block_when_crossing_boundary(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # Tick 1: prefill.
    scheduler.step()

    assert request.state == RequestState.DECODING

    assert request.kv_tokens == 4
    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 3

    # Tick 2:
    #
    # Decode commits one more KV token:
    #
    # KV:     4 -> 5
    # blocks: 1 -> 2
    scheduler.step()

    assert request.kv_tokens == 5
    assert request.block_table == [0, 1]

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 2

def test_decode_does_not_allocate_block_when_capacity_exists(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3],
    )

    scheduler.add_request(request)

    # Tick 1: prefill.
    scheduler.step()

    assert request.state == RequestState.DECODING

    assert request.kv_tokens == 3
    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 3

    # Tick 2:
    #
    # KV: 3 -> 4
    #
    # Still fits in block 0.
    scheduler.step()

    assert request.kv_tokens == 4
    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 3


def test_decode_does_not_run_when_new_block_unavailable(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = make_request(
        request_id="request-a",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request)

    # Tick 1: prefill fills the only block.
    scheduler.step()

    assert request.state == RequestState.DECODING
    assert request.kv_tokens == 4
    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 0

    # Tick 2:
    #
    # KV 4 -> 5 requires another block.
    # None is available, so request must not advance.
    scheduler.step()

    assert request.state == RequestState.DECODING

    assert request.kv_tokens == 4
    assert request.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 0



def test_decode_runs_requests_that_fit_and_skips_blocked_ones(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        max_decode_batch_size=2,
        prefix_cache=prefix_cache,
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

    assert request_a.kv_tokens == 3
    assert request_b.kv_tokens == 4

    assert len(request_a.block_table) == 1
    assert len(request_b.block_table) == 1

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 0

    old_a_kv_tokens = request_a.kv_tokens
    old_b_kv_tokens = request_b.kv_tokens

    # Tick 2:
    #
    # A: 3 -> 4
    #    fits in existing block.
    #
    # B: 4 -> 5
    #    requires another block.
    #
    # No free blocks, so only A can decode.
    scheduler.step()

    assert (
        request_a.kv_tokens
        == old_a_kv_tokens + 1
    )

    assert (
        request_b.kv_tokens
        == old_b_kv_tokens
    )

    assert len(request_a.block_table) == 1
    assert len(request_b.block_table) == 1

    assert block_manager.num_allocated_blocks == 2
    assert block_manager.num_free_blocks == 0


def test_finished_request_frees_all_request_blocks(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    # Use a prompt shorter than one complete block.
    #
    # That avoids PrefixCache retaining a full-block prefix,
    # because this test is about request cleanup.
    request = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=1,
    )

    scheduler.add_request(request)

    scheduler.step()

    assert request.state == RequestState.FINISHED

    assert request.kv_tokens == 3
    assert request.block_table == []

    assert block_manager.num_allocated_blocks == 0
    assert block_manager.num_free_blocks == 4

    assert (
        request.request_id
        in scheduler.completed
    )


def test_finished_request_releases_kv_state(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        prefix_cache=prefix_cache,
    )

    request = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=1,
    )

    scheduler.add_request(request)

    scheduler.step()

    assert request.state == RequestState.FINISHED

    # Request no longer owns a contiguous past_key_values.
    #
    # Logical KV state should be reset instead.
    assert request.kv_tokens == 3
    assert request.block_table == []

def test_waiting_request_reuses_block_freed_by_finished_request(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )

    runner = fake_runner_all_dim

    scheduler = make_scheduler(
        runner=runner,
        block_manager=block_manager,
        max_prefill_batch_size=1,
    )

    request_a = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=1,
    )

    request_b = Request(
        request_id="request-b",
        input_ids=[4, 5, 6],
        max_new_tokens=1,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # Tick 1:
    # A is admitted.
    # B stays waiting because max_prefill_batch_size = 1.
    #
    # A finishes immediately and frees block 0.
    scheduler.step()

    assert request_a.state == RequestState.FINISHED
    assert request_a.block_table == []

    assert request_b.state == RequestState.WAITING

    assert block_manager.num_free_blocks == 1
    assert block_manager.num_allocated_blocks == 0

    # Tick 2:
    # B should now be able to use the freed block.
    scheduler.step()

    assert request_b.state == RequestState.FINISHED
    assert request_b.block_table == []

    assert (
        request_b.request_id
        in scheduler.completed
    )

    assert block_manager.num_free_blocks == 1

def test_waiting_request_reuses_specific_freed_block(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=1,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        max_prefill_batch_size=1,
        prefix_cache=prefix_cache,
    )

    request_a = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=1,
    )

    request_b = Request(
        request_id="request-b",
        input_ids=[4, 5, 6],
        max_new_tokens=8,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    # Tick 1:
    #
    # A gets block 0.
    # A immediately finishes.
    # Its block is released.
    #
    # B is still WAITING because max_prefill_batch_size=1.
    scheduler.step()

    assert request_a.state == RequestState.FINISHED
    assert request_a.block_table == []

    assert request_b.state == RequestState.WAITING

    assert block_manager.num_allocated_blocks == 0
    assert block_manager.num_free_blocks == 1

    # Tick 2:
    #
    # B should reuse physical block 0.
    scheduler.step()

    assert request_b.state == RequestState.DECODING

    assert request_b.kv_tokens == 3
    assert request_b.block_table == [0]

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 0

def test_finishing_one_request_does_not_free_other_request_blocks(
    fake_runner_all_dim: FakeRunner,
) -> None:
    block_manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager=block_manager,
    )

    scheduler = make_scheduler(
        runner=fake_runner_all_dim,
        block_manager=block_manager,
        max_prefill_batch_size=2,
        prefix_cache=prefix_cache,
    )

    request_a = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=1,
    )

    request_b = Request(
        request_id="request-b",
        input_ids=[4, 5, 6],
        max_new_tokens=8,
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.step()

    # A finishes and releases its request-owned blocks.
    assert request_a.state == RequestState.FINISHED
    assert request_a.block_table == []
    assert request_a.kv_tokens == 3

    # B is still active.
    assert request_b.state == RequestState.DECODING
    assert request_b.kv_tokens == 3

    assert len(request_b.block_table) == 1

    # The surviving request must still retain its block.
    b_block_id = request_b.block_table[0]

    assert b_block_id in request_b.block_table

    assert block_manager.num_allocated_blocks == 1
    assert block_manager.num_free_blocks == 3
