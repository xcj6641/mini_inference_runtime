# app/runtime/continuous_scheduler.py

from collections import deque

from app.runtime.request import Request, RequestState
from app.runtime.scheduler_result import StepResult
from app.runtime.kv_cache_utils import (
    split_legacy_kv_cache,
    get_kv_sequence_length,
)


class ContinuousScheduler:
    def __init__(
        self,
        runner,
        batch_builder,
        block_manager,
        max_prefill_batch_size: int = 4,
        max_decode_batch_size: int = 4,
    ) -> None:
        if max_prefill_batch_size <= 0:
            raise ValueError(
                "max_prefill_batch_size must be positive"
            )

        if max_decode_batch_size <= 0:
            raise ValueError(
                "max_decode_batch_size must be positive"
            )

        self.runner = runner
        self.batch_builder = batch_builder
        self.block_manager = block_manager
        
        self.max_prefill_batch_size = (
            max_prefill_batch_size
        )
        self.max_decode_batch_size = (
            max_decode_batch_size
        )

        self.waiting: deque[Request] = deque()
        self.active: dict[str, Request] = {}
        self.completed: dict[str, Request] = {}

        self.tick_id = 0

    def add_request(self, request: Request) -> None:
        if self._contains_request_id(
            request.request_id
        ):
            raise ValueError(
                "Duplicate request ID: "
                f"{request.request_id}"
            )

        if request.state != RequestState.WAITING:
            raise ValueError(
                "New request must be in WAITING state, "
                f"but got {request.state}"
            )

        self.waiting.append(request)

    def has_pending_work(self) -> bool:
        return bool(self.waiting or self.active)

    def get_request(
        self,
        request_id: str,
    ) -> Request:
        for request in self.waiting:
            if request.request_id == request_id:
                return request

        if request_id in self.active:
            return self.active[request_id]

        if request_id in self.completed:
            return self.completed[request_id]

        raise KeyError(
            f"Unknown request ID: {request_id}"
        )

    def _select_prefill_requests(
        self,
    ) -> list[Request]:
        selected: list[Request] = []

        while (
            self.waiting
            and len(selected)
            < self.max_prefill_batch_size
        ):
            selected.append(self.waiting.popleft())

        return selected
    
    def _run_prefill(
        self,
    ) -> tuple[list[str], dict[str, int]]:
        requests = self._select_prefill_requests()

        if not requests:
            return [], {}

        batch = self.batch_builder.build_prefill_batch(
            requests,
            pad_token_id=self.runner.pad_token_id,
            device=self.runner.device,
        )

        output = self.runner.prefill_batch(batch)

        if len(output.next_token_ids) != len(requests):
            raise RuntimeError(
                "Prefill output token count does not "
                "match request count"
            )

        per_request_caches = (
            split_legacy_kv_cache(
                output.past_key_values
            )
        )

        if len(per_request_caches) != len(requests):
            raise RuntimeError(
                "Split prefill KV-cache count does not "
                "match request count"
            )

        prefetched_request_ids: list[str] = []
        generated_token_ids: dict[str, int] = {}

        for index, request in enumerate(requests):
            next_token_id = int(
                output.next_token_ids[index]
            )

            request.attach_kv_cache(
                per_request_caches[index]
            )

            request.append_generated_token(
                next_token_id
            )

            request.state = RequestState.DECODING

            self._update_finish_state(
                request=request,
                generated_token_id=next_token_id,
            )

            prefetched_request_ids.append(
                request.request_id
            )

            generated_token_ids[
                request.request_id
            ] = next_token_id

            if request.state == RequestState.DECODING:
                self.active[
                    request.request_id
                ] = request
            else:
                self._complete_request(request)

        return (
            prefetched_request_ids,
            generated_token_ids,
        )

    def _select_decode_requests(
        self,
    ) -> list[Request]:
        selected: list[Request] = []
        target_kv_length: int | None = None

        for request in self.active.values():
            if request.state != RequestState.DECODING:
                continue

            if request.past_key_values is None:
                raise RuntimeError(
                    f"Request {request.request_id} "
                    "has no KV cache"
                )

            kv_length = get_kv_sequence_length(
                request.past_key_values
            )

            if target_kv_length is None:
                target_kv_length = kv_length

            if kv_length != target_kv_length:
                continue

            selected.append(request)

            if len(selected) >= self.max_decode_batch_size:
                break

        return selected

    def _run_decode(
        self,
        requests: list[Request],
    ) -> tuple[list[str], dict[str, int]]:
        if not requests:
            return [], {}

        batch = (
            self.batch_builder
            .build_equal_length_decode_batch(
                requests,
                device=self.runner.device,
            )
        )

        output = self.runner.decode_batch(batch)

        if len(output.next_token_ids) != len(requests):
            raise RuntimeError(
                "Decode output token count does not "
                "match request count"
            )

        per_request_caches = (
            split_legacy_kv_cache(
                output.past_key_values
            )
        )

        if len(per_request_caches) != len(requests):
            raise RuntimeError(
                "Decode output KV cache batch size "
                "does not match request count"
            )

        generated_token_ids: dict[str, int] = {}
        decoded_request_ids: list[str] = []

        for index, request in enumerate(requests):
            next_token_id = int(
                output.next_token_ids[index]
            )

            request.attach_kv_cache(
                per_request_caches[index]
            )

            request.append_generated_token(
                token_id=next_token_id  
            )

            decoded_request_ids.append(
                request.request_id
            )
            
            generated_token_ids[
                request.request_id
            ] = next_token_id

            self._update_finish_state(
                request=request,
                generated_token_id=next_token_id,
            )

            if request.state == RequestState.FINISHED:
                self._complete_request(request)

        return (
            decoded_request_ids,
            generated_token_ids,
        )

    def _complete_request(
        self,
        request: Request,
    ) -> None:
        self.active.pop(
            request.request_id,
            None,
        )

        request.release_kv_cache()

        self.completed[request.request_id] = request

    def step(self) -> StepResult:
        self.tick_id += 1

        decode_requests = (
            self._select_decode_requests()
        )

        (
            prefetched_request_ids,
            prefill_generated_token_ids,
        ) = self._run_prefill()

        (
            decoded_request_ids,
            decode_generated_token_ids,
        ) = self._run_decode(decode_requests)


        generated_token_ids = {
            **prefill_generated_token_ids,
            **decode_generated_token_ids,
        }

        finished_request_ids = [
            request_id
            for request_id in (
                prefetched_request_ids
                + decoded_request_ids
            )
            if request_id in self.completed
        ]

        return StepResult(
            prefetched_request_ids=(
                prefetched_request_ids
            ),
            decoded_request_ids=(
                decoded_request_ids
            ),
            finished_request_ids=(
                finished_request_ids
            ),
            generated_token_ids=generated_token_ids,
        )

    def _contains_request_id(
        self,
        request_id: str,
    ) -> bool:
        if request_id in self.active:
            return True

        if request_id in self.completed:
            return True

        return any(
            request.request_id == request_id
            for request in self.waiting
        )

    def _update_finish_state(
        self,
        request: Request,
        generated_token_id: int,
    ) -> None:
        eos_token_ids = getattr(
            self.runner,
            "eos_token_ids",
            set(),
        )

        if generated_token_id in eos_token_ids:
            request.state = RequestState.FINISHED
            request.finish_reason = "eos"
            return

        if (
            request.generated_tokens_count
            >= request.max_new_tokens
        ):
            request.state = RequestState.FINISHED
            request.finish_reason = "length"

