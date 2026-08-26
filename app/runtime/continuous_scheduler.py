# app/runtime/continuous_scheduler.py

import torch
from collections import deque

from app.runtime.request import Request, RequestState
from app.runtime.scheduler_result import StepResult
from app.runtime.kv_cache_utils import (
    split_legacy_kv_cache,
    get_kv_sequence_length,
)
from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.prefix_cache import PrefixCache, PrefixCacheEntry
from app.runtime.kv_block_manager import KVBlockManager


class ContinuousScheduler:
    def __init__(
        self,
        runner,
        batch_builder,
        block_manager,
        paged_kv_cache: PagedKVCache,
        prefix_cache,
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
        self.paged_kv_cache = paged_kv_cache
        self.prefix_cache = prefix_cache
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

    def _batch_equal_length_kv_caches(
        self,
        per_request_caches,
    ):
        if not per_request_caches:
            raise ValueError(
                "per_request_caches must not be empty"
            )

        num_layers = len(
            per_request_caches[0]
        )

        batched_layers = []

        for layer_index in range(num_layers):
            keys = [
                cache[layer_index][0]
                for cache in per_request_caches
            ]

            values = [
                cache[layer_index][1]
                for cache in per_request_caches
            ]

            batched_key = torch.cat(
                keys,
                dim=0,
            )

            batched_value = torch.cat(
                values,
                dim=0,
            )

            batched_layers.append(
                (
                    batched_key,
                    batched_value,
                )
            )

        return tuple(
            batched_layers
        )
    ##########################################
    #                 Prefill                #
    ##########################################
    def _try_attach_longest_cached_prefix(
            self,
            request: Request,
            entry: PrefixCacheEntry,
        ) -> None:
        request.block_table.extend(
            entry.block_ids
        )

        request.kv_tokens += len(
            entry.token_ids
        )

        self.block_manager.retain_blocks(
            entry.block_ids
        )

    def _detach_cached_prefix(
            self,
            request: Request,
            entry: PrefixCacheEntry,
        ) -> None:
        self.block_manager.release_blocks(
            entry.block_ids
        )

        num_blocks = len(entry.block_ids)

        if num_blocks > 0:
            del request.block_table[-num_blocks:]

        request.kv_tokens -= len(
            entry.token_ids
        )

    def _split_prefill_requests_by_cache_hit(
            self,
            requests: list[Request],
        ) -> tuple[list[Request], list[Request]]:
        cache_misses: list[Request] = []
        cache_hits: list[Request] = []

        for request in requests:
            if request.kv_tokens > 0:
                cache_hits.append(request)
            else:
                cache_misses.append(request)

        return (
            cache_misses,
            cache_hits,
        )

    def _select_prefill_requests(
            self,
        ) -> list[Request]:
        selected: list[Request] = []
        remaining: deque[Request] = deque()

        while self.waiting:
            request = self.waiting.popleft()

            # Batch is already full.
            # Keep unselected requests in waiting.
            if (
                len(selected)
                >= self.max_prefill_batch_size
            ):
                remaining.append(request)
                continue

            entry = (
                self.prefix_cache.lookup_longest_prefix(
                    token_ids=request.input_ids,
                    block_size=(
                        self.block_manager.block_size
                    ),
                )
            )

            if entry is not None:
                self._try_attach_longest_cached_prefix(
                    request=request,
                    entry=entry,
                )

            total_block_requirement = (
                len(request.input_ids)
                + self.block_manager.block_size
                - 1
            ) // self.block_manager.block_size

            additional_blocks = (
                total_block_requirement
                - len(request.block_table)
            )

            if (
                additional_blocks
                > self.block_manager.num_free_blocks
            ):
                if entry is not None:
                    self._detach_cached_prefix(
                        request=request,
                        entry=entry,
                    )

                # Could not admit this request.
                # Put it back into waiting.
                remaining.append(request)
                continue

            self.block_manager.ensure_capacity(
                request=request,
                total_tokens=len(
                    request.input_ids
                ),
            )

            # Successfully admitted.
            # Do NOT put it back into waiting.
            selected.append(request)

        self.waiting = remaining

        return selected
   
    def _prefill_token_requirement(
            self,
            request: Request,
        ) -> int:
        return len(request.input_ids)

    def _try_attach_cached_prefix(
            self,
            request: Request,
        ) -> int:
        cacheable_tokens = (
            PrefixCache.get_cacheable_prefix_length(
                num_tokens=len(request.input_ids),
                block_size=self.block_manager.block_size,
            )
        )

        if cacheable_tokens == 0:
            return 0

        prefix_tokens = request.input_ids[
            :cacheable_tokens
        ]

        entry = self.prefix_cache.lookup_longest_prefix(
            token_ids=request.input_ids,
            block_size=self.block_manager.block_size,
        )

        if entry is None:
            return 0

        request.block_table.extend(
            entry.block_ids
        )

        self.block_manager.retain_blocks(
            entry.block_ids
        )

        request.kv_tokens = len(
            entry.token_ids
        )

        return request.kv_tokens

    def _run_full_prefill_batch(
            self,
            requests: list[Request],
        ) -> tuple[list[str], dict[str, int]]:
        if not requests:
            return [], {}

        for request in requests:
            assert request.kv_tokens == 0

        batch = self.batch_builder.build_prefill_batch(
            requests,
            pad_token_id=self.runner.pad_token_id,
            device=self.runner.device,
        )

        output = self.runner.prefill_batch(
            batch
        )

        if (
            len(output.next_token_ids)
            != len(requests)
        ):
            raise RuntimeError(
                "Prefill output token count does not "
                "match request count"
            )

        per_request_caches = (
            split_legacy_kv_cache(
                output.past_key_values
            )
        )

        if (
            len(per_request_caches)
            != len(requests)
        ):
            raise RuntimeError(
                "Split prefill KV-cache count does not "
                "match request count"
            )

        prefetched_request_ids: list[str] = []
        generated_token_ids: dict[str, int] = {}

        for index, request in enumerate(requests):
            per_request_cache = (
                per_request_caches[index]
            )

            prompt_length = len(
                request.input_ids
            )

            physical_kv_length = (
                per_request_cache[0][0].shape[2]
            )

            # Account for left padding in
            # batched prefill output.
            logical_prompt_start = (
                physical_kv_length
                - prompt_length
            )

            self.paged_kv_cache.write_request_kv(
                block_table=request.block_table,
                past_key_values=per_request_cache,
                num_tokens=prompt_length,
                source_start=logical_prompt_start,
            )

            # Entire prompt KV is now valid.
            request.set_kv_tokens_from_prompt()

            # Publish all full-block prefixes.
            self._maybe_cache_prefilled_prefix(
                request
            )

            next_token_id = int(
                output.next_token_ids[index]
            )

            request.append_generated_token(
                next_token_id
            )

            request.state = (
                RequestState.DECODING
            )

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

            if (
                request.state
                == RequestState.DECODING
            ):
                self.active[
                    request.request_id
                ] = request
            else:
                self._complete_request(
                    request
                )

        return (
            prefetched_request_ids,
            generated_token_ids,
        )
    
    def _run_prefill(
            self,
            requests: list[Request],
        ) -> tuple[list[str], dict[str, int]]:
        if not requests:
            return [], {}

        cache_misses, cache_hits = (
            self._split_prefill_requests_by_cache_hit(
                requests
            )
        )

        prefetched_request_ids: list[str] = []
        generated_token_ids: dict[str, int] = {}

        if cache_misses:
            (
                miss_request_ids,
                miss_generated_tokens,
            ) = self._run_full_prefill_batch(
                cache_misses
            )

            prefetched_request_ids.extend(
                miss_request_ids
            )

            generated_token_ids.update(
                miss_generated_tokens
            )

        if cache_hits:
            (
                hit_request_ids,
                hit_generated_tokens,
            ) = self._run_cached_prefill_batch(
                cache_hits
            )

            prefetched_request_ids.extend(
                hit_request_ids
            )

            generated_token_ids.update(
                hit_generated_tokens
            )

        return (
            prefetched_request_ids,
            generated_token_ids,
        )

    def _run_prefill_with_cached_prefix(
            self,
            *,
            request: Request,
            cached_prefix_length: int,
        ) -> tuple[list[str], dict[str, int]]:
        suffix_input_ids = request.input_ids[
            cached_prefix_length:
        ]

        if not suffix_input_ids:
            raise RuntimeError(
                "cached prefix covers the entire prompt"
            )

        cached_past_key_values = (
            self.paged_kv_cache.materialize_request_kv(
                block_table=request.block_table,
                num_tokens=cached_prefix_length,
            )
        )

        suffix_input_ids_tensor = torch.tensor(
            [suffix_input_ids],
            dtype=torch.long,
            device=self.runner.device,
        )

        suffix_length = len(
            suffix_input_ids
        )

        position_ids = torch.arange(
            cached_prefix_length,
            cached_prefix_length + suffix_length,
            dtype=torch.long,
            device=self.runner.device,
        ).unsqueeze(0)

        attention_mask = torch.ones(
            (
                1,
                cached_prefix_length
                + suffix_length,
            ),
            dtype=torch.long,
            device=self.runner.device,
        )

        (
            next_token_id,
            updated_kv,
        ) = self.runner.prefill_with_past(
            input_ids=suffix_input_ids_tensor,
            past_key_values=cached_past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            suffix_lengths=suffix_lengths,

        )

        assert (
            get_kv_sequence_length(updated_kv)
            == len(request.input_ids)
        )

        # Write only newly computed suffix KV.
        self.paged_kv_cache.write_request_kv_prefix_cache(
            block_table=request.block_table,
            past_key_values=updated_kv,
            num_tokens=suffix_length,
            source_start=cached_prefix_length,
            destination_start=cached_prefix_length,
        )

        # Complete prompt KV is now valid.
        request.set_kv_tokens_from_prompt()

        self._maybe_cache_prefilled_prefix(
            request
        )

        request.append_generated_token(
            next_token_id
        )

        request.state = RequestState.DECODING

        self._update_finish_state(
            request=request,
            generated_token_id=next_token_id,
        )

        if request.state == RequestState.DECODING:
            self.active[
                request.request_id
            ] = request
        else:
            self._complete_request(
                request
            )

        return (
            [request.request_id],
            {
                request.request_id: next_token_id
            },
        )

    def _run_cached_prefill_batch(
            self,
            requests: list[Request],
        ) -> tuple[list[str], dict[str, int]]:
        if not requests:
            return [], {}

        # --------------------------------------------------
        # 1. Cached-prefix lengths must still be equal
        # --------------------------------------------------
        cached_prefix_lengths = {
            request.kv_tokens
            for request in requests
        }

        if len(cached_prefix_lengths) != 1:
            raise RuntimeError(
                "cached-prefix batch currently requires "
                "equal cached-prefix lengths"
            )

        cached_prefix_length = (
            requests[0].kv_tokens
        )

        # --------------------------------------------------
        # 2. Suffix lengths may now be different
        # --------------------------------------------------
        suffix_lengths = [
            len(request.input_ids)
            - request.kv_tokens
            for request in requests
        ]

        if any(
            suffix_length <= 0
            for suffix_length in suffix_lengths
        ):
            raise RuntimeError(
                "cached prefix covers the entire prompt"
            )

        max_suffix_length = max(
            suffix_lengths
        )

        batch_size = len(requests)

        # --------------------------------------------------
        # 3. Right-pad suffix input_ids
        #
        # Example:
        #
        # B suffix = [99]
        # C suffix = [100, 101, 102]
        #
        # becomes:
        #
        # [
        #     [99,  PAD, PAD],
        #     [100, 101, 102],
        # ]
        # --------------------------------------------------
        suffix_input_ids_tensor = torch.full(
            (
                batch_size,
                max_suffix_length,
            ),
            fill_value=self.runner.pad_token_id,
            dtype=torch.long,
            device=self.runner.device,
        )

        for batch_index, request in enumerate(
            requests
        ):
            suffix = request.input_ids[
                cached_prefix_length:
            ]

            suffix_length = len(suffix)

            suffix_input_ids_tensor[
                batch_index,
                :suffix_length,
            ] = torch.tensor(
                suffix,
                dtype=torch.long,
                device=self.runner.device,
            )

        # --------------------------------------------------
        # 4. Position IDs
        #
        # Real suffix tokens continue immediately after
        # the cached prefix.
        #
        # Example:
        #
        # B -> [4, 0, 0]
        # C -> [4, 5, 6]
        #
        # Padded positions are masked, so their position
        # IDs are irrelevant.
        # --------------------------------------------------
        position_ids = torch.zeros(
            (
                batch_size,
                max_suffix_length,
            ),
            dtype=torch.long,
            device=self.runner.device,
        )

        for (
            batch_index,
            suffix_length,
        ) in enumerate(
            suffix_lengths
        ):
            position_ids[
                batch_index,
                :suffix_length,
            ] = torch.arange(
                cached_prefix_length,
                cached_prefix_length
                + suffix_length,
                dtype=torch.long,
                device=self.runner.device,
            )

        # --------------------------------------------------
        # 5. Attention mask
        #
        # cached prefix length = 4
        #
        # B suffix length = 1
        # C suffix length = 3
        #
        # B -> [1,1,1,1, 1,0,0]
        # C -> [1,1,1,1, 1,1,1]
        # --------------------------------------------------
        total_physical_length = (
            cached_prefix_length
            + max_suffix_length
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                total_physical_length,
            ),
            dtype=torch.long,
            device=self.runner.device,
        )

        for (
            batch_index,
            suffix_length,
        ) in enumerate(
            suffix_lengths
        ):
            logical_length = (
                cached_prefix_length
                + suffix_length
            )

            attention_mask[
                batch_index,
                :logical_length,
            ] = 1

        # --------------------------------------------------
        # 6. Materialize each request's cached prefix KV
        # --------------------------------------------------
        per_request_caches = [
            self.paged_kv_cache.materialize_request_kv(
                block_table=request.block_table,
                num_tokens=cached_prefix_length,
            )
            for request in requests
        ]

        # --------------------------------------------------
        # 7. Batch the equal-length cached KV
        #
        # Each cache:
        # [1, num_kv_heads, prefix_len, head_dim]
        #
        # becomes:
        # [batch_size, num_kv_heads, prefix_len, head_dim]
        # --------------------------------------------------
        batched_past_key_values = (
            self._batch_equal_length_kv_caches(
                per_request_caches
            )
        )

        # --------------------------------------------------
        # 8. ONE batched runner call
        # --------------------------------------------------
        (
            next_token_ids,
            updated_kv,
        ) = self.runner.prefill_with_past(
            input_ids=suffix_input_ids_tensor,
            past_key_values=batched_past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if len(next_token_ids) != len(requests):
            raise RuntimeError(
                "cached-prefix prefill output token "
                "count does not match request count"
            )

        # Physical output KV length is:
        #
        # cached_prefix_length + max_suffix_length
        #
        # for EVERY request.
        expected_physical_kv_length = (
            cached_prefix_length
            + max_suffix_length
        )

        if (
            get_kv_sequence_length(updated_kv)
            != expected_physical_kv_length
        ):
            raise RuntimeError(
                "cached-prefix prefill output KV length "
                "does not match padded batch length"
            )

        # --------------------------------------------------
        # 9. Split batched KV back into per-request KV
        # --------------------------------------------------
        per_request_updated_kv = (
            split_legacy_kv_cache(
                updated_kv
            )
        )

        if (
            len(per_request_updated_kv)
            != len(requests)
        ):
            raise RuntimeError(
                "cached-prefix KV batch size does not "
                "match request count"
            )

        prefetched_request_ids: list[str] = []
        generated_token_ids: dict[str, int] = {}

        # --------------------------------------------------
        # 10. Write ONLY each request's REAL suffix KV
        # --------------------------------------------------
        for (
            request,
            suffix_length,
            next_token_id,
            request_updated_kv,
        ) in zip(
            requests,
            suffix_lengths,
            next_token_ids,
            per_request_updated_kv,
            strict=True,
        ):
            # Important:
            #
            # request_updated_kv physically contains:
            #
            # [cached prefix][real suffix][padding]
            #
            # But we only copy:
            #
            # [real suffix]
            #
            # into this request's paged cache.
            self.paged_kv_cache.write_request_kv_prefix_cache(
                block_table=request.block_table,
                past_key_values=request_updated_kv,
                num_tokens=suffix_length,
                source_start=cached_prefix_length,
                destination_start=cached_prefix_length,
            )

            # Logical KV now covers the complete prompt.
            request.set_kv_tokens_from_prompt()

            self._maybe_cache_prefilled_prefix(
                request
            )

            request.append_generated_token(
                next_token_id
            )

            request.state = (
                RequestState.DECODING
            )

            self._update_finish_state(
                request=request,
                generated_token_id=next_token_id,
            )

            if (
                request.state
                == RequestState.DECODING
            ):
                self.active[
                    request.request_id
                ] = request
            else:
                self._complete_request(
                    request
                )

            prefetched_request_ids.append(
                request.request_id
            )

            generated_token_ids[
                request.request_id
            ] = next_token_id

        return (
            prefetched_request_ids,
            generated_token_ids,
        )
    ##########################################
    #                 Decode                 #
    ##########################################
    def _decode_token_requirement(
            self,
            request: Request,
        ) -> int:
        return request.kv_tokens + 1

    def _select_decode_requests(
            self,
        ) -> list[Request]:
        selected: list[Request] = []

        available_blocks = (
            self.block_manager.num_free_blocks
        )

        required_blocks = 0

        for request in self.active.values():
            if request.state != RequestState.DECODING:
                continue

            if (
                len(selected)
                >= self.max_decode_batch_size
            ):
                break


            token_requirement = (
                self._decode_token_requirement(
                    request
                )
            )

            additional_blocks = (
                self.block_manager
                .additional_blocks_required(
                    request,
                    token_requirement,
                )
            )

            if (
                required_blocks
                + additional_blocks
                > available_blocks
            ):
                continue

            selected.append(request)

            required_blocks += (
                additional_blocks
            )

        return selected

    def _run_decode(
            self,
            requests: list[Request],
        ) -> tuple[list[str], dict[str, int]]:
        if not requests:
            return [], {}

        input_request_cache = [
            self.paged_kv_cache.materialize_request_kv(
                block_table=request.block_table,
                num_tokens=request.kv_tokens,
            )
            for request in requests
        ]

        batch = (
            self.batch_builder
            .build_decode_batch(
                requests,
                per_request_caches=input_request_cache,
                device=self.runner.device,
            )
        )

        output = self.runner.decode_batch(batch)

        if len(output.next_token_ids) != len(requests):
            raise RuntimeError(
                "Decode output token count does not "
                "match request count"
            )

        output_request_caches = (
            split_legacy_kv_cache(
                output.past_key_values
            )
        )

        if len(output_request_caches) != len(requests):
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

            updated_cache = output_request_caches[index]

            new_kv_tokens = request.kv_tokens + 1

            # # Temporary old/reference path.
            # request.attach_kv_cache(
            #     updated_cache
            # )

            # New paged persistent path.
            physical_kv_length = (
                updated_cache[0][0].shape[2]
            )

            source_start = (
                physical_kv_length - new_kv_tokens
            )

            self.paged_kv_cache.write_request_kv(
                block_table=request.block_table,
                past_key_values=updated_cache,
                num_tokens=new_kv_tokens,
                source_start=source_start,
            )

            request.increment_kv_tokens()

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

        self.block_manager.release_blocks(
            request.block_table
        )

        request.block_table.clear()

        self.completed[
            request.request_id
        ] = request

    def step(self) -> StepResult:
        self.tick_id += 1

        # PHASE 1 - Planning
        # select decode batch
        decode_requests = (
            self._select_decode_requests()
        )

        # reserve decode growth
        decode_requirements = [
            (
                request,
                self._decode_token_requirement(
                    request
                ),
            )
            for request in decode_requests
        ]

        # reserve blocks for decode requests before prefill to avoid deadlock
        self.block_manager.ensure_batch_capacity(
            decode_requirements
        )

        # select prefill batch
        prefill_requests = self._select_prefill_requests()

        # reserve prefill capacity
        prefill_requirements = [
            (
                request,
                self._prefill_token_requirement(
                    request
                ),
            )
            for request in prefill_requests
        ]

        self.block_manager.ensure_batch_capacity(
            prefill_requirements
        )

        # PHASE 2 - Execution

        (
            prefetched_request_ids,
            prefill_generated_token_ids,
        ) = self._run_prefill(prefill_requests)

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

    ##########################################
    #             Prefix Cache               #
    ##########################################
    def _maybe_cache_prefilled_prefix(
        self,
        request: Request,
    ) -> None:
        block_size = self.block_manager.block_size

        num_full_blocks = (
            len(request.input_ids)
            // block_size
        )

        for num_blocks in range(
            1,
            num_full_blocks + 1,
        ):
            prefix_length = (
                num_blocks * block_size
            )

            prefix_tokens = request.input_ids[
                :prefix_length
            ]

            existing = self.prefix_cache.lookup(
                prefix_tokens
            )

            if existing is not None:
                continue

            prefix_blocks = request.block_table[
                :num_blocks
            ]

            self.prefix_cache.insert(
                token_ids=prefix_tokens,
                block_ids=prefix_blocks,
            )

    def _maybe_cache_longest_prefilled_prefix(
        self,
        request: Request,
    ) -> None:
        prompt_length = len(request.input_ids)

        cacheable_tokens = (
            prompt_length
            // self.block_manager.block_size
        ) * self.block_manager.block_size

        if cacheable_tokens == 0:
            return

        prefix_tokens = request.input_ids[
            :cacheable_tokens
        ]

        # Avoid retaining the same blocks twice.
        existing = self.prefix_cache.lookup(
            prefix_tokens
        )

        if existing is not None:
            return

        num_cacheable_blocks = (
            cacheable_tokens
            // self.block_manager.block_size
        )

        prefix_blocks = request.block_table[
            :num_cacheable_blocks
        ]

        self.prefix_cache.insert(
            token_ids=prefix_tokens,
            block_ids=prefix_blocks,
        )

