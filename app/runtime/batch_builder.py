from __future__ import annotations

import torch

from app.runtime.batch import DecodeBatch, PrefillBatch
from app.runtime.request import Request
from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
    stack_legacy_kv_caches,
    left_pad_legacy_kv_cache,
)


class BatchBuilder:
    def build_equal_length_prefill_batch(
        self,
        requests: list[Request],
        *,
        device: torch.device | str = "cpu",
    ) -> PrefillBatch:
        if not requests:
            raise ValueError("requests cannot be empty")

        prompt_lengths = {
            len(request.input_ids)
            for request in requests
        }

        if len(prompt_lengths) != 1:
            raise ValueError(
                "All requests must have equal prompt length"
            )

        if next(iter(prompt_lengths)) == 0:
            raise ValueError(
                "Request input_ids cannot be empty"
            )

        input_ids = torch.tensor(
            [
                request.input_ids
                for request in requests
            ],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
            device=device,
        )

        return PrefillBatch(
            request_ids=[
                request.request_id
                for request in requests
            ],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    def build_prefill_batch(
        self,
        requests: list[Request],
        *,
        pad_token_id: int,
        device: torch.device | str = "cpu",
    ) -> PrefillBatch:
        if not requests:
            raise ValueError("requests cannot be empty")

        if any(
            len(request.input_ids) == 0
            for request in requests
        ):
            raise ValueError(
                "Request input_ids cannot be empty"
            )

        max_prompt_length = max(
            len(request.input_ids)
            for request in requests
        )

        padded_input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for request in requests:
            prompt_length = len(request.input_ids)

            padding_length = (
                max_prompt_length - prompt_length
            )

            padded_input_ids.append(
                [pad_token_id] * padding_length
                + request.input_ids
            )

            attention_masks.append(
                [0] * padding_length
                + [1] * prompt_length
            )

        input_ids = torch.tensor(
            padded_input_ids,
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=device,
        )

        position_ids = (
            attention_mask.cumsum(dim=-1) - 1
        )

        position_ids.masked_fill_(
            attention_mask == 0,
            0,
        )

        return PrefillBatch(
            request_ids=[
                request.request_id
                for request in requests
            ],
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    
    def build_decode_batch(
    #def build_equal_length_decode_batch(
        self,
        requests: list[Request],
        *,
        per_request_caches,
        device: torch.device | str = "cpu",
    ) -> DecodeBatch:
        if not requests:
            raise ValueError("requests cannot be empty")

        if len(requests) != len(per_request_caches):
            raise ValueError(
                "requests and per_request_caches "
                "must have the same length"
            )

        for request in requests:
            if not request.generated_ids:
                raise ValueError(
                    f"Request {request.request_id} "
                    "has no generated token for decode"
                )

        kv_lengths = [
            get_kv_sequence_length(cache)
            for cache in per_request_caches
        ]

        max_kv_length = max(kv_lengths)

        padded_request_caches = [
            left_pad_legacy_kv_cache(
                cache,
                target_length=max_kv_length,
            )
            for cache in per_request_caches
        ]

        input_ids = torch.tensor(
            [
                [request.generated_ids[-1]]
                for request in requests
            ],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.zeros(
            (
                len(requests),
                max_kv_length + 1,
            ),
            dtype=torch.long,
            device=device,
        )
        for batch_index, kv_length in enumerate(
            kv_lengths
        ):
            pad_length = (
                max_kv_length - kv_length
            )

            attention_mask[
                batch_index,
                pad_length:,
            ] = 1

        batched_cache = stack_legacy_kv_caches(
            padded_request_caches
        )
        position_ids = torch.tensor(
            [
                [kv_length]
                for kv_length in kv_lengths
            ],
            dtype=torch.long,
            device=device,
        )

        return DecodeBatch(
            request_ids=[
                request.request_id
                for request in requests
            ],
            input_ids=input_ids,
            past_key_values=batched_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

