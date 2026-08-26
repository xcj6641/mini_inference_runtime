import torch
from app.runtime.request import Request
from app.runtime.types import BatchedPrefillOutput, BatchedDecodeOutput
from app.runtime.kv_cache_utils import get_kv_sequence_length
from app.runtime.batch import DecodeBatch, PrefillBatch
from app.runtime.batch_builder import BatchBuilder

class FakeRunner:
    def __init__(
            self, 
            head_dim: int=1,
            num_layers: int=1,
            num_kv_heads: int=1,
        ) -> None:
        self.prefill_calls: list[list[str]] = []
        self.prefill_with_past_batch_sizes: list[int] = []
        self.decode_calls: list[list[str]] = []

        self.decode_counts: dict[str, int] = {}

        self.eos_token_ids: set[int] = {9999}
        self.eos_on_decode_for: set[str] = set()

        self.pad_token_id: int = 0
        self.device = torch.device("cpu")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
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
            batch: PrefillBatch,
        ) -> BatchedPrefillOutput:
        self.prefill_calls.append(
            list(batch.request_ids)
        )

        batch_size = batch.input_ids.shape[0]

        # Mimic padded batched prefill:
        # physical KV length equals the longest prompt
        # in the batch, while each Request.kv_tokens
        # still tracks its own logical prompt length.
        physical_kv_length = int(
            batch.input_ids.shape[1]
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

    def prefill_with_past(
            self,
            *,
            input_ids: torch.Tensor,
            past_key_values,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
        ):
        batch_size = int(
            input_ids.shape[0]
        )

        self.prefill_with_past_batch_sizes.append(
            batch_size
        )

        cached_kv_length = (
            get_kv_sequence_length(
                past_key_values
            )
        )

        padded_suffix_length = int(
            input_ids.shape[1]
        )

        updated_kv_length = (
            cached_kv_length
            + padded_suffix_length
        )

        next_token_ids = [
            1000 + index
            for index in range(batch_size)
        ]

        updated_kv = (
            self._make_batched_kv_cache(
                batch_size=batch_size,
                sequence_length=updated_kv_length,
            )
        )

        return (
            next_token_ids,
            updated_kv,
        )

    def decode_batch(
            self,
            batch: DecodeBatch,
        ) -> BatchedDecodeOutput:
        self.decode_calls.append(
            list(batch.request_ids)
        )

        batch_size = int(
            batch.input_ids.shape[0]
        )

        old_physical_kv_length = (
            get_kv_sequence_length(
                batch.past_key_values
            )
        )

        next_token_ids: list[int] = []

        for request_id in batch.request_ids:
            decode_count = self.decode_counts.get(
                request_id,
                0,
            )

            if (
                request_id
                in self.eos_on_decode_for
            ):
                next_token_id = 9999
            else:
                next_token_id = (
                    2001 + decode_count
                )

            next_token_ids.append(
                next_token_id
            )

            self.decode_counts[request_id] = (
                decode_count + 1
            )

        return BatchedDecodeOutput(
            next_token_ids=next_token_ids,
            past_key_values=(
                self._make_batched_kv_cache(
                    batch_size=batch_size,
                    sequence_length=(
                        old_physical_kv_length + 1
                    ),
                )
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

        self._real_builder = BatchBuilder()

    def build_prefill_batch(
        self,
        requests: list[Request],
        pad_token_id: int | None,
        device: torch.device | None,
    ):
        self.prefill_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        return self._real_builder.build_prefill_batch(
            requests=requests,
            pad_token_id=pad_token_id,
            device=device,
        )

    def build_decode_batch(
        self,
        requests: list[Request],
        per_request_caches,
        device: torch.device | None,
    ):
        self.decode_calls.append(
            [
                request.request_id
                for request in requests
            ]
        )

        return self._real_builder.build_decode_batch(
            requests=requests,
            per_request_caches=per_request_caches,
            device=device,
        )

