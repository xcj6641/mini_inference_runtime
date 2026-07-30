from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.runtime.types import RequestState


@dataclass
class Request:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int

    # The reason using `field` here is to ensure that each instance of Request has its own list for generated_ids, 
    # rather than sharing a single list across all instances.
    # When using List[int] = [], Python creates the default value once, when the class is defined.
    generated_ids: list[int] = field(default_factory=list)
    past_key_values: Any | None = None

    prompt_tokens: int = 0 #number of tokens in the prompt
    generated_tokens_count: int = 0

    state: RequestState = RequestState.WAITING
    finish_reason: str | None = None
    error_message: str | None = None

    # This table will store physical block IDs in logical sequence order.

    # Example:

    # request.block_table = [3, 8, 1]

    # means:

    # first logical block  -> physical block 3
    # second logical block -> physical block 8
    # third logical block  -> physical block 1
    
    block_table: list[int] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")

        if not self.input_ids:
            raise ValueError("input_ids cannot be empty")

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero"
            )

        self.prompt_tokens = len(self.input_ids)

    @property
    def has_kv_cache(self) -> bool:
        return self.past_key_values is not None

    @property
    def total_sequence_length(self) -> int:
        return (
            self.prompt_tokens
            + self.generated_tokens_count
        )

    def attach_kv_cache(
        self,
        past_key_values: Any,
    ) -> None:
        if past_key_values is None:
            raise ValueError(
                "past_key_values cannot be None"
            )

        self.past_key_values = past_key_values

    def append_generated_token(
        self,
        token_id: int,
    ) -> None:
        if not isinstance(token_id, int):
            raise TypeError("token_id must be an int")

        self.generated_ids.append(token_id)
        self.generated_tokens_count += 1

    def release_kv_cache(self) -> None:
        self.past_key_values = None

    def mark_finished(
        self,
        finish_reason: str,
    ) -> None:
        self.state = RequestState.FINISHED
        self.finish_reason = finish_reason
        self.release_kv_cache()

    def cancel(self) -> None:
        self.state = RequestState.CANCELLED
        self.finish_reason = "cancelled"
        self.release_kv_cache()

    def mark_failed(
        self,
        error: Exception,
    ) -> None:
        self.state = RequestState.FAILED
        self.finish_reason = "error"
        self.error_message = str(error)
        self.release_kv_cache()