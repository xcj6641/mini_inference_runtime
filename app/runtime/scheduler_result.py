# app/runtime/scheduler_result.py

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepResult:
    prefetched_request_ids: list[str] = field(
        default_factory=list
    )
    decoded_request_ids: list[str] = field(
        default_factory=list
    )
    finished_request_ids: list[str] = field(
        default_factory=list
    )
    generated_token_ids: dict[str, int] = field(
        default_factory=dict
    )