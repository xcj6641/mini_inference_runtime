from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrefillOutput:
    next_token_id: int
    past_key_values: Any
    logits: torch.Tensor


@dataclass(frozen=True)
class DecodeOutput:
    next_token_id: int
    past_key_values: Any
    logits: torch.Tensor


@dataclass(frozen=True)
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    text: str
    finish_reason: str