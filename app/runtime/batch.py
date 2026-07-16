from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillBatch:
    request_ids: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape "
                "as input_ids"
            )

        if (
            self.position_ids is not None
            and self.position_ids.shape
            != self.input_ids.shape
        ):
            raise ValueError(
                "position_ids must have the same shape "
                "as input_ids"
            )

        if len(self.request_ids) != self.input_ids.shape[0]:
            raise ValueError(
                "request_ids length must equal batch size"
            )

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DecodeBatch:
    request_ids: list[str]
    input_ids: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError(
                "decode input_ids must have shape "
                "[batch_size, 1]"
            )

        if self.input_ids.shape[1] != 1:
            raise ValueError(
                "decode input_ids sequence length "
                "must equal 1"
            )

        if len(self.request_ids) != self.input_ids.shape[0]:
            raise ValueError(
                "request_ids length must match "
                "decode batch size"
            )

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])