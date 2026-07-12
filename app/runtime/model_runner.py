from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from app.runtime.types import DecodeOutput, PrefillOutput

import logging

logger = logging.getLogger(__name__)

class ModelRunner(ABC):

    @abstractmethod
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> PrefillOutput:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor | None = None,
    ) -> DecodeOutput:
        raise NotImplementedError
