from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.runtime.batch import DecodeBatch, PrefillBatch
import torch

from app.runtime.types import BatchedDecodeOutput, BatchedPrefillOutput, DecodeOutput, PrefillOutput

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

    @torch.inference_mode()
    def prefill_batch(
        self,
        batch: PrefillBatch,
    ) -> BatchedPrefillOutput:
        ...

    @abstractmethod
    def decode(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor | None = None,
    ) -> DecodeOutput:
        raise NotImplementedError
    
    @abstractmethod
    def decode_batch(
        self,
        batch: DecodeBatch,
    ) -> BatchedDecodeOutput:
        ...