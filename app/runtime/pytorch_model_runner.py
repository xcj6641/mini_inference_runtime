from __future__ import annotations

import logging
from typing import Any

from app.runtime.batch import DecodeBatch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.runtime.types import BatchedDecodeOutput, BatchedPrefillOutput, DecodeOutput, PrefillOutput
from app.runtime.model_runner import ModelRunner

from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
    move_cache_to_device,
)

logger = logging.getLogger(__name__)


class PyTorchModelRunner(ModelRunner):
    def __init__(
            self,
            model_name: str,
            device: str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._resolve_dtype(self.device)

        logger.info(
            "Loading model model=%s device=%s dtype=%s",
            self.model_name,
            self.device,
            self.dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError(
                    "Tokenizer has neither pad nor EOS token"
                )

            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Model loaded model=%s eos_token_id=%s",
            self.model_name,
            self.eos_token_ids,
        )

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            return torch.float16

        return torch.float32

    @property
    def eos_token_ids(self) -> set[int]:
        eos_token_id = self.tokenizer.eos_token_id

        if eos_token_id is None:
            return set()

        if isinstance(eos_token_id, int):
            return {eos_token_id}

        return set(eos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        value = self.tokenizer.pad_token_id

        if value is None:
            raise RuntimeError(
                "Tokenizer has no pad_token_id"
            )

        return int(value)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return encoded["input_ids"].to(self.device)

    def encode_chat_prompt(
            self,
            user_message: str,
            system_message: str | None = None,
        ) -> torch.Tensor:
        messages: list[dict[str, str]] = []

        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        return input_ids.to(self.device)

    def decode_tokens(
            self,
            token_ids: list[int],
            skip_special_tokens: bool = True,
        ) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    @torch.inference_mode()
    def prefill(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> PrefillOutput:
        self._validate_prefill_input(input_ids)

        input_ids = input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        next_token_id = int(torch.argmax(last_token_logits, dim=-1).item())

        cache_length = get_kv_sequence_length(outputs.past_key_values)

        logger.info(
            "[prefill] input_shape=%s logits_shape=%s "
            "next_token_id=%d cache_length=%s",
            tuple(input_ids.shape),
            tuple(logits.shape),
            next_token_id,
            cache_length,
        )

        return PrefillOutput(
            next_token_id=next_token_id,
            past_key_values=outputs.past_key_values,
            logits=logits,
        )

    @torch.inference_mode()
    def prefill_batch(
            self,
            batch: PrefillBatch,
        ) -> BatchedPrefillOutput:
        input_ids = batch.input_ids.to(self.device)

        attention_mask = batch.attention_mask.to(
            self.device
        )

        position_ids = batch.position_ids

        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits

        expected_batch_size = batch.batch_size

        if logits.ndim != 3:
            raise RuntimeError(
                "Expected logits with shape "
                "[batch_size, sequence_length, vocab_size], "
                f"got {tuple(logits.shape)}"
            )

        if logits.shape[0] != expected_batch_size:
            raise RuntimeError(
                "Model output batch size does not match "
                "input batch size"
            )

        last_token_logits = logits[:, -1, :]

        next_token_ids_tensor = torch.argmax(
            last_token_logits,
            dim=-1,
        )

        next_token_ids = [
            int(token_id)
            for token_id
            in next_token_ids_tensor.detach().cpu().tolist()
        ]

        return BatchedPrefillOutput(
            next_token_ids=next_token_ids,
            past_key_values=outputs.past_key_values,
            logits=logits,
        )

    @torch.inference_mode()
    def decode(
            self,
            input_ids: torch.Tensor,
            past_key_values: Any,
            attention_mask: torch.Tensor | None = None,
        ) -> DecodeOutput:
        self._validate_decode_input(input_ids, past_key_values)

        input_ids = input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        next_token_id = int(torch.argmax(last_token_logits, dim=-1).item())

        cache_length = get_kv_sequence_length(outputs.past_key_values)

        logger.info(
            "[decode] input_shape=%s logits_shape=%s "
            "next_token_id=%d cache_length=%s",
            tuple(input_ids.shape),
            tuple(logits.shape),
            next_token_id,
            cache_length,
        )

        return DecodeOutput(
            next_token_id=next_token_id,
            past_key_values=outputs.past_key_values,
            logits=logits,
        )

    @torch.inference_mode()
    def decode_batch(
            self,
            batch: DecodeBatch,
        ) -> BatchedDecodeOutput:
        input_ids = batch.input_ids.to(self.device)

        past_key_values = move_cache_to_device(
            batch.past_key_values,
            self.device,
        )

        attention_mask = batch.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                self.device
            )

        position_ids = batch.position_ids
        if position_ids is not None:
            position_ids = position_ids.to(
                self.device
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits

        if logits.shape[:2] != (
            batch.batch_size,
            1,
        ):
            raise RuntimeError(
                "Expected decode logits with shape "
                "[batch_size, 1, vocab_size], "
                f"got {tuple(logits.shape)}"
            )

        last_token_logits = logits[:, -1, :]

        next_token_ids_tensor = torch.argmax(
            last_token_logits,
            dim=-1,
        )

        next_token_ids = [
            int(token_id)
            for token_id in next_token_ids_tensor
            .detach()
            .cpu()
            .tolist()
        ]

        return BatchedDecodeOutput(
            next_token_ids=next_token_ids,
            past_key_values=outputs.past_key_values,
            logits=logits,
        )

    @staticmethod
    def _validate_prefill_input(input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "prefill input_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if input_ids.shape[0] != 1:
            raise ValueError(
                "Day 11 PyTorchModelRunner only supports batch_size=1"
            )

        if input_ids.shape[1] == 0:
            raise ValueError("prefill input_ids cannot be empty")

    @staticmethod
    def _validate_decode_input(
            input_ids: torch.Tensor,
            past_key_values: Any,
        ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "decode input_ids must have shape [batch_size, 1]"
            )

        if input_ids.shape != (1, 1):
            raise ValueError(
                "Day 11 decode expects exactly one new token "
                "with shape [1, 1]"
            )

        if past_key_values is None:
            raise ValueError(
                "decode requires past_key_values returned by prefill "
                "or a previous decode step"
            )
        
    @torch.inference_mode()
    def prefill_with_past(
            self,
            *,
            input_ids: torch.Tensor,
            past_key_values,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            suffix_lengths: list[int],
        ) -> tuple[list[int], tuple]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, seq_len]"
            )

        if input_ids.shape[1] <= 0:
            raise ValueError(
                "suffix input_ids must contain "
                "at least one token"
            )

        batch_size = input_ids.shape[0]

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, total_seq_len]"
            )

        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                "attention_mask batch size must match "
                "input_ids batch size"
            )

        if position_ids.ndim != 2:
            raise ValueError(
                "position_ids must have shape "
                "[batch_size, suffix_len]"
            )

        if position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids shape must match "
                "input_ids shape"
            )

        if past_key_values is None:
            raise ValueError(
                "past_key_values must not be None"
            )

        if len(past_key_values) == 0:
            raise ValueError(
                "past_key_values must contain "
                "at least one layer"
            )

        first_key = past_key_values[0][0]

        if first_key.shape[0] != batch_size:
            raise ValueError(
                "past_key_values batch size must match "
                "input_ids batch size"
            )

        cached_kv_length = int(
            first_key.shape[2]
        )

        max_suffix_length = int(
            input_ids.shape[1]
        )
        if len(suffix_lengths) != batch_size:
            raise ValueError(
                "suffix_lengths size must match batch size"
            )

        if any(
            suffix_length <= 0
            for suffix_length in suffix_lengths
        ):
            raise ValueError(
                "every request must contain at least "
                "one real suffix token"
            )

        if any(
            suffix_length > max_suffix_length
            for suffix_length in suffix_lengths
        ):
            raise ValueError(
                "suffix length cannot exceed "
                "padded suffix length"
            )

        expected_attention_length = (
            cached_kv_length
            + max_suffix_length
        )

        if (
            attention_mask.shape[1]
            != expected_attention_length
        ):
            raise ValueError(
                "attention_mask length must equal "
                "cached KV length + padded suffix length"
            )

        input_ids = input_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        attention_mask = attention_mask.to(
            device=self.device,
        )

        position_ids = position_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        if output.logits.shape[:2] != (
            batch_size,
            max_suffix_length,
        ):
            raise RuntimeError(
                "prefill_with_past logits shape does not "
                "match padded suffix batch"
            )

        suffix_lengths_tensor = torch.tensor(
            suffix_lengths,
            dtype=torch.long,
            device=output.logits.device,
        )

        last_suffix_indices = (
            suffix_lengths_tensor - 1
        )

        batch_indices = torch.arange(
            batch_size,
            device=output.logits.device,
        )

        last_token_logits = output.logits[
            batch_indices,
            last_suffix_indices,
            :,
        ]

        next_token_ids_tensor = torch.argmax(
            last_token_logits,
            dim=-1,
        )

        next_token_ids = [
            int(token_id)
            for token_id in (
                next_token_ids_tensor
                .detach()
                .cpu()
                .tolist()
            )
        ]

        return (
            next_token_ids,
            output.past_key_values,
        )

