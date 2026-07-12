from __future__ import annotations

import logging

import torch

from app.runtime.pytorch_model_runner import PyTorchModelRunner
from app.runtime.types import GenerationResult


logger = logging.getLogger(__name__)


def generate(
    runner: PyTorchModelRunner,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    prompt_token_ids = input_ids[0].tolist()
    generated_token_ids: list[int] = []

    prefill_output = runner.prefill(input_ids)

    current_token_id = prefill_output.next_token_id
    past_key_values = prefill_output.past_key_values

    for step in range(max_new_tokens):
        generated_token_ids.append(current_token_id)

        token_text = runner.decode_tokens(
            [current_token_id],
            skip_special_tokens=False,
        )

        logger.info(
            "[generate] step=%d token_id=%d token=%r",
            step,
            current_token_id,
            token_text,
        )

        if current_token_id in runner.eos_token_ids:
            finish_reason = "eos"
            break

        if len(generated_token_ids) >= max_new_tokens:
            finish_reason = "length"
            break

        decode_input_ids = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=runner.device,
        )

        decode_output = runner.decode(
            input_ids=decode_input_ids,
            past_key_values=past_key_values,
        )

        current_token_id = decode_output.next_token_id
        past_key_values = decode_output.past_key_values
    else:
        finish_reason = "length"

    generated_text = runner.decode_tokens(
        generated_token_ids,
        skip_special_tokens=True,
    )

    return GenerationResult(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        text=generated_text,
        finish_reason=finish_reason,
    )