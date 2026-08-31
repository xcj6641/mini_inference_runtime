from __future__ import annotations

from dataclasses import dataclass

import torch

from app.runtime.pytorch_model_runner import (
    PyTorchModelRunner,
)


@dataclass(frozen=True)
class GenerationResult:
    generated_token_ids: list[int]
    text: str
    finish_reason: str


def generate(
    *,
    runner: PyTorchModelRunner,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be positive"
        )

    prefill_output = runner.prefill(
        input_ids=input_ids,
    )

    generated_token_ids: list[int] = []

    current_token_id = (
        prefill_output.next_token_id
    )

    # Dense KV belongs to this simple generation loop,
    # not to Request.
    past_key_values = (
        prefill_output.past_key_values
    )

    while True:
        generated_token_ids.append(
            current_token_id
        )

        # Stop on EOS.
        if current_token_id in runner.eos_token_ids:
            return GenerationResult(
                generated_token_ids=(
                    generated_token_ids
                ),
                text=runner.decode_tokens(
                    generated_token_ids
                ),
                finish_reason="eos",
            )

        # Stop on max_new_tokens.
        if (
            len(generated_token_ids)
            >= max_new_tokens
        ):
            return GenerationResult(
                generated_token_ids=(
                    generated_token_ids
                ),
                text=runner.decode_tokens(
                    generated_token_ids
                ),
                finish_reason="length",
            )

        decode_input_ids = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=runner.device,
        )

        decode_output = runner.decode(
            input_ids=decode_input_ids,
            past_key_values=past_key_values,
        )

        # Carry forward the dense execution KV locally.
        past_key_values = (
            decode_output.past_key_values
        )

        current_token_id = (
            decode_output.next_token_id
        )