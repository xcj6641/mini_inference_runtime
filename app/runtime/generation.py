from __future__ import annotations

import logging

import torch

from app.runtime.pytorch_model_runner import PyTorchModelRunner
from app.runtime.request import Request
from app.runtime.types import GenerationResult, RequestState
from app.runtime.kv_cache_utils import get_kv_sequence_length


logger = logging.getLogger(__name__)


def generate(
    runner: PyTorchModelRunner,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    """
    Backward-compatible generation API.

    Existing callers can continue passing a tensor, but the real
    generation work is now performed through a Request object.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape "
            "[batch_size, sequence_length]"
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "generate currently supports batch_size=1"
        )

    request = Request(
        request_id="direct-generate",
        input_ids=input_ids[0].tolist(),
        max_new_tokens=max_new_tokens,
    )

    return generate_request(
        runner=runner,
        request=request,
    )


def generate_request(
    runner: PyTorchModelRunner,
    request: Request,
) -> GenerationResult:
    """
    Generate tokens while the Request owns its KV cache.

    Lifecycle:

    WAITING
        -> PREFILLING
        -> DECODING
        -> FINISHED / FAILED
    """
    try:
        request.state = RequestState.PREFILLING

        input_ids = torch.tensor(
            [request.input_ids],
            dtype=torch.long,
            device=runner.device,
        )

        prefill_output = runner.prefill(
            input_ids=input_ids,
        )

        request.attach_kv_cache(
            prefill_output.past_key_values
        )
        request.set_kv_tokens_from_prompt()

        request.state = RequestState.DECODING
        current_token_id = prefill_output.next_token_id

        while True:
            request.append_generated_token(
                current_token_id
            )

            token_text = runner.decode_tokens(
                [current_token_id],
                skip_special_tokens=False,
            )

            logger.info(
                "[generate-request] request_id=%s "
                "generated_tokens=%d token_id=%d token=%r",
                request.request_id,
                request.generated_tokens_count,
                current_token_id,
                token_text,
            )

            if current_token_id in runner.eos_token_ids:
                result = _build_generation_result(
                    runner=runner,
                    request=request,
                    finish_reason="eos",
                )

                final_cache_length = get_kv_sequence_length(
                    request.past_key_values
                )

                logger.info(
                    "[kv-lifecycle] request_id=%s "
                    "state=%s cache_length=%d "
                    "generated_tokens=%d",
                    request.request_id,
                    request.state.name,
                    final_cache_length,
                    request.generated_tokens_count,
                )
                request.mark_finished("eos")
                logger.info(
                    "[kv-lifecycle] request_id=%s "
                    "state=%s has_kv_cache=%s",
                    request.request_id,
                    request.state.name,
                    request.has_kv_cache,
                )
                return result

            if (
                request.generated_tokens_count
                >= request.max_new_tokens
            ):
                result = _build_generation_result(
                    runner=runner,
                    request=request,
                    finish_reason="length",
                )
                final_cache_length = get_kv_sequence_length(
                    request.past_key_values
                )

                logger.info(
                    "[kv-lifecycle] request_id=%s "
                    "state=%s cache_length=%d "
                    "generated_tokens=%d",
                    request.request_id,
                    request.state.name,
                    final_cache_length,
                    request.generated_tokens_count,
                )
                request.mark_finished("length")
                logger.info(
                    "[kv-lifecycle] request_id=%s "
                    "state=%s has_kv_cache=%s",
                    request.request_id,
                    request.state.name,
                    request.has_kv_cache,
                )
                return result

            decode_input_ids = torch.tensor(
                [[current_token_id]],
                dtype=torch.long,
                device=runner.device,
            )

            decode_output = runner.decode(
                input_ids=decode_input_ids,
                past_key_values=request.past_key_values,
            )

            request.attach_kv_cache(
                decode_output.past_key_values
            )
            request.increment_kv_tokens()

            current_token_id = decode_output.next_token_id

    except Exception as exc:
        request.mark_failed(exc)

        logger.exception(
            "[generate-request] request_id=%s failed",
            request.request_id,
        )

        raise


def _build_generation_result(
    *,
    runner: PyTorchModelRunner,
    request: Request,
    finish_reason: str,
) -> GenerationResult:
    generated_ids = list(request.generated_ids)

    text = runner.decode_tokens(
        generated_ids,
        skip_special_tokens=True,
    )

    return GenerationResult(
        prompt_token_ids=list(request.input_ids),
        generated_token_ids=generated_ids,
        text=text,
        finish_reason=finish_reason,
    )