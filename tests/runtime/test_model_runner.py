from typing import Any
import pytest
import torch
import time

from app.runtime.pytorch_model_runner import PyTorchModelRunner
from app.runtime.generation import generate
from app.runtime.kv_cache_utils import get_cache_length

def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

@pytest.mark.integration
def test_real_prefill_returns_logits(real_runner) -> None:
    input_ids = real_runner.encode_prompt(
        "The capital of France is"
    )

    synchronize(real_runner.device)
    start = time.perf_counter()
    output = real_runner.prefill(input_ids)
    synchronize(real_runner.device)
    prefill_ms = (time.perf_counter() - start) * 1000

    assert output.logits.ndim == 3
    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] == input_ids.shape[1]
    assert output.logits.shape[2] > 1

    assert isinstance(output.next_token_id, int)
    assert output.past_key_values is not None



# def get_cache_length(cache) -> int:
#     if hasattr(cache, "get_seq_length"):
#         return int(cache.get_seq_length())

#     return int(cache[0][0].shape[-2])

@pytest.mark.integration
def test_decode_adds_one_token(real_runner) -> None:
    input_ids = real_runner.encode_prompt(
        "The capital of France is"
    )


    synchronize(real_runner.device)
    start = time.perf_counter()
    prefill_output = real_runner.prefill(input_ids)    
    synchronize(real_runner.device)
    prefill_ms = (time.perf_counter() - start) * 1000

    prefill_cache_length = get_cache_length(
        prefill_output.past_key_values
    )

    decode_input_ids = torch.tensor(
        [[prefill_output.next_token_id]],
        dtype=torch.long,
        device=real_runner.device,
    )

    decode_output = real_runner.decode(
        input_ids=decode_input_ids,
        past_key_values=prefill_output.past_key_values,
    )

    decode_cache_length = get_cache_length(
        decode_output.past_key_values
    )

    assert decode_output.logits.shape[0] == 1
    assert decode_output.logits.shape[1] == 1

    assert decode_cache_length == prefill_cache_length + 1


@pytest.mark.integration
def test_greedy_output_is_deterministic(real_runner) -> None:
    input_ids = real_runner.encode_prompt(
        "The capital of France is"
    )


    first = generate(
        runner=real_runner,
        input_ids=input_ids,
        max_new_tokens=8,
    )

    second = generate(
        runner=real_runner,
        input_ids=input_ids,
        max_new_tokens=8,
    )

    assert (
        first.generated_token_ids
        == second.generated_token_ids
    )

    assert first.text == second.text

@pytest.mark.integration
def test_stop_on_max_new_tokens(real_runner) -> None:
    input_ids = real_runner.encode_prompt(
        "Continue counting: one, two, three,"
    )


    result = generate(
        runner=real_runner,
        input_ids=input_ids,
        max_new_tokens=3,
    )

    assert len(result.generated_token_ids) == 3
    assert result.finish_reason == "length"

@pytest.mark.integration
def test_real_runner_uses_cuda(
    real_runner: PyTorchModelRunner,
) -> None:
    assert real_runner.device.type == "cuda"

@pytest.mark.integration
def test_cache_grows_across_multiple_decode_steps(
    real_runner: PyTorchModelRunner,
) -> None:
    input_ids = real_runner.encode_prompt(
        "The capital of France is"
    )

    synchronize(real_runner.device)
    start = time.perf_counter()
    output = real_runner.prefill(input_ids)
    synchronize(real_runner.device)
    prefill_ms = (time.perf_counter() - start) * 1000

    cache = output.past_key_values
    current_token_id = output.next_token_id

    initial_cache_length = get_cache_length(cache)

    for step in range(3):
        decode_input_ids = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=real_runner.device,
        )

        output = real_runner.decode(
            input_ids=decode_input_ids,
            past_key_values=cache,
        )

        cache = output.past_key_values
        current_token_id = output.next_token_id

        assert get_cache_length(cache) == initial_cache_length + step + 1