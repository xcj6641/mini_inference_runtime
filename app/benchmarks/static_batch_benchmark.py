import time
from app.runtime.batch import DecodeBatch
from app.runtime.kv_cache_utils import get_kv_sequence_length, split_batched_legacy_kv_cache
import torch

from app.runtime.pytorch_model_runner import (
    PyTorchModelRunner,
)
from tests.runtime.test_batch_builder import (
    make_request,
)
from app.runtime.batch_builder import BatchBuilder

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_LENGTH = 64

def build_prompt_tokens(
    runner: PyTorchModelRunner,
    prompt_length: int,
) -> list[int]:
    text = (
        "Explain how a language model inference runtime "
        "processes an incoming request. "
    )

    repeated_text = text

    while True:
        input_ids = runner.encode_prompt(
            repeated_text
        )

        token_ids = (
            input_ids
            .squeeze(0)
            .tolist()
        )

        if len(token_ids) >= prompt_length:
            return token_ids[:prompt_length]

        repeated_text += text

def warm_up(
    runner: PyTorchModelRunner,
    batch,
) -> None:
    with torch.inference_mode():
        warmup_prefill = runner.prefill_batch(batch)

        warmup_input_ids = torch.tensor(
            warmup_prefill.next_token_ids,
            dtype=torch.long,
            device="cuda",
        ).unsqueeze(1)

        warmup_attention_mask = torch.cat(
            [
                batch.attention_mask.to("cuda"),
                torch.ones(
                    (batch.input_ids.shape[0], 1),
                    dtype=batch.attention_mask.dtype,
                    device="cuda",
                ),
            ],
            dim=1,
        )

        warmup_decode_batch = DecodeBatch(
            request_ids=batch.request_ids,
            input_ids=warmup_input_ids,
            past_key_values=(
                warmup_prefill.past_key_values
            ),
            attention_mask=warmup_attention_mask,
        )

        runner.decode_batch(warmup_decode_batch)

    torch.cuda.synchronize()

def run_decode_steps(
    runner: PyTorchModelRunner,
    *,
    first_token_ids: list[int],
    past_key_values: object,
    attention_mask: torch.Tensor,
    request_ids: list[str],
    decode_steps: int,
):
    next_token_ids = torch.tensor(
        first_token_ids,
        dtype=torch.long,
        device="cuda",
    ).unsqueeze(1)

    current_past_key_values = past_key_values
    current_attention_mask = attention_mask.to(
        "cuda"
    )

    all_logits_finite = torch.tensor(
        True,
        device="cuda",
    )

    final_output = None

    for _ in range(decode_steps):
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (
                        current_attention_mask.shape[0],
                        1,
                    ),
                    dtype=(
                        current_attention_mask.dtype
                    ),
                    device="cuda",
                ),
            ],
            dim=1,
        )

        decode_batch = DecodeBatch(
            request_ids=request_ids,
            input_ids=next_token_ids,
            past_key_values=(
                current_past_key_values
            ),
            attention_mask=(
                current_attention_mask
            ),
        )

        final_output = runner.decode_batch(
            decode_batch
        )

        all_logits_finite = (
            all_logits_finite
            & torch.isfinite(
                final_output.logits
            ).all()
        )

        next_token_ids = torch.tensor(
            final_output.next_token_ids,
            dtype=torch.long,
            device="cuda",
        ).unsqueeze(1)

        current_past_key_values = (
            final_output.past_key_values
        )

    assert final_output is not None

    return (
        final_output,
        current_past_key_values,
        all_logits_finite,
    )

def main() -> None:
    runner = PyTorchModelRunner(
        model_name=MODEL_NAME,
        device="cuda",
        dtype=torch.float32,
    )

    prompt_token_ids = build_prompt_tokens(
        runner=runner,
        prompt_length=PROMPT_LENGTH,
    )

    request = make_request(
        request_id="benchmark-0",
        input_ids=prompt_token_ids,
    )

    # build batch
    batch_builder = BatchBuilder()
    batch = batch_builder.build_equal_length_prefill_batch(
        [request]
    )

    assert batch.input_ids.shape == (1, 64)
    assert batch.attention_mask.shape == (1, 64)

    # Warm-up: do not measure this operation.
    warm_up(runner, batch)

    # Measured prefill.
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        batched_prefill_output = runner.prefill_batch(batch)

    torch.cuda.synchronize()

    prefill_kv_length = get_kv_sequence_length(
        batched_prefill_output.past_key_values
    )

    prefill_ms = (
        time.perf_counter() - start
    ) * 1000

    all_logits_finite = bool(
        torch.isfinite(
            batched_prefill_output.logits
        ).all().item()
    )

    # decode
    decode_input_ids = torch.tensor(
        batched_prefill_output.next_token_ids,
        dtype=torch.long,
        device="cuda",
    ).unsqueeze(1)

    assert decode_input_ids.shape == (1, 1)
    
    decode_attention_mask = torch.cat(
        [
            batch.attention_mask.to("cuda"),
            torch.ones(
                (1, 1),
                dtype=batch.attention_mask.dtype,
                device="cuda",
            ),
        ],
        dim=1,
    )
    assert decode_attention_mask.shape == (1, 65)


    decode_batch = DecodeBatch(
        request_ids=["benchmark-0"],
        input_ids=decode_input_ids,
        past_key_values=batched_prefill_output.past_key_values,
        attention_mask=decode_attention_mask,
    )

    torch.cuda.synchronize()
    decode_start = time.perf_counter()

    with torch.inference_mode():
        batched_decode_output = runner.decode_batch(
            decode_batch,
        )

    torch.cuda.synchronize()

    decode_kv_length = get_kv_sequence_length(
        batched_decode_output.past_key_values
    )

    assert prefill_kv_length == PROMPT_LENGTH
    assert decode_kv_length == PROMPT_LENGTH + 1
    decode_ms = (
        time.perf_counter() - decode_start
    ) * 1000


    print("Benchmark result")
    print("----------------")
    print("dtype: fp32")
    print("batch_size: 1")
    print(f"prompt_length: {PROMPT_LENGTH}")
    print(f"prefill_ms: {prefill_ms:.3f}")
    print(
        "first_next_token_ids:",
        batched_prefill_output.next_token_ids,
    )
    print(
        f"all_logits_finite: "
        f"{all_logits_finite}"
    )
    print(f"first_decode_ms: {decode_ms:.3f}")
    print(
        "decode_next_token_ids:",
        batched_decode_output.next_token_ids,
    )

    print(
        "decode_logits_finite:",
        bool(
            torch.isfinite(
                batched_decode_output.logits
            ).all().item()
        ),
    )
    print(f"prefill_kv_length: {prefill_kv_length}")
    print(f"decode_kv_length: {decode_kv_length}")

if __name__ == "__main__":
    main()