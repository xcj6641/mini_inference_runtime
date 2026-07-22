import gc
import logging
import statistics
import time
from pathlib import Path

import torch

from app.runtime.batch import DecodeBatch
from app.runtime.pytorch_model_runner import (
    PyTorchModelRunner,
)
from tests.runtime.test_batch_builder import (
    make_request,
)
from app.runtime.batch_builder import BatchBuilder
from app.runtime.kv_cache_utils import (
    calculate_actual_kv_bytes,
    get_kv_sequence_length,
)

from dataclasses import dataclass


def configure_logging() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "static_batch_benchmark.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )

    return log_file


@dataclass
class StaticBatchBenchmarkResult:
    dtype: str
    batch_size: int
    prompt_length: int
    decode_steps: int

    prefill_ms: float
    total_decode_ms: float
    average_decode_ms: float
    generated_tokens: int
    tokens_per_second: float

    memory_after_model_load_mb: float
    baseline_allocated_mb: float
    peak_allocated_mb: float
    incremental_over_model_load_mb: float
    incremental_over_baseline_mb: float

    kv_bytes: int
    final_kv_length: int
    first_next_token_ids: list[int]
    all_logits_finite: bool


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

def run_single_benchmark(
    runner: PyTorchModelRunner,
    *,
    dtype_name: str,
    batch_size: int,
    prompt_length: int,
    decode_steps: int,
    memory_after_model_load_mb: float,
):
    # clean up objects from the previous batch-size run before reading the baseline
    gc.collect()
    torch.cuda.empty_cache()

    prompt_token_ids = build_prompt_tokens(
        runner=runner,
        prompt_length=prompt_length,
    )

    requests = [
        make_request(
            request_id=f"benchmark-{index}",
            input_ids=prompt_token_ids.copy(),
        )
        for index in range(batch_size)
    ]

    # build batch
    batch_builder = BatchBuilder()
    batch = batch_builder.build_equal_length_prefill_batch(
        requests
    )

    assert len(batch.request_ids) == batch_size
    assert len(set(batch.request_ids)) == batch_size

    assert batch.input_ids.shape == (
        batch_size,
        prompt_length,
    )

    assert batch.attention_mask.shape == (
        batch_size,
        prompt_length,
    )

    # Warm-up: do not measure this operation.
    # Warm-up runs the workload before timing it 
    # so that one-time initialization costs are excluded from the measured benchmark.
    warm_up(runner, batch)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


    baseline_allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
    )
    torch.cuda.reset_peak_memory_stats()

    # Measured prefill.
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        batched_prefill_output = runner.prefill_batch(batch)

    torch.cuda.synchronize()
    assert len(
        batched_prefill_output.next_token_ids
    ) == batch_size

    prefill_kv_length = get_kv_sequence_length(
        batched_prefill_output.past_key_values
    )

    prefill_ms = (
        time.perf_counter() - start
    ) * 1000

    prefill_logits_finite = bool(
        torch.isfinite(
            batched_prefill_output.logits
        ).all().item()
    )

    # decode
    torch.cuda.synchronize()
    decode_start = time.perf_counter()

    with torch.inference_mode():
        (
            final_decode_output,
            final_past_key_values,
            decode_logits_finite_tensor,
        ) = run_decode_steps(
            runner,
            first_token_ids=(
                batched_prefill_output.next_token_ids
            ),
            past_key_values=(
                batched_prefill_output.past_key_values
            ),
            attention_mask=batch.attention_mask,
            request_ids=batch.request_ids,
            decode_steps=decode_steps,
        )

    torch.cuda.synchronize()
    # mesure total decode time
    total_decode_ms = (
        time.perf_counter() - decode_start
    ) * 1000

    average_decode_ms = (
        total_decode_ms / decode_steps
    )
    
    decode_logits_finite = bool(
        decode_logits_finite_tensor.item()
    )

    all_logits_finite = (
        prefill_logits_finite
        and decode_logits_finite
    )

    assert len(
        final_decode_output.next_token_ids
    ) == batch_size

    # batch_size * decode_steps = total generated tokens
    generated_tokens = (
        batch.input_ids.shape[0]
        * decode_steps
    )
    
    tokens_per_second = (
        generated_tokens
        / (total_decode_ms / 1000)
    )

    # mesure KV cache size
    kv_bytes = calculate_actual_kv_bytes(
        final_past_key_values
    )
    kv_mb = kv_bytes / 1024**2

    torch.cuda.synchronize()

    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated()
    )

    peak_allocated_mb = (
        peak_allocated_bytes
        / 1024**2
    )
    incremental_over_model_load_mb = (
        peak_allocated_mb
        - memory_after_model_load_mb
    )

    incremental_over_baseline_mb = (
        peak_allocated_mb
        - baseline_allocated_mb
    )
    # measure final KV cache length
    final_kv_length = get_kv_sequence_length(
        final_past_key_values
    )

    assert prefill_kv_length == prompt_length
    assert final_kv_length == prompt_length + decode_steps

    logger = logging.getLogger(__name__)
    logger.info("Benchmark result")
    logger.info("----------------")
    logger.info("dtype: %s", dtype_name)
    logger.info("batch_size: %s", batch_size)
    logger.info("prompt_length: %s", prompt_length)
    logger.info("prefill_ms: %.3f", prefill_ms)
    logger.info("prefill_kv_length: %s", prefill_kv_length)
    logger.info(
        "first_next_token_ids: %s",
        batched_prefill_output.next_token_ids,
    )
    logger.info("all_logits_finite: %s", all_logits_finite)
    logger.info(
        "final_decode_output next_token_ids: %s",
        final_decode_output.next_token_ids,
    )
    logger.info("prefill_logits_finite: %s", prefill_logits_finite)
    logger.info("decode_logits_finite: %s", decode_logits_finite)
    logger.info("decode_steps: %s", decode_steps)
    logger.info("total_decode_ms: %.3f", total_decode_ms)
    logger.info("average_decode_ms: %.3f", average_decode_ms)
    logger.info("generated_tokens: %s", generated_tokens)
    logger.info("tokens_per_second: %.3f", tokens_per_second)
    logger.info("final_kv_length: %s", final_kv_length)
    logger.info(
        "memory_after_model_load_mib: %.3f",
        memory_after_model_load_mb,
    )
    logger.info("baseline_allocated_mib: %.3f", baseline_allocated_mb)
    logger.info("peak_allocated_mib: %.3f", peak_allocated_mb)
    logger.info("incremental_over_model_load_mib: %.3f", incremental_over_model_load_mb)
    logger.info("incremental_over_baseline_mib: %.3f", incremental_over_baseline_mb)
    logger.info("kv_bytes: %s", kv_bytes)
    logger.info("kv_mib: %.3f", kv_bytes / 1024**2)

    return StaticBatchBenchmarkResult(
        dtype=dtype_name,
        batch_size=batch_size,
        prompt_length=prompt_length,
        decode_steps=decode_steps,
        prefill_ms=prefill_ms,
        average_decode_ms=average_decode_ms,
        generated_tokens=generated_tokens,
        total_decode_ms=total_decode_ms,
        tokens_per_second=tokens_per_second,
        memory_after_model_load_mb=(
            memory_after_model_load_mb
        ),
        baseline_allocated_mb=baseline_allocated_mb,
        peak_allocated_mb=peak_allocated_mb,
        incremental_over_model_load_mb=incremental_over_model_load_mb,
        incremental_over_baseline_mb=incremental_over_baseline_mb,
        kv_bytes=kv_bytes,
        final_kv_length=final_kv_length,
        first_next_token_ids=(
            batched_prefill_output.next_token_ids
        ),
        all_logits_finite=all_logits_finite,
    )


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_LENGTH = 64
DECODE_STEPS = 32

REPETITIONS = 5

if __name__ == "__main__":
    log_file = configure_logging()

    results = []
    median_results: dict[tuple[str, int], dict[str, float]] = {}
    DTYPE_CONFIGS = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]


    for dtype_name, dtype in DTYPE_CONFIGS:

        try:
            # Run all batch sizes and repetitions.
            ...
            runner = PyTorchModelRunner(
                model_name=MODEL_NAME,
                device="cuda",
                dtype=dtype,
            )

            torch.cuda.synchronize()

            memory_after_model_load_bytes = (
                torch.cuda.memory_allocated()
            )

            memory_after_model_load_mb = (
                memory_after_model_load_bytes
                / 1024**2
            )
            
            for batch_size in [1, 2, 4]:

                logging.info("")
                logging.info("-" + "=" * 60)
                logging.info(
                    "Benchmark: %s, batch_size=%s",
                    dtype_name,
                    batch_size,
                )
                logging.info("-" + "=" * 60)
                logging.info("")

                prefill_samples_ms = []
                decode_samples_ms = []
                tokens_per_second_samples = []
                peak_memory_mb_samples = []
                kv_memory_mb_samples = []
                for _ in range(REPETITIONS):
                    result = run_single_benchmark(
                        runner,
                        dtype_name=dtype_name,
                        batch_size=batch_size,
                        prompt_length=PROMPT_LENGTH,
                        decode_steps=DECODE_STEPS,
                        memory_after_model_load_mb=(
                            memory_after_model_load_mb
                        ),
                    )
                    results.append(result)
                    prefill_samples_ms.append(result.prefill_ms)
                    decode_samples_ms.append(result.average_decode_ms)
                    tokens_per_second_samples.append(result.tokens_per_second)
                    peak_memory_mb_samples.append(result.peak_allocated_mb)
                    kv_memory_mb_samples.append(
                        result.kv_bytes / 1024**2
                    )
                
                median_prefill_ms = statistics.median(
                    prefill_samples_ms
                )

                median_decode_ms = statistics.median(
                    decode_samples_ms
                )

                median_tok_per_s = statistics.median(
                    tokens_per_second_samples
                )
                median_peak_memory_mb = statistics.median(
                    peak_memory_mb_samples
                )
                median_kv_memory_mb = statistics.median(
                    kv_memory_mb_samples
                )

                median_results[(dtype_name, batch_size)] = {
                    "prefill_ms": median_prefill_ms,
                    "decode_ms": median_decode_ms,
                    "median_tok_per_s": median_tok_per_s,
                    "peak_memory_mb": median_peak_memory_mb,
                    "kv_memory_mb": median_kv_memory_mb,
                }

                logging.info(
                    "Median prefill_ms for %s, batch_size=%s: %.3f",
                    dtype_name,
                    batch_size,
                    median_prefill_ms,
                )
                logging.info(
                    "Median decode_ms for %s, batch_size=%s: %.3f",
                    dtype_name,
                    batch_size,
                    median_decode_ms,
                )
                logging.info(
                    "Median tok/s for %s, batch_size=%s: %.3f",
                    dtype_name,
                    batch_size,
                    median_tok_per_s,
                )
                logging.info(
                    "Median peak memory for %s, batch_size=%s: %.3f MiB",
                    dtype_name,
                    batch_size,
                    median_peak_memory_mb,
                )
                logging.info(
                    "Median KV memory for %s, batch_size=%s: %.3f MiB",
                    dtype_name,
                    batch_size,
                    median_kv_memory_mb,
                )
                logging.info("median_results: %s", median_results)
        finally:
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    logging.info("Benchmark log saved to: %s", log_file.resolve())
