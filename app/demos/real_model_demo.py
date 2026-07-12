import logging
import time
from pathlib import Path

import torch

from app.runtime.pytorch_model_runner import PyTorchModelRunner


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def log_gpu_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return

    allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    reserved_mb = torch.cuda.memory_reserved(device) / 1024**2

    print(f"{prefix} GPU allocated: {allocated_mb:.2f} MB")
    print(f"{prefix} GPU reserved: {reserved_mb:.2f} MB")


def configure_logging() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "real_model_demo.log"

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
        handlers=[
            console_handler,
            file_handler,
        ],
        force=True,
    )

    return log_file


def main() -> None:
    log_file = configure_logging()

    runner = PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
    log_gpu_memory("after model load", runner.device)

    input_ids = runner.encode_chat_prompt(
        user_message="Explain KV cache in one short sentence.",
    )

    synchronize(runner.device)
    start = time.perf_counter()
    prefill_output = runner.prefill(input_ids)
    synchronize(runner.device)
    prefill_ms = (time.perf_counter() - start) * 1000
    log_gpu_memory("after prefill", runner.device)

    current_token_id = prefill_output.next_token_id
    past_key_values = prefill_output.past_key_values
    generated_token_ids: list[int] = []
    decode_latencies_ms: list[float] = []
    finish_reason = "length"

    for step in range(32):
        generated_token_ids.append(current_token_id)

        if current_token_id in runner.eos_token_ids:
            finish_reason = "eos"
            break

        if len(generated_token_ids) >= 32:
            finish_reason = "length"
            break

        decode_input_ids = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=runner.device,
        )

        synchronize(runner.device)
        start = time.perf_counter()
        decode_output = runner.decode(
            input_ids=decode_input_ids,
            past_key_values=past_key_values,
        )
        synchronize(runner.device)
        decode_latency_ms = (time.perf_counter() - start) * 1000
        decode_latencies_ms.append(decode_latency_ms)

        current_token_id = decode_output.next_token_id
        past_key_values = decode_output.past_key_values

    generated_text = runner.decode_tokens(
        generated_token_ids,
        skip_special_tokens=True,
    )
    average_decode_ms = (
        sum(decode_latencies_ms) / len(decode_latencies_ms)
        if decode_latencies_ms
        else 0.0
    )

    latency_difference_ms = prefill_ms - average_decode_ms
    log_gpu_memory("after generation", runner.device)

    logging.info(
        "[result] finish_reason=%s generated_tokens=%d text=%r",
        finish_reason,
        len(generated_token_ids),
        generated_text,
    )

    print(f"Prefill latency: {prefill_ms:.1f} ms")
    print(f"Average decode latency: {average_decode_ms:.1f} ms/token")
    print(f"Prefill vs decode difference: {latency_difference_ms:.1f} ms")
    print(f"\nGenerated text: {generated_text}")
    print(f"Log saved to: {log_file.resolve()}")


if __name__ == "__main__":
    main()