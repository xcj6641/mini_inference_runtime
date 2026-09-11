from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from app.runtime.cuda.paged_attention import paged_attention
from app.runtime.paged_kv_cache import PagedKVCache


SEQUENCE_LENGTHS = (128, 512, 1024, 2048, 4096)
WARMUP_ITERS = 5
BENCHMARK_ITERS = 20

DEVICE = torch.device("cuda")
DTYPE = torch.float16
NUM_LAYERS = 1
NUM_KV_HEADS = 2
HEAD_DIM = 64
BLOCK_SIZE = 16
LAYER_IDX = 0
EXTRA_PHYSICAL_BLOCKS = 4
RANDOM_SEED = 21

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "paged_attention_benchmark.log"
CSV_PATH = RESULTS_DIR / "paged_attention_benchmark.csv"


class PacificTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(
            record.created,
            tz=ZoneInfo("America/Los_Angeles"),
        )
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


def configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = PacificTimeFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = configure_logger()


def benchmark_cuda_operation(
    operation: Callable[[], Any],
) -> dict[str, float]:
    for _ in range(WARMUP_ITERS):
        operation()
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(BENCHMARK_ITERS):
        start.record()
        operation()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def append_csv_row(row: dict[str, object]) -> None:
    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_past_key_values(seq_len: int):
    layers = []
    for _ in range(NUM_LAYERS):
        key = torch.randn(
            1,
            NUM_KV_HEADS,
            seq_len,
            HEAD_DIM,
            dtype=DTYPE,
            device=DEVICE,
        )
        value = torch.randn_like(key)
        layers.append((key, value))
    return tuple(layers)


def make_fixture(seq_len: int):
    required_blocks = math.ceil(seq_len / BLOCK_SIZE)
    num_blocks = required_blocks + EXTRA_PHYSICAL_BLOCKS

    physical_blocks = list(range(num_blocks))
    random.Random(RANDOM_SEED + seq_len).shuffle(physical_blocks)
    block_table_list = physical_blocks[:required_blocks]
    block_table_tensor = torch.tensor(
        block_table_list,
        dtype=torch.int32,
        device=DEVICE,
    )

    cache = PagedKVCache(
        num_layers=NUM_LAYERS,
        num_blocks=num_blocks,
        num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    past_key_values = make_past_key_values(seq_len)
    cache.write_request_kv(
        block_table=block_table_list,
        past_key_values=past_key_values,
        num_tokens=seq_len,
    )
    query = torch.randn(
        1,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    torch.cuda.synchronize()
    return (
        cache,
        block_table_list,
        block_table_tensor,
        query,
    )


def contiguous_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query_4d = query.unsqueeze(2)
    scores = torch.matmul(
        query_4d,
        key.transpose(-2, -1),
    ) * (HEAD_DIM ** -0.5)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value).squeeze(2)


def check_correctness(
    *,
    seq_len: int,
    cache: PagedKVCache,
    block_table_list: list[int],
    block_table_tensor: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    materialized = cache.materialize_request_kv(
        block_table=block_table_list,
        num_tokens=seq_len,
    )
    key, value = materialized[LAYER_IDX]

    reference_output = contiguous_attention(query, key, value)
    cuda_output = paged_attention(
        query=query,
        key_cache=cache.key_cache,
        value_cache=cache.value_cache,
        block_table=block_table_tensor,
        num_tokens=seq_len,
        layer_idx=LAYER_IDX,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        cuda_output,
        reference_output.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    logger.info(
        "Correctness passed | seq_len=%d | max_abs_error=%.6f",
        seq_len,
        (cuda_output - reference_output.float()).abs().max().item(),
    )
    return key, value


def run_case(seq_len: int) -> dict[str, object]:
    (
        cache,
        block_table_list,
        block_table_tensor,
        query,
    ) = make_fixture(seq_len)

    key, value = check_correctness(
        seq_len=seq_len,
        cache=cache,
        block_table_list=block_table_list,
        block_table_tensor=block_table_tensor,
        query=query,
    )

    materialize_stats = benchmark_cuda_operation(
        lambda: cache.materialize_request_kv(
            block_table=block_table_list,
            num_tokens=seq_len,
        )
    )
    contiguous_stats = benchmark_cuda_operation(
        lambda: contiguous_attention(query, key, value)
    )

    def materialized_path():
        current = cache.materialize_request_kv(
            block_table=block_table_list,
            num_tokens=seq_len,
        )
        current_key, current_value = current[LAYER_IDX]
        return contiguous_attention(query, current_key, current_value)

    materialized_path_stats = benchmark_cuda_operation(materialized_path)
    direct_paged_stats = benchmark_cuda_operation(
        lambda: paged_attention(
            query=query,
            key_cache=cache.key_cache,
            value_cache=cache.value_cache,
            block_table=block_table_tensor,
            num_tokens=seq_len,
            layer_idx=LAYER_IDX,
        )
    )

    materialized_total_ms = materialized_path_stats["median_ms"]
    direct_ms = direct_paged_stats["median_ms"]
    speedup = materialized_total_ms / direct_ms
    
    component_sum_ms = (
        materialize_stats["median_ms"]
        + contiguous_stats["median_ms"]
    )

    materialization_percent = (
        materialize_stats["median_ms"]
        / component_sum_ms
        * 100
    )

    element_size = torch.tensor([], dtype=DTYPE).element_size()
    kv_size_mib = (
        seq_len
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * 2
        * element_size
        / (1024**2)
    )

    row: dict[str, object] = {
        "seq_len": seq_len,
        "kv_size_mib": kv_size_mib,
        "num_layers": NUM_LAYERS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "dtype": str(DTYPE),
        "warmup_iters": WARMUP_ITERS,
        "benchmark_iters": BENCHMARK_ITERS,
        "materialize_median_ms": materialize_stats["median_ms"],
        "contiguous_attention_median_ms": contiguous_stats["median_ms"],
        "component_sum_ms": (
            materialize_stats["median_ms"]
            + contiguous_stats["median_ms"]
        ),
        "materialized_total_median_ms": materialized_total_ms,
        "cuda_paged_attention_median_ms": direct_ms,
        "speedup_vs_materialized": speedup,
        "materialization_percent": materialization_percent,
        "materialize_mean_ms": materialize_stats["mean_ms"],
        "materialize_min_ms": materialize_stats["min_ms"],
        "materialize_max_ms": materialize_stats["max_ms"],
        "contiguous_attention_mean_ms": contiguous_stats["mean_ms"],
        "contiguous_attention_min_ms": contiguous_stats["min_ms"],
        "contiguous_attention_max_ms": contiguous_stats["max_ms"],
        "materialized_total_mean_ms": materialized_path_stats["mean_ms"],
        "materialized_total_min_ms": materialized_path_stats["min_ms"],
        "materialized_total_max_ms": materialized_path_stats["max_ms"],
        "cuda_paged_attention_mean_ms": direct_paged_stats["mean_ms"],
        "cuda_paged_attention_min_ms": direct_paged_stats["min_ms"],
        "cuda_paged_attention_max_ms": direct_paged_stats["max_ms"],
    }
    append_csv_row(row)

    comparison = (
        f"{speedup:.3f}x speedup"
        if speedup >= 1.0
        else f"{1.0 / speedup:.3f}x slowdown"
    )
    logger.info(
        "Result | seq_len=%d | materialize=%.4f ms | "
        "contiguous=%.4f ms | materialized_total=%.4f ms | "
        "direct_paged=%.4f ms | %s | materialization=%.2f%%",
        seq_len,
        materialize_stats["median_ms"],
        contiguous_stats["median_ms"],
        materialized_total_ms,
        direct_ms,
        comparison,
        materialization_percent,
    )
    return row


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    logger.info(
        "Starting Day 21 benchmark | sequence_lengths=%s",
        SEQUENCE_LENGTHS,
    )
    for seq_len in SEQUENCE_LENGTHS:
        run_case(seq_len)
    logger.info("Benchmark CSV written to %s", CSV_PATH)


if __name__ == "__main__":
    main()
