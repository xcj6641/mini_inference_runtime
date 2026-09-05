import csv
import logging
import math
import statistics
from pathlib import Path

import torch

from app.runtime.paged_kv_cache import PagedKVCache
from collections.abc import Callable
from typing import Any



WARMUP_ITERS = 5
BENCHMARK_ITERS = 20

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "materialize.log"
MATERIALIZE_CSV_PATH = RESULTS_DIR / "materialize.csv"

ATTENTION_CSV_PATH = (
    RESULTS_DIR / "contiguous_attention.csv"
)

from datetime import datetime
from zoneinfo import ZoneInfo


class PacificTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(
            record.created,
            tz=ZoneInfo("America/Los_Angeles"),
        )

        if datefmt:
            return dt.strftime(datefmt)

        return dt.isoformat()

formatter = PacificTimeFormatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)

file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


##### common
def benchmark_cuda_operation(
        operation: Callable[[], Any],
    ) -> list[float]:
    # Warmup
    for _ in range(WARMUP_ITERS):
        operation()

    torch.cuda.synchronize()

    latencies_ms = []

    for _ in range(BENCHMARK_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        operation()

        end.record()

        torch.cuda.synchronize()

        latencies_ms.append(
            start.elapsed_time(end)
        )

    return latencies_ms

def calculate_latency_stats(
        latencies_ms: list[float],
    ) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }

##### materialize
def make_fake_past_key_values(
        *,
        num_layers: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
    layers = []

    for _ in range(num_layers):
        key = torch.randn(
            1,
            num_kv_heads,
            seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        value = torch.randn_like(key)

        layers.append((key, value))

    return tuple(layers)

def calculate_kv_bytes(
        *,
        seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
    element_size = torch.tensor(
        [],
        dtype=dtype,
    ).element_size()

    return (
        seq_len
        * num_layers
        * num_kv_heads
        * head_dim
        * 2  # K + V
        * element_size
    )

def prepare_kv_cache(
        seq_len: int,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
    required_blocks = math.ceil(
        seq_len / block_size
    )

    block_table = list(range(required_blocks))

    paged_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_blocks=required_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    past_key_values = make_fake_past_key_values(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=past_key_values,
        num_tokens=seq_len,
    )

    materialized = (
        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=seq_len,
        )
    )

    torch.testing.assert_close(
        materialized[0][0],
        past_key_values[0][0],
    )

    torch.testing.assert_close(
        materialized[0][1],
        past_key_values[0][1],
    )

    torch.cuda.synchronize()

    logger.info(
        "Correctness check passed: seq_len=%d",
        seq_len,
    )

    return (
        paged_kv_cache,
        block_table,
        past_key_values,
        materialized,
    )

def run_materialize_benchmark(
        *,
        seq_len: int,
        paged_kv_cache: PagedKVCache,
        block_table: list[int],
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> dict[str, object]:

    def operation():
        return paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=seq_len,
        )

    latencies_ms = benchmark_cuda_operation(
        operation
    )

    stats = calculate_latency_stats(
        latencies_ms
    )

    bytes_copied = calculate_kv_bytes(
        seq_len=seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    kv_size_mib = bytes_copied / (1024 ** 2)

    effective_gbps = (
        bytes_copied
        / (stats["median_ms"] / 1000)
        / 1e9
    )

    return {
        "seq_len": seq_len,
        "kv_size_mib": kv_size_mib,
        "materialize_median_ms": stats["median_ms"],
        "effective_gbps": effective_gbps,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "dtype": str(dtype),
        "warmup_iters": WARMUP_ITERS,
        "benchmark_iters": BENCHMARK_ITERS,
        "mean_ms": stats["mean_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
    }

def run_contiguous_attention_benchmark(
        *,
        seq_len: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> dict[str, object]:

    def operation():
        return contiguous_attention(
            query=query,
            key=key,
            value=value,
        )

    latencies_ms = benchmark_cuda_operation(
        operation
    )

    stats = calculate_latency_stats(
        latencies_ms
    )

    return {
        "seq_len": seq_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "warmup_iters": WARMUP_ITERS,
        "benchmark_iters": BENCHMARK_ITERS,
        "median_ms": stats["median_ms"],
        "mean_ms": stats["mean_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
    }

##### attention
def contiguous_attention(
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5

    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) * scale

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    return torch.matmul(
        probabilities,
        value,
    )

def append_csv_row(
        *,
        path: Path,
        row: dict[str, object],
    ) -> None:
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def main() -> None:
    device = torch.device("cuda")
    dtype = torch.float16

    num_layers = 24
    num_kv_heads = 2
    head_dim = 64
    block_size = 16

    seq_lengths = [
        128,
        512,
        1024,
        2048,
        4096,
    ]

    for seq_len in seq_lengths:
        logger.info(
            "Preparing benchmark: seq_len=%d",
            seq_len,
        )

        (
            paged_kv_cache,
            block_table,
            past_key_values,
            materialized,
        ) = prepare_kv_cache(
            seq_len=seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        # --------------------------------
        # Materialization benchmark
        # --------------------------------

        # materialize_result = (
        #     run_materialize_benchmark(
        #         seq_len=seq_len,
        #         paged_kv_cache=paged_kv_cache,
        #         block_table=block_table,
        #         num_layers=num_layers,
        #         num_kv_heads=num_kv_heads,
        #         head_dim=head_dim,
        #         block_size=block_size,
        #         dtype=dtype,
        #     )
        # )

        # append_csv_row(
        #     path=MATERIALIZE_CSV_PATH,
        #     row=materialize_result,
        # )

        # logger.info(
        #     "Materialize: seq_len=%d, "
        #     "median=%.3f ms, "
        #     "bandwidth=%.4f GB/s",
        #     seq_len,
        #     materialize_result[
        #         "materialize_median_ms"
        #     ],
        #     materialize_result[
        #         "effective_gbps"
        #     ],
        # )

        # --------------------------------
        # Contiguous attention benchmark
        # --------------------------------

        key, value = materialized[0]

        query = torch.randn(
            1,
            num_kv_heads,
            1,
            head_dim,
            dtype=dtype,
            device=device,
        )

        attention_result = (
            run_contiguous_attention_benchmark(
                seq_len=seq_len,
                query=query,
                key=key,
                value=value,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
        )

        append_csv_row(
            path=ATTENTION_CSV_PATH,
            row=attention_result,
        )

        logger.info(
            "Contiguous attention: seq_len=%d, "
            "median=%.4f ms",
            seq_len,
            attention_result["median_ms"],
        )

    # logger.info(
    #     "Materialization results: %s",
    #     MATERIALIZE_CSV_PATH,
    # )

    logger.info(
        "Attention results: %s",
        ATTENTION_CSV_PATH,
    )


if __name__ == "__main__":
    main()