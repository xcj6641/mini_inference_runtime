import csv
import logging
import math
import statistics
from pathlib import Path

import torch

from app.runtime.paged_kv_cache import PagedKVCache


WARMUP_ITERS = 10
BENCHMARK_ITERS = 50

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "materialize.log"
CSV_PATH = RESULTS_DIR / "materialize.csv"


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
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

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

def benchmark_materialize(
        *,
        paged_kv_cache: PagedKVCache,
        block_table: list[int],
        seq_len: int,
    ) -> list[float]:
    # Warmup
    for _ in range(WARMUP_ITERS):
        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=seq_len,
        )

    torch.cuda.synchronize()

    latencies_ms = []

    for _ in range(BENCHMARK_ITERS):
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        paged_kv_cache.materialize_request_kv(
            block_table=block_table,
            num_tokens=seq_len,
        )

        end.record()

        torch.cuda.synchronize()

        latencies_ms.append(
            start.elapsed_time(end)
        )

    return latencies_ms


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

import logging
import math

import torch


logger = logging.getLogger(__name__)


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.float16

    seq_len = 128

    # Representative values for Qwen2.5-0.5B.
    num_layers = 24
    num_kv_heads = 2
    head_dim = 64

    # Keep this equal to runtime configuration.
    block_size = 16

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

    logger.info("-- Materialization correctness check")
    logger.info("seq_len: %d", seq_len)
    logger.info(
        "required_blocks: %d",
        required_blocks,
    )
    logger.info(
        "block_table: %s",
        block_table,
    )
    logger.info(
        "original K shape: %s",
        past_key_values[0][0].shape,
    )
    logger.info(
        "materialized K shape: %s",
        materialized[0][0].shape,
    )
    logger.info(
        "materialized V shape: %s",
        materialized[0][1].shape,
    )

    torch.testing.assert_close(
        materialized[0][0],
        past_key_values[0][0],
    )

    torch.testing.assert_close(
        materialized[0][1],
        past_key_values[0][1],
    )

    logger.info(
        "Materialization correctness check passed."
    )

    latencies_ms = benchmark_materialize(
        paged_kv_cache=paged_kv_cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

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
        / (median_ms / 1000)
        / 1e9
    )

    logger.info("-- Materialization benchmark")
    logger.info(
        "warmup iterations: %d",
        WARMUP_ITERS,
    )
    logger.info(
        "benchmark iterations: %d",
        BENCHMARK_ITERS,
    )
    logger.info(
        "mean latency: %.3f ms",
        mean_ms,
    )
    logger.info(
        "median latency: %.3f ms",
        median_ms,
    )
    logger.info(
        "min latency: %.3f ms",
        min_ms,
    )
    logger.info(
        "max latency: %.3f ms",
        max_ms,
    )
    logger.info("KV size: %.3f MiB", kv_size_mib)
    logger.info("Effective bandwidth: %.3f GB/s", effective_gbps)

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "seq_len",
                    "kv_size_mib",
                    "materialize_median_ms",
                    "effective_gbps",
                    "num_layers",
                    "num_kv_heads",
                    "head_dim",
                    "block_size",
                    "dtype",
                    "warmup_iters",
                    "benchmark_iters",
                    "mean_ms",
                    "min_ms",
                    "max_ms",
                ]
            )

        writer.writerow(
            [
                seq_len,
                kv_size_mib,
                median_ms,
                effective_gbps,
                num_layers,
                num_kv_heads,
                head_dim,
                block_size,
                str(dtype),
                WARMUP_ITERS,
                BENCHMARK_ITERS,
                mean_ms,
                min_ms,
                max_ms,
            ]
        )

    logger.info(
        "Benchmark result logged to CSV: %s",
        CSV_PATH,
    )


if __name__ == "__main__":
    main()