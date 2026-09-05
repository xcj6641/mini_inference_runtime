import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("benchmarks/results")

MATERIALIZE_CSV_PATH = (
    RESULTS_DIR / "3_materialize_plot.csv"
)

ATTENTION_CSV_PATH = (
    RESULTS_DIR / "5_contiguous_attention_all_layers.csv"
)

COMBINED_CSV_PATH = (
    RESULTS_DIR / "7_materialize_plus_attention.csv"
)

SUMMARY_CSV_PATH = (
    RESULTS_DIR / "materialization_summary.csv"
)

PLOT_PATH = (
    RESULTS_DIR / "materialization_latency.png"
)


logger = logging.getLogger(__name__)


def read_latest_by_seq_len(
    path: Path,
) -> dict[int, dict[str, str]]:
    """
    Read a benchmark CSV and keep the latest row
    for each sequence length.

    This is useful because our benchmark files
    contain results from multiple runs.
    """

    results = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            seq_len = int(row["seq_len"])

            # Later rows overwrite earlier runs.
            results[seq_len] = row

    return results


def build_summary():
    materialize = read_latest_by_seq_len(
        MATERIALIZE_CSV_PATH
    )

    attention = read_latest_by_seq_len(
        ATTENTION_CSV_PATH
    )

    combined = read_latest_by_seq_len(
        COMBINED_CSV_PATH
    )

    seq_lengths = sorted(
        set(materialize)
        & set(attention)
        & set(combined)
    )

    rows = []

    for seq_len in seq_lengths:
        materialize_row = materialize[seq_len]
        attention_row = attention[seq_len]
        combined_row = combined[seq_len]

        materialize_ms = float(
            materialize_row[
                "materialize_median_ms"
            ]
        )

        attention_ms = float(
            attention_row["median_ms"]
        )

        combined_ms = float(
            combined_row["median_ms"]
        )

        kv_size_mib = float(
            materialize_row["kv_size_mib"]
        )

        effective_gbps = float(
            materialize_row["effective_gbps"]
        )

        materialization_percent = (
            materialize_ms
            / combined_ms
            * 100
        )

        rows.append(
            {
                "seq_len": seq_len,
                "kv_size_mib": kv_size_mib,
                "materialize_ms": materialize_ms,
                "attention_ms": attention_ms,
                "materialize_plus_attention_ms":
                    combined_ms,
                "materialization_percent":
                    materialization_percent,
                "effective_gbps":
                    effective_gbps,
            }
        )

    return rows


def write_summary(
    rows: list[dict[str, float]],
) -> None:
    if not rows:
        raise RuntimeError(
            "No matching benchmark results found"
        )

    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Summary CSV written to: %s",
        SUMMARY_CSV_PATH,
    )


def create_plot(
    rows: list[dict[str, float]],
) -> None:
    seq_lengths = [
        row["seq_len"]
        for row in rows
    ]

    materialize_ms = [
        row["materialize_ms"]
        for row in rows
    ]

    combined_ms = [
        row["materialize_plus_attention_ms"]
        for row in rows
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        seq_lengths,
        materialize_ms,
        marker="o",
        label="Materialize only",
    )

    plt.plot(
        seq_lengths,
        combined_ms,
        marker="o",
        label="Materialize + attention",
    )

    plt.xlabel("Sequence length")
    plt.ylabel("Median latency (ms)")
    plt.title(
        "Paged KV materialization overhead"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=160,
    )

    plt.close()

    logger.info(
        "Plot written to: %s",
        PLOT_PATH,
    )


def log_summary(
    rows: list[dict[str, float]],
) -> None:
    logger.info(
        "%-8s %-8s %-14s %-14s %-14s %-10s",
        "seq_len",
        "KV MiB",
        "materialize",
        "attention",
        "combined",
        "mat_%",
    )

    for row in rows:
        logger.info(
            "%-8d %-8.1f %-14.2f %-14.3f "
            "%-14.2f %-10.2f",
            row["seq_len"],
            row["kv_size_mib"],
            row["materialize_ms"],
            row["attention_ms"],
            row[
                "materialize_plus_attention_ms"
            ],
            row["materialization_percent"],
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    rows = build_summary()

    log_summary(rows)
    write_summary(rows)
    create_plot(rows)


if __name__ == "__main__":
    main()