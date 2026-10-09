"""Plot publication-ready resource and retrieval-scaling figures.

The input contract is the pair of CSV files written by
``measure_efficiency_resources.py``.  The script intentionally performs no
benchmarking and changes no experiment implementation.  It validates that all
rows are complete measurements from one benchmark id and one protocol before
creating figures and traceable source-data tables.

Figure contract
---------------
Core conclusion:
    Retrieval has a measurable resource trade-off whose K/M scaling can be
    inspected independently from accuracy.
Archetype:
    Quantitative grid (six resource panels) plus a three-panel scaling figure.
Evidence hierarchy:
    End-to-end latency and throughput are primary; parameters/training time are
    deployment context; CUDA peak and memory payload localise the overhead.
Export contract:
    Editable-text SVG is primary; PDF and 600-dpi PNG are secondary.  Exact
    plotted values are copied to source-data CSV files.
Reviewer risk:
    These are descriptive measurements on one stack.  Training times come from
    completed experiment rows, while inference uses deterministic random
    weights solely because tensor shapes, retrieval, and allocation determine
    the reported resource quantities.  No accuracy claim is made here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Mandatory publication/export settings: SVG labels remain editable text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"


CONDITIONS = ("Native", "LoRA-Only", "SRF-Only", "SRF+LoRA")
EXPECTED_PARAMETERS = {
    "Native": (197448, 0),
    # True weight-level LoRA injects low-rank factors into every selected
    # architecture-specific projection.  These counts supersede the legacy
    # final-latent adapter (1,025 deployed / 1,545 trainable parameters).
    "LoRA-Only": (226760, 29832),
    "SRF-Only": (212147, 15219),
    # Joint is trained in two stages.  The reported adaptation-trainable count
    # is the final (LoRA-only) stage, while deployed parameters include SRF.
    "SRF+LoRA": (241459, 29312),
}
EXPECTED_TOP_K = (1, 3, 5, 10)
EXPECTED_MEMORY = (64, 128, 256, 512)
EXPECTED_PAYLOAD_BYTES = {64: 202800, 128: 401200, 256: 798000, 512: 1591600}
CONDITION_SHORT = {
    "Native": "Native",
    "LoRA-Only": "LoRA",
    "SRF-Only": "SRF",
    "SRF+LoRA": "Joint",
}
CONDITION_COLORS = {
    "Native": "#484878",
    "LoRA-Only": "#B4C0E4",
    "SRF-Only": "#E4CCD8",
    "SRF+LoRA": "#F0AFC0",
}
CONDITION_HATCHES = {
    "Native": "",
    "LoRA-Only": "//",
    "SRF-Only": "..",
    "SRF+LoRA": "xx",
}
LINE_COLORS = ("#484878", "#7884B4", "#B05B82", "#0F7C83", "#A77A22")
LINE_MARKERS = ("o", "s", "^", "D", "P")

RESOURCE_REQUIRED = {
    "benchmark_id",
    "measurement_status",
    "dataset",
    "backbone",
    "condition",
    "seed",
    "device",
    "gpu",
    "context_length",
    "forecast_horizon",
    "channels",
    "deployed_parameters",
    "adaptation_trainable_parameters",
    "psm_total_training_seconds_observed",
    "single_e2e_latency_ms_median",
    "batch_e2e_windows_per_second",
    "cuda_training_peak_allocated_mib_batch",
    "memory_bank_payload_bytes",
    "batch_size_for_throughput",
    "microbenchmark_weights",
    "timing_scope",
}

SENSITIVITY_REQUIRED = {
    "benchmark_id",
    "measurement_status",
    "dataset",
    "backbone",
    "condition",
    "seed",
    "device",
    "context_length",
    "forecast_horizon",
    "channels",
    "retrieval_top_k",
    "memory_per_channel",
    "memory_bank_payload_bytes",
    "single_retrieval_latency_ms_median",
    "single_e2e_latency_ms_median",
}

RESOURCE_SOURCE_COLUMNS = (
    "benchmark_id",
    "measurement_status",
    "dataset",
    "backbone",
    "seed",
    "device",
    "gpu",
    "context_length",
    "forecast_horizon",
    "channels",
    "condition",
    "deployed_parameters",
    "adaptation_trainable_parameters",
    "deployed_parameters_million",
    "psm_total_training_seconds_observed",
    "single_e2e_latency_ms_median",
    "batch_e2e_windows_per_second",
    "cuda_training_peak_allocated_mib_batch",
    "memory_bank_payload_bytes",
    "memory_bank_payload_mib",
    "batch_size_for_throughput",
    "microbenchmark_weights",
    "timing_scope",
)

SCALING_SOURCE_COLUMNS = (
    "benchmark_id",
    "measurement_status",
    "dataset",
    "backbone",
    "seed",
    "device",
    "context_length",
    "forecast_horizon",
    "channels",
    "condition",
    "retrieval_top_k",
    "memory_per_channel",
    "single_retrieval_latency_ms_median",
    "single_e2e_latency_ms_median",
    "memory_bank_payload_bytes",
    "memory_bank_payload_mib",
)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def _single(rows: Sequence[Mapping[str, str]], key: str, label: str) -> str:
    values = {str(row.get(key, "")).strip() for row in rows}
    if len(values) != 1 or "" in values:
        raise ValueError(f"{label}: expected one non-empty {key}, got {sorted(values)}")
    return next(iter(values))


def _number(row: Mapping[str, str], key: str, label: str, *, positive: bool = False) -> float:
    raw = str(row.get(key, "")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label}: {key} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: {key} must be finite, got {raw!r}")
    if value < 0 or (positive and value <= 0):
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{label}: {key} must be {relation}, got {value}")
    return value


def _integer(row: Mapping[str, str], key: str, label: str, *, positive: bool = False) -> int:
    value = _number(row, key, label, positive=positive)
    rounded = int(round(value))
    if not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(f"{label}: {key} must be integral, got {value}")
    return rounded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_identity(
    resource_rows: Sequence[Mapping[str, str]],
    scaling_rows: Sequence[Mapping[str, str]],
    *,
    expected_status: str,
    expected_dataset: str,
    expected_backbone: str,
    expected_seed: int,
    expected_context_length: int,
    expected_forecast_horizon: int,
    expected_channels: int,
) -> dict[str, str]:
    combined = list(resource_rows) + list(scaling_rows)
    identity = {
        "benchmark_id": _single(combined, "benchmark_id", "combined inputs"),
        "measurement_status": _single(combined, "measurement_status", "combined inputs"),
        "dataset": _single(combined, "dataset", "combined inputs"),
        "backbone": _single(combined, "backbone", "combined inputs"),
        "seed": _single(combined, "seed", "combined inputs"),
        "device": _single(combined, "device", "combined inputs"),
        "context_length": _single(combined, "context_length", "combined inputs"),
        "forecast_horizon": _single(combined, "forecast_horizon", "combined inputs"),
        "channels": _single(combined, "channels", "combined inputs"),
    }
    if identity["measurement_status"] != expected_status:
        raise ValueError(
            "measurement_status mismatch: "
            f"required {expected_status!r}, got {identity['measurement_status']!r}; "
            "static_only or partially measured rows cannot be plotted"
        )
    if identity["dataset"] != expected_dataset:
        raise ValueError(f"expected dataset {expected_dataset!r}, got {identity['dataset']!r}")
    if identity["backbone"] != expected_backbone:
        raise ValueError(f"expected backbone {expected_backbone!r}, got {identity['backbone']!r}")
    try:
        seed = int(identity["seed"])
    except ValueError as exc:
        raise ValueError(f"seed is not an integer: {identity['seed']!r}") from exc
    if seed != expected_seed:
        raise ValueError(f"expected seed {expected_seed}, got {seed}")
    for key, expected in (
        ("context_length", expected_context_length),
        ("forecast_horizon", expected_forecast_horizon),
        ("channels", expected_channels),
    ):
        try:
            actual = int(identity[key])
        except ValueError as exc:
            raise ValueError(f"{key} is not an integer: {identity[key]!r}") from exc
        if actual != expected:
            raise ValueError(f"expected {key}={expected}, got {actual}")
    if not identity["device"].lower().startswith("cuda"):
        raise ValueError(
            f"GPU resource figure requires a CUDA measurement, got device={identity['device']!r}"
        )
    return identity


def _validate_resources(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    by_condition: dict[str, dict[str, str]] = {}
    numeric = (
        "deployed_parameters",
        "adaptation_trainable_parameters",
        "psm_total_training_seconds_observed",
        "single_e2e_latency_ms_median",
        "batch_e2e_windows_per_second",
        "cuda_training_peak_allocated_mib_batch",
        "memory_bank_payload_bytes",
    )
    for index, row in enumerate(rows, 2):
        label = f"efficiency row {index}"
        condition = row["condition"].strip()
        if condition not in CONDITIONS:
            raise ValueError(f"{label}: unexpected condition {condition!r}")
        if condition in by_condition:
            raise ValueError(f"duplicate efficiency row for {condition}")
        for key in numeric:
            _number(
                row,
                key,
                label,
                positive=key not in {"memory_bank_payload_bytes", "adaptation_trainable_parameters"},
            )
        _integer(row, "batch_size_for_throughput", label, positive=True)
        if "random" not in row["microbenchmark_weights"].lower():
            raise ValueError(f"{label}: microbenchmark_weights must disclose random initialization")
        payload = _number(row, "memory_bank_payload_bytes", label)
        if condition in {"Native", "LoRA-Only"} and payload != 0:
            raise ValueError(f"{label}: non-retrieval condition must have zero memory payload")
        if condition in {"SRF-Only", "SRF+LoRA"} and payload <= 0:
            raise ValueError(f"{label}: retrieval condition must have positive memory payload")
        expected_deployed, expected_trainable = EXPECTED_PARAMETERS[condition]
        if _integer(row, "deployed_parameters", label, positive=True) != expected_deployed:
            raise ValueError(f"{label}: unexpected deployed parameter count")
        if _integer(row, "adaptation_trainable_parameters", label) != expected_trainable:
            raise ValueError(f"{label}: unexpected adaptation-trainable parameter count")
        expected_payload = EXPECTED_PAYLOAD_BYTES[256] if condition in {"SRF-Only", "SRF+LoRA"} else 0
        if _integer(row, "memory_bank_payload_bytes", label) != expected_payload:
            raise ValueError(
                f"{label}: expected memory payload {expected_payload}, got {payload}"
            )
        by_condition[condition] = row
    missing = [condition for condition in CONDITIONS if condition not in by_condition]
    if missing or len(rows) != len(CONDITIONS):
        raise ValueError(f"resource rows must contain exactly the four conditions; missing={missing}")
    return [by_condition[condition] for condition in CONDITIONS]


def _validate_scaling(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], list[int], list[int]]:
    cells: dict[tuple[int, int], dict[str, str]] = {}
    for index, row in enumerate(rows, 2):
        label = f"sensitivity row {index}"
        if row["condition"].strip() != "SRF-Only":
            raise ValueError(f"{label}: expected condition 'SRF-Only', got {row['condition']!r}")
        k = _integer(row, "retrieval_top_k", label, positive=True)
        memory = _integer(row, "memory_per_channel", label, positive=True)
        if k > memory:
            raise ValueError(f"{label}: K={k} cannot exceed M={memory}")
        for key in (
            "memory_bank_payload_bytes",
            "single_retrieval_latency_ms_median",
            "single_e2e_latency_ms_median",
        ):
            _number(row, key, label, positive=True)
        key = (memory, k)
        if key in cells:
            raise ValueError(f"duplicate sensitivity cell M={memory}, K={k}")
        cells[key] = row
    memories = list(EXPECTED_MEMORY)
    top_ks = list(EXPECTED_TOP_K)
    expected = {(memory, k) for memory in memories for k in top_ks}
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected)
    if missing or extra or len(rows) != 16:
        raise ValueError(
            "sensitivity CSV must be exactly "
            "K={1,3,5,10} x M={64,128,256,512}; "
            f"missing={missing}, extra={extra}, rows={len(rows)}"
        )
    for memory in memories:
        payloads = {
            _number(cells[(memory, k)], "memory_bank_payload_bytes", f"M={memory}, K={k}")
            for k in top_ks
        }
        if len(payloads) != 1:
            raise ValueError(
                f"memory payload should be invariant to K at fixed M={memory}, got {sorted(payloads)}"
            )
        payload = int(round(next(iter(payloads))))
        if payload != EXPECTED_PAYLOAD_BYTES[memory]:
            raise ValueError(
                f"M={memory}: expected retained payload {EXPECTED_PAYLOAD_BYTES[memory]}, got {payload}"
            )
    ordered = [cells[(memory, k)] for memory in memories for k in top_ks]
    return ordered, memories, top_ks


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    temporary.replace(path)


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("svg", "pdf", "png"):
        path = output_base.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _soft_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax.tick_params(width=0.7, length=3)


def _format_value(value: float, kind: str) -> str:
    if kind == "parameters":
        return f"{value:.2f}"
    if kind in {"seconds", "throughput", "vram", "memory"}:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _bar_panel(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    value_kind: str,
    panel: str,
) -> None:
    x = np.arange(len(CONDITIONS))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[CONDITION_COLORS[name] for name in CONDITIONS],
        edgecolor="#333333",
        linewidth=0.65,
    )
    for bar, condition in zip(bars, CONDITIONS):
        bar.set_hatch(CONDITION_HATCHES[condition])
    finite_max = max(values)
    top = finite_max * 1.22 if finite_max > 0 else 1.0
    ax.set_ylim(0, top)
    for bar, value in zip(bars, values):
        offset = top * 0.018
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            _format_value(value, value_kind),
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#272727",
        )
    ax.set_xticks(x, [CONDITION_SHORT[name] for name in CONDITIONS], rotation=18, ha="right")
    ax.set_title(title, fontsize=8.5, pad=5, fontweight="semibold")
    ax.set_ylabel(ylabel)
    _soft_grid(ax)
    _panel_label(ax, panel)


def _plot_resource_grid(
    rows: Sequence[Mapping[str, str]],
    identity: Mapping[str, str],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    metrics = (
        (
            "deployed_parameters",
            "Deployed parameters",
            "Parameters (million)",
            "parameters",
            1e-6,
        ),
        (
            "psm_total_training_seconds_observed",
            "Observed model-loop time",
            "Train/validation loop (s)",
            "seconds",
            1.0,
        ),
        (
            "single_e2e_latency_ms_median",
            "Single-window latency",
            "End-to-end median (ms)",
            "latency",
            1.0,
        ),
        (
            "batch_e2e_windows_per_second",
            "Fixed-batch throughput",
            "Windows s$^{-1}$ (thousands)",
            "throughput",
            1e-3,
        ),
        (
            "cuda_training_peak_allocated_mib_batch",
            "Stage training CUDA peak",
            "Stage peak allocation (MiB)",
            "vram",
            1.0,
        ),
        (
            "memory_bank_payload_bytes",
            "Retained memory bank",
            "NumPy payload (MiB)",
            "memory",
            1.0 / (1024.0**2),
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.15))
    letters = "abcdef"
    for ax, metric, panel in zip(axes.flat, metrics, letters):
        key, title, ylabel, kind, scale = metric
        values = [_number(row, key, row["condition"]) * scale for row in rows]
        _bar_panel(
            ax,
            values,
            title=title,
            ylabel=ylabel,
            value_kind=kind,
            panel=panel,
        )
    handles = [
        Patch(
            facecolor=CONDITION_COLORS[name],
            edgecolor="#333333",
            linewidth=0.65,
            hatch=CONDITION_HATCHES[name],
            label=name,
        )
        for name in CONDITIONS
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    batch_size = _single(rows, "batch_size_for_throughput", "resource rows")
    gpu = _single(rows, "gpu", "resource rows")
    fig.suptitle(
        "Resource profile across adaptation conditions",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.923,
        f"{identity['dataset']} · {identity['backbone']} · seed {identity['seed']} · "
        f"L/H/C={identity['context_length']}/{identity['forecast_horizon']}/{identity['channels']} · "
        f"batch={batch_size} · {gpu}",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#4D4D4D",
    )
    fig.text(
        0.5,
        0.008,
        "Random weights are used only for the resource microbenchmark (no accuracy claim). "
        "Training time is the model train/validation loop and excludes data, windows, memory fit and retrieval precompute.\n"
        "Training CUDA allocation is stage-specific; full-pipeline peak is max(base, adaptation).",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#606060",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.12, wspace=0.34, hspace=0.50)
    return _save_figure(fig, output_dir / "efficiency_resources", dpi)


def _plot_scaling(
    rows: Sequence[Mapping[str, str]],
    memories: Sequence[int],
    top_ks: Sequence[int],
    identity: Mapping[str, str],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    cells = {
        (
            _integer(row, "memory_per_channel", "scaling"),
            _integer(row, "retrieval_top_k", "scaling"),
        ): row
        for row in rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.25))
    for metric, title, ylabel, ax, panel in (
        (
            "single_retrieval_latency_ms_median",
            "Exact retrieval latency",
            "Median latency (ms)",
            axes[0],
            "a",
        ),
        (
            "single_e2e_latency_ms_median",
            "End-to-end latency",
            "Median latency (ms)",
            axes[1],
            "b",
        ),
    ):
        for index, memory in enumerate(memories):
            values = [_number(cells[(memory, k)], metric, f"M={memory}, K={k}") for k in top_ks]
            ax.plot(
                top_ks,
                values,
                color=LINE_COLORS[index % len(LINE_COLORS)],
                marker=LINE_MARKERS[index % len(LINE_MARKERS)],
                markersize=4.5,
                linewidth=1.5,
                label=f"M={memory}",
            )
        ax.set_xticks(top_ks)
        ax.set_xlabel("Retrieved candidates, K")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.5, pad=5, fontweight="semibold")
        _soft_grid(ax)
        _panel_label(ax, panel)

    payloads = [
        _number(cells[(memory, top_ks[0])], "memory_bank_payload_bytes", f"M={memory}")
        / (1024.0**2)
        for memory in memories
    ]
    axes[2].plot(
        memories,
        payloads,
        color="#B05B82",
        marker="o",
        markersize=5,
        linewidth=1.7,
    )
    axes[2].fill_between(memories, 0, payloads, color="#E4CCD8", alpha=0.45)
    axes[2].set_xticks(memories)
    axes[2].set_xlabel("Memory entries per channel, M")
    axes[2].set_ylabel("Retained payload (MiB)")
    axes[2].set_title("Memory-bank scaling", fontsize=8.5, pad=5, fontweight="semibold")
    axes[2].set_ylim(bottom=0)
    _soft_grid(axes[2])
    _panel_label(axes[2], "c")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(memories), loc="upper center", bbox_to_anchor=(0.5, 0.925))
    fig.suptitle(
        "Retrieval overhead scales with candidate count and memory size",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"{identity['dataset']} · {identity['backbone']} · seed {identity['seed']} · "
        f"L/H/C={identity['context_length']}/{identity['forecast_horizon']}/{identity['channels']} · "
        "SRF-Only · median single-window timing · "
        "CPU exact retrieval + GPU forecast",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#606060",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.22, wspace=0.34)
    return _save_figure(fig, output_dir / "retrieval_scaling", dpi)


def _source_resource_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["deployed_parameters_million"] = _number(row, "deployed_parameters", row["condition"]) / 1e6
        item["memory_bank_payload_mib"] = _number(row, "memory_bank_payload_bytes", row["condition"]) / (1024.0**2)
        output.append(item)
    return output


def _source_scaling_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["memory_bank_payload_mib"] = _number(row, "memory_bank_payload_bytes", "scaling") / (1024.0**2)
        output.append(item)
    return output


def render(
    efficiency_csv: Path,
    sensitivity_csv: Path,
    output_dir: Path,
    *,
    expected_status: str = "real",
    expected_dataset: str = "PSM",
    expected_backbone: str = "PatchTST",
    expected_seed: int = 42,
    expected_context_length: int = 192,
    expected_forecast_horizon: int = 8,
    expected_channels: int = 25,
    dpi: int = 600,
) -> list[Path]:
    resource_rows = _read_csv(efficiency_csv, RESOURCE_REQUIRED)
    scaling_rows = _read_csv(sensitivity_csv, SENSITIVITY_REQUIRED)
    identity = _validate_identity(
        resource_rows,
        scaling_rows,
        expected_status=expected_status,
        expected_dataset=expected_dataset,
        expected_backbone=expected_backbone,
        expected_seed=expected_seed,
        expected_context_length=expected_context_length,
        expected_forecast_horizon=expected_forecast_horizon,
        expected_channels=expected_channels,
    )
    ordered_resources = _validate_resources(resource_rows)
    ordered_scaling, memories, top_ks = _validate_scaling(scaling_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    resource_source = output_dir / "efficiency_resources_source_data.csv"
    scaling_source = output_dir / "retrieval_scaling_source_data.csv"
    _write_csv(resource_source, RESOURCE_SOURCE_COLUMNS, _source_resource_rows(ordered_resources))
    _write_csv(scaling_source, SCALING_SOURCE_COLUMNS, _source_scaling_rows(ordered_scaling))

    outputs = [resource_source, scaling_source]
    outputs.extend(_plot_resource_grid(ordered_resources, identity, output_dir, dpi))
    outputs.extend(_plot_scaling(ordered_scaling, memories, top_ks, identity, output_dir, dpi))

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": identity["benchmark_id"],
        "measurement_status": identity["measurement_status"],
        "dataset": identity["dataset"],
        "backbone": identity["backbone"],
        "seed": int(identity["seed"]),
        "device": identity["device"],
        "context_length": int(identity["context_length"]),
        "forecast_horizon": int(identity["forecast_horizon"]),
        "channels": int(identity["channels"]),
        "input_sha256": {
            "efficiency_resources.csv": _sha256(efficiency_csv),
            "sensitivity_latency.csv": _sha256(sensitivity_csv),
        },
        "grid": {"memory_per_channel": memories, "retrieval_top_k": top_ks},
        "resource_metric_definitions": {
            "parameters": "deployed_parameters / 1e6",
            "training_time": "PSM model train/validation loop wall time; excludes data preparation, window generation, memory fit and retrieval precompute",
            "latency": "single_e2e_latency_ms_median; batch=1 wall clock including retrieval where applicable",
            "throughput": "batch_e2e_windows_per_second at the CSV-reported fixed batch size",
            "gpu_memory": "cuda_training_peak_allocated_mib_batch; stage-specific PyTorch allocator peak (base for Native, adaptation for non-Native); full pipeline takes max(base, adaptation)",
            "memory_bank": "retained NumPy payload bytes / 2^20",
        },
        "integrity_notes": [
            "All plotted rows have one measurement_status and one benchmark_id across both inputs.",
            "The K x M grid is exactly {1,3,5,10} x {64,128,256,512}; payload matches the fixed protocol and is invariant to K at fixed M.",
            "SVG text is retained as editable text nodes.",
            "Resource microbenchmark weights are deterministic random initialization; no accuracy claim.",
            "Measurements are descriptive for one device/process/software stack and have no error bars.",
        ],
        "outputs": [path.name for path in outputs],
    }
    metadata_path = output_dir / "plot_efficiency_resources_metadata.json"
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    outputs.append(metadata_path)
    return outputs


def _write_self_test_inputs(root: Path) -> tuple[Path, Path]:
    efficiency = root / "efficiency_resources.csv"
    sensitivity = root / "sensitivity_latency.csv"
    resource_fields = sorted(RESOURCE_REQUIRED)
    rows = []
    for index, condition in enumerate(CONDITIONS):
        retrieval = condition in {"SRF-Only", "SRF+LoRA"}
        rows.append(
            {
                "benchmark_id": "selftest-20260924",
                "measurement_status": "synthetic",
                "dataset": "PSM",
                "backbone": "PatchTST",
                "condition": condition,
                "seed": 42,
                "device": "cuda:0",
                "gpu": "Synthetic GPU",
                "context_length": 192,
                "forecast_horizon": 8,
                "channels": 25,
                "deployed_parameters": EXPECTED_PARAMETERS[condition][0],
                "adaptation_trainable_parameters": EXPECTED_PARAMETERS[condition][1],
                "psm_total_training_seconds_observed": [36.0, 44.0, 51.0, 60.0][index],
                "single_e2e_latency_ms_median": [0.31, 0.34, 2.20, 2.34][index],
                "batch_e2e_windows_per_second": [21400, 20700, 6600, 6200][index],
                "cuda_training_peak_allocated_mib_batch": [430, 448, 505, 523][index],
                "memory_bank_payload_bytes": EXPECTED_PAYLOAD_BYTES[256] if retrieval else 0,
                "batch_size_for_throughput": 128,
                "microbenchmark_weights": "deterministic random initialization; no accuracy claim",
                "timing_scope": "synthetic self-test",
            }
        )
    _write_csv(efficiency, resource_fields, rows)

    sensitivity_fields = sorted(SENSITIVITY_REQUIRED)
    scaling = []
    for memory in (64, 128, 256, 512):
        payload = EXPECTED_PAYLOAD_BYTES[memory]
        for k in (1, 3, 5, 10):
            retrieval_ms = 0.20 + 0.0018 * memory + 0.030 * k
            scaling.append(
                {
                    "benchmark_id": "selftest-20260924",
                    "measurement_status": "synthetic",
                    "dataset": "PSM",
                    "backbone": "PatchTST",
                    "condition": "SRF-Only",
                    "seed": 42,
                    "device": "cuda:0",
                    "context_length": 192,
                    "forecast_horizon": 8,
                    "channels": 25,
                    "retrieval_top_k": k,
                    "memory_per_channel": memory,
                    "memory_bank_payload_bytes": payload,
                    "single_retrieval_latency_ms_median": retrieval_ms,
                    "single_e2e_latency_ms_median": retrieval_ms + 0.36 + 0.006 * k,
                }
            )
    _write_csv(sensitivity, sensitivity_fields, scaling)
    return efficiency, sensitivity


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="plot-efficiency-resources-") as temporary:
        root = Path(temporary)
        efficiency, sensitivity = _write_self_test_inputs(root)
        outputs = render(
            efficiency,
            sensitivity,
            root / "figures",
            expected_status="synthetic",
            expected_context_length=192,
            expected_forecast_horizon=8,
            expected_channels=25,
            dpi=150,
        )
        expected_names = {
            "efficiency_resources.svg",
            "efficiency_resources.pdf",
            "efficiency_resources.png",
            "retrieval_scaling.svg",
            "retrieval_scaling.pdf",
            "retrieval_scaling.png",
            "efficiency_resources_source_data.csv",
            "retrieval_scaling_source_data.csv",
            "plot_efficiency_resources_metadata.json",
        }
        actual_names = {path.name for path in outputs}
        if actual_names != expected_names:
            raise AssertionError(f"unexpected output set: {sorted(actual_names)}")
        for path in outputs:
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"missing/empty output: {path}")
        for name in ("efficiency_resources.svg", "retrieval_scaling.svg"):
            svg = (root / "figures" / name).read_text(encoding="utf-8")
            if "<text" not in svg:
                raise AssertionError(f"{name}: editable SVG text nodes not found")
        metadata = json.loads(
            (root / "figures" / "plot_efficiency_resources_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            metadata.get("context_length"),
            metadata.get("forecast_horizon"),
            metadata.get("channels"),
        ) != (192, 8, 25):
            raise AssertionError("L/H/C identity was not preserved in plot metadata")
        # Negative integrity check: benchmark mismatch must be rejected.
        rows = _read_csv(sensitivity, SENSITIVITY_REQUIRED)
        rows[0]["benchmark_id"] = "wrong-benchmark"
        bad = root / "bad_sensitivity.csv"
        _write_csv(bad, sorted(SENSITIVITY_REQUIRED), rows)
        try:
            render(
                efficiency,
                bad,
                root / "must_not_render",
                expected_status="synthetic",
                dpi=72,
            )
        except ValueError as exc:
            if "benchmark_id" not in str(exc):
                raise AssertionError(f"wrong rejection for benchmark mismatch: {exc}") from exc
        else:
            raise AssertionError("benchmark mismatch was not rejected")
        # Negative protocol check: CLI/render expectations must bind L/H/C.
        try:
            render(
                efficiency,
                sensitivity,
                root / "must_not_render_context",
                expected_status="synthetic",
                expected_context_length=384,
                expected_forecast_horizon=8,
                expected_channels=25,
                dpi=72,
            )
        except ValueError as exc:
            if "context_length" not in str(exc):
                raise AssertionError(f"wrong rejection for L/H/C mismatch: {exc}") from exc
        else:
            raise AssertionError("L/H/C protocol mismatch was not rejected")
        # Negative grid check: a truncated K x M input must not yield a figure.
        scaling_rows = _read_csv(sensitivity, SENSITIVITY_REQUIRED)
        truncated = root / "truncated_sensitivity.csv"
        _write_csv(truncated, sorted(SENSITIVITY_REQUIRED), scaling_rows[:-1])
        try:
            render(
                efficiency,
                truncated,
                root / "must_not_render_truncated",
                expected_status="synthetic",
                dpi=72,
            )
        except ValueError as exc:
            if "exactly" not in str(exc) and "missing" not in str(exc):
                raise AssertionError(f"wrong rejection for truncated grid: {exc}") from exc
        else:
            raise AssertionError("truncated K x M grid was not rejected")
    print("self-test passed: schema, integrity checks, SVG/PDF/PNG exports, and source data")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficiency-csv", type=Path)
    parser.add_argument("--sensitivity-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/efficiency_resources_figures_v1"))
    parser.add_argument("--expected-status", default="real")
    parser.add_argument("--expected-dataset", default="PSM")
    parser.add_argument("--expected-backbone", default="PatchTST")
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-context-length", type=int, default=192)
    parser.add_argument("--expected-forecast-horizon", type=int, default=8)
    parser.add_argument("--expected-channels", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    for option in (
        "expected_context_length",
        "expected_forecast_horizon",
        "expected_channels",
    ):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if not args.self_test and (args.efficiency_csv is None or args.sensitivity_csv is None):
        parser.error("--efficiency-csv and --sensitivity-csv are required unless --self-test is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    outputs = render(
        args.efficiency_csv,
        args.sensitivity_csv,
        args.output_dir,
        expected_status=args.expected_status,
        expected_dataset=args.expected_dataset,
        expected_backbone=args.expected_backbone,
        expected_seed=args.expected_seed,
        expected_context_length=args.expected_context_length,
        expected_forecast_horizon=args.expected_forecast_horizon,
        expected_channels=args.expected_channels,
        dpi=args.dpi,
    )
    print(
        f"complete: {len(outputs)} files; "
        f"figures={args.output_dir / 'efficiency_resources.svg'}, "
        f"{args.output_dir / 'retrieval_scaling.svg'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
