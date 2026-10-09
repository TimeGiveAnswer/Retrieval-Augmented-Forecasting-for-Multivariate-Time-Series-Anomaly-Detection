"""Full-data main matrix, all entities, three backbones and four conditions.

Normal history trains the models. Disjoint labelled validation selects score
settings; held-out labels are used only for final raw, no-PA metrics.
Point/window limits default to zero (unlimited).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from full_data import discover_datasets, concatenate_arrays
import torch

from data_pipeline import (
    ChannelMemoryBank,
    GlobalMemoryBank,
    RobustScaler,
    load_series,
    make_windows,
)
from models import ConditionModel, build_backbone
from training import (
    NumpyWindowDataset,
    count_parameters,
    evaluate_scores,
    overlap_add_scores,
    predict_windows,
    save_json,
    seed_all,
    train_model,
)


DEFAULT_DATASETS = ("PSM", "MSL", "SMAP", "SMD", "SWaT")
DEFAULT_BACKBONES = ("patch_transformer", "period_conv", "selective_ssm")
DEFAULT_CONDITIONS = ("native", "lora", "srf", "joint")
CONTEXT_LENGTH = 192
FORECAST_HORIZON = 8
SPLIT_FRACTIONS = (0.70, 0.10, 0.10, 0.10)
BASE_INTERNAL_VALIDATION_FRACTION = 0.15


@dataclass
class PreparedDataset:
    """All label-free arrays needed by every backbone for one dataset."""

    name: str
    channels: int
    metadata: dict[str, Any]
    base: NumpyWindowDataset
    base_validation: NumpyWindowDataset
    adapt_plain: NumpyWindowDataset
    validation_plain: NumpyWindowDataset
    score_validation_plain: NumpyWindowDataset
    test_plain: NumpyWindowDataset
    adapt_retrieval: NumpyWindowDataset | None
    validation_retrieval: NumpyWindowDataset | None
    score_validation_retrieval: NumpyWindowDataset | None
    test_retrieval: NumpyWindowDataset | None
    score_validation_labels: np.ndarray
    score_validation_segment_lengths: tuple[int, ...]
    test_labels: np.ndarray
    test_segment_lengths: tuple[int, ...]
    retrieval_seconds: dict[str, float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-").lower()


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    return (int(seed) ^ (zlib.crc32(payload) & 0x7FFFFFFF)) & 0x7FFFFFFF


def _finite_or_none(value: Any) -> Any:
    """Recursively make a value strict-JSON compatible."""

    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _finite_or_none(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _finite_or_none(value.tolist())
    if isinstance(value, np.generic):
        return _finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            ends_with_newline = existing.read(1) in {b"\n", b"\r"}
        if not ends_with_newline:
            # A hard interruption may leave one incomplete final JSON object.
            # Isolate it on its own line so the next valid record remains
            # recoverable by _read_jsonl.
            with path.open("ab") as existing:
                existing.write(b"\n")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                _finite_or_none(dict(row)),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING ignoring malformed JSONL record at "
                    f"{path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _result_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("configuration_id", "")),
        str(row.get("dataset", "")),
        str(row.get("backbone", "")),
        str(row.get("condition", "")),
        int(row.get("seed", -1)),
    )


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _write_summary(results_path: Path, summary_path: Path) -> None:
    """Atomically rebuild a compact CSV from the latest row for every key."""

    latest: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        latest[_result_key(row)] = row
    columns = (
        "configuration_id",
        "timestamp_utc",
        "status",
        "dataset",
        "backbone",
        "condition",
        "seed",
        "auprc",
        "auroc",
        "forecast_loss",
        "gate_mean",
        "total_parameters",
        "comparison_deployed_parameters",
        "runtime_wrapper_parameters",
        "srf_fallback_snapshot_parameters",
        "trainable_parameters",
        "final_stage_trainable_parameters",
        "latency_ms_per_window",
        "model_latency_ms_per_window",
        "inference_pipeline_ms_per_window",
        "retrieval_latency_ms_per_window",
        "windows_per_second",
        "base_best_val_loss",
        "condition_best_val_loss",
        "test_windows",
        "evaluated_points",
        "positive_points",
        "error_type",
        "error_message",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in sorted(latest):
            row = latest[key]
            output = {
                "configuration_id": row.get("configuration_id"),
                "timestamp_utc": row.get("timestamp_utc"),
                "status": row.get("status"),
                "dataset": row.get("dataset"),
                "backbone": row.get("backbone"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "auprc": _nested(row, "metrics", "auprc"),
                "auroc": _nested(row, "metrics", "auroc"),
                "forecast_loss": row.get("forecast_loss"),
                "gate_mean": row.get("gate_mean"),
                # Backward-compatible total/trainable columns now use the
                # scientifically comparable deployed and peak-stage counts.
                "total_parameters": _nested(
                    row, "parameters", "comparison_deployed_total"
                ),
                "comparison_deployed_parameters": _nested(
                    row, "parameters", "comparison_deployed_total"
                ),
                "runtime_wrapper_parameters": _nested(
                    row, "parameters", "runtime_wrapper_total"
                ),
                "srf_fallback_snapshot_parameters": _nested(
                    row, "parameters", "srf_fallback_snapshot_parameters"
                ),
                "trainable_parameters": _nested(
                    row, "parameters", "peak_trainable_parameters"
                ),
                "final_stage_trainable_parameters": _nested(
                    row, "parameters", "trainable"
                ),
                "latency_ms_per_window": _nested(row, "latency", "total_ms_per_window"),
                "model_latency_ms_per_window": _nested(row, "latency", "model_ms_per_window"),
                "inference_pipeline_ms_per_window": _nested(
                    row, "latency", "model_ms_per_window"
                ),
                "retrieval_latency_ms_per_window": _nested(row, "latency", "retrieval_ms_per_window"),
                "windows_per_second": _nested(row, "latency", "end_to_end_windows_per_second"),
                "base_best_val_loss": _nested(row, "base_training", "best_val_loss"),
                "condition_best_val_loss": _nested(row, "condition_training", "best_val_loss"),
                "test_windows": row.get("test_windows"),
                "evaluated_points": _nested(row, "metrics", "n_evaluated"),
                "positive_points": _nested(row, "metrics", "positives"),
                "error_type": _nested(row, "error", "type"),
                "error_message": _nested(row, "error", "message"),
            }
            writer.writerow({key: _finite_or_none(output.get(key)) for key in columns})
    os.replace(temporary, summary_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    return {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "implementation_sha256": {
            name: _file_sha256(project_dir / name)
            for name in ("run_main.py", "data_pipeline.py", "models.py", "training.py", "full_data.py")
        },
        "datasets": list(args.datasets),
        "backbones": list(args.backbones),
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "crop_seed": args.crop_seed,
        "crop_mode": args.crop_mode,
        "context_length": args.context_length,
        "forecast_horizon": args.forecast_horizon,
        "split_fractions": {
            "base": SPLIT_FRACTIONS[0],
            "memory": SPLIT_FRACTIONS[1],
            "adapt": SPLIT_FRACTIONS[2],
            "validation": SPLIT_FRACTIONS[3],
        },
        "base_internal_validation_fraction": BASE_INTERNAL_VALIDATION_FRACTION,
        "max_train_points": args.max_train_points,
        "max_test_points": args.max_test_points,
        "base_max_windows": args.base_max_windows,
        "condition_max_windows": args.condition_max_windows,
        "validation_max_windows": args.validation_max_windows,
        "memory_candidate_max_windows": args.memory_candidate_max_windows,
        "test_stride": args.test_stride,
        "labeled_validation_fraction": args.labeled_validation_fraction,
        "score_validation_block_size": args.score_validation_block_size,
        "memory_per_channel": args.memory_per_channel,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_top_k": args.retrieval_top_k,
        "memory_selection": args.memory_selection,
        "memory_source": args.memory_source,
        "memory_temperature": args.memory_temperature,
        "score_memory_grid": list(args.score_memory_grid),
        "score_agreement_grid": list(args.score_agreement_grid),
        "score_top_q_grid": list(args.score_top_q_grid),
        "score_smoothing_grid": list(args.score_smoothing_grid),
        "score_selection": args.score_selection,
        "d_model": args.d_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_epochs": args.lora_epochs,
        "lora_lr": args.lora_lr,
        "joint_lora_lr": args.joint_lora_lr,
        "joint_update_fusion_head": args.joint_update_fusion_head,
        "joint_fusion_lr": args.joint_fusion_lr,
        "joint_head_lr": args.joint_head_lr,
        "head_lr": args.head_lr,
        "reference_consistency_weight": args.reference_consistency_weight,
        "gate_regularization_weight": args.gate_regularization_weight,
        "reference_quality_margin": args.reference_quality_margin,
        "reference_quality_temperature": args.reference_quality_temperature,
        "lora_scale_grid": list(args.lora_scale_grid),
        "report_full_strength_lora_sensitivity": True,
        "lora_train_head": args.lora_train_head,
        "fusion_hidden": args.fusion_hidden,
        "base_epochs": args.base_epochs,
        "condition_epochs": args.condition_epochs,
        "base_lr": args.base_lr,
        "condition_lr": args.condition_lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss": args.loss,
        "top_q": args.top_q,
        "device": args.device,
        "amp_requested": args.amp,
        "period_conv_amp": False,
        "quick": args.quick,
    }


def _configuration_id(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _base_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can change fitted base-model weights."""

    keys = (
        "data_root",
        "implementation_sha256",
        "crop_seed",
        "crop_mode",
        "context_length",
        "forecast_horizon",
        "split_fractions",
        "base_internal_validation_fraction",
        "max_train_points",
        "base_max_windows",
        "validation_max_windows",
        "d_model",
        "base_epochs",
        "base_lr",
        "batch_size",
        "patience",
        "loss",
        "device",
        "amp_requested",
        "period_conv_amp",
    )
    return {key: configuration[key] for key in keys}


def _effective_limit(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _four_way_split(
    values: np.ndarray, context_length: int, forecast_horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact contiguous 70/10/10/10 blocks without an unassigned tail."""

    n_points = len(values)
    counts = [int(math.floor(n_points * fraction)) for fraction in SPLIT_FRACTIONS[:3]]
    counts.append(n_points - sum(counts))
    if min(counts) < context_length + forecast_horizon:
        raise ValueError(
            f"Training series with {n_points} points is too short for four L+H blocks; "
            f"split sizes would be {counts}"
        )
    first = counts[0]
    second = first + counts[1]
    third = second + counts[2]
    return values[:first], values[first:second], values[second:third], values[third:]


def _window_dataset(
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    retrieval: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> NumpyWindowDataset:
    contexts, futures, indices = windows
    if retrieval is None:
        return NumpyWindowDataset(contexts, futures, end_indices=indices)
    analog, distance, divergence = retrieval
    return NumpyWindowDataset(
        contexts,
        futures,
        analog=analog,
        distance=distance,
        divergence=divergence,
        end_indices=indices,
    )


def _make_test_windows(
    values: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the configured stride and append the final legal window when needed.

    The terminal window preserves the requested regular stride everywhere else
    while preventing an avoidable uncovered suffix of up to H-1 samples.
    """

    windows = make_windows(
        values,
        args.context_length,
        args.forecast_horizon,
        stride=args.test_stride,
        max_windows=None,
        seed=0,
    )
    contexts, futures, indices = windows
    terminal_start = len(values) - args.context_length - args.forecast_horizon
    terminal_end = terminal_start + args.context_length
    if int(indices[-1]) == terminal_end:
        return windows
    terminal_context = values[terminal_start:terminal_end][None, ...]
    terminal_future = values[
        terminal_end : terminal_end + args.forecast_horizon
    ][None, ...]
    return (
        np.concatenate((contexts, terminal_context.astype(np.float32)), axis=0),
        np.concatenate((futures, terminal_future.astype(np.float32)), axis=0),
        np.concatenate((indices, np.asarray([terminal_end], dtype=np.int64))),
    )


def _split_labeled_blocks(
    values: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Deterministically interleave labeled-validation and held-out test blocks.

    A whole contiguous block, rather than individual points, is assigned at a
    time.  Windows are later built independently inside every block, so no
    artificial transition is introduced and no label values influence the
    assignment.  This yields validation anomalies even when attacks occur only
    late in the official test sequence (as in SWaT and several SMD machines).
    """

    block_size = max(
        int(args.score_validation_block_size),
        args.context_length + args.forecast_horizon + 1,
    )
    period = max(2, int(round(1.0 / args.labeled_validation_fraction)))
    slices: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        stop = min(start + block_size, len(values))
        if stop - start < args.context_length + args.forecast_horizon + 1 and slices:
            prior_start, _ = slices[-1]
            slices[-1] = (prior_start, stop)
        else:
            slices.append((start, stop))
        start = stop
    validation_values: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    test_values: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    for block_index, (start, stop) in enumerate(slices):
        if block_index % period == 0:
            validation_values.append(values[start:stop])
            validation_labels.append(labels[start:stop])
        else:
            test_values.append(values[start:stop])
            test_labels.append(labels[start:stop])
    if not validation_values or not test_values:
        raise ValueError("blockwise labeled validation split produced an empty side")
    return validation_values, validation_labels, test_values, test_labels


def _make_segmented_windows(
    segments: Sequence[np.ndarray], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    offset = 0
    for segment in segments:
        current = _make_test_windows(segment, args)
        contexts.append(current[0])
        futures.append(current[1])
        indices.append(current[2] + offset)
        offset += len(segment)
    return (
        concatenate_arrays(contexts),
        concatenate_arrays(futures),
        np.concatenate(indices, axis=0),
    )


def _query_memory(
    memory: ChannelMemoryBank | GlobalMemoryBank,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    start = time.perf_counter()
    retrieved = memory.query(
        windows[0],
        top_k=args.retrieval_top_k,
        return_candidates=True,
    )
    return retrieved, time.perf_counter() - start


def _prepare_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    seed: int,
    need_retrieval: bool,
) -> PreparedDataset:
    """Load, split, scale, window, and retrieve without consulting test labels."""

    train_raw, test_raw, test_labels, source_metadata = load_series(
        dataset_name,
        args.data_root,
        max_train_points=_effective_limit(args.max_train_points),
        max_test_points=_effective_limit(args.max_test_points),
        seed=args.crop_seed,
        crop_mode=args.crop_mode,
    )
    # Label *values* never influence scaling, crops, windows, retrieval, or
    # model fitting.  The official test timeline is deterministically divided
    # into labelled-validation and held-out blocks by block index; only the
    # former labels tune score-combination hyperparameters and the LoRA safety
    # blend.  The disjoint held-out labels are never exposed here.
    base_raw, memory_raw, adapt_raw, validation_raw = _four_way_split(
        train_raw, args.context_length, args.forecast_horizon
    )
    base_fit_stop = int(
        math.floor(len(base_raw) * (1.0 - BASE_INTERNAL_VALIDATION_FRACTION))
    )
    base_fit_raw = base_raw[:base_fit_stop]
    base_validation_raw = base_raw[base_fit_stop:]
    minimum_block = args.context_length + args.forecast_horizon
    if min(len(base_fit_raw), len(base_validation_raw)) < minimum_block:
        raise ValueError(
            "base block is too short for a leakage-separated internal validation split"
        )
    # Every value is from the normal training split.  Fitting the scaler on the
    # full normal history is leakage-safe and prevents early-regime statistics
    # from turning later normal operating modes into enormous outliers.
    scaler = RobustScaler().fit(train_raw)
    base_full_values = scaler.transform(base_raw)
    base_values = scaler.transform(base_fit_raw)
    base_validation_values = scaler.transform(base_validation_raw)
    memory_values = scaler.transform(memory_raw)
    adapt_values = scaler.transform(adapt_raw)
    validation_values = scaler.transform(validation_raw)
    (
        score_validation_raw_segments,
        score_validation_label_segments,
        evaluation_test_raw_segments,
        evaluation_test_label_segments,
    ) = _split_labeled_blocks(test_raw, np.asarray(test_labels), args)
    score_validation_values_segments = [
        scaler.transform(segment) for segment in score_validation_raw_segments
    ]
    test_values_segments = [
        scaler.transform(segment) for segment in evaluation_test_raw_segments
    ]
    score_validation_labels = np.concatenate(score_validation_label_segments).astype(
        np.int64, copy=False
    )
    evaluation_test_labels = np.concatenate(evaluation_test_label_segments).astype(
        np.int64, copy=False
    )

    base_windows = make_windows(
        base_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=args.base_max_windows,
        seed=_stable_seed(args.crop_seed, dataset_name, "base-windows"),
    )
    base_validation_windows = make_windows(
        base_validation_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=args.validation_max_windows,
        seed=_stable_seed(args.crop_seed, dataset_name, "base-validation-windows"),
    )
    adapt_windows = make_windows(
        adapt_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=args.condition_max_windows,
        seed=_stable_seed(args.crop_seed, dataset_name, "adapt-windows"),
    )
    validation_windows = make_windows(
        validation_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=args.validation_max_windows,
        seed=_stable_seed(args.crop_seed, dataset_name, "validation-windows"),
    )
    score_validation_windows = _make_segmented_windows(
        score_validation_values_segments, args
    )
    # Full chronological coverage of the held-out test segment.
    test_windows = _make_segmented_windows(test_values_segments, args)

    adapt_retrieval = None
    validation_retrieval = None
    score_validation_retrieval = None
    test_retrieval = None
    retrieval_seconds = {
        "adapt": 0.0,
        "validation": 0.0,
        "score_validation": 0.0,
        "test": 0.0,
    }
    memory_sizes: list[int] = []
    memory_window_count = 0
    memory_source_points = 0
    if need_retrieval:
        # Most datasets benefit from broad operating-mode coverage in the first
        # 80% of normal training history.  A dedicated disjoint 10% memory
        # block remains available for datasets where a more local reference
        # distribution validates better.
        if args.memory_source == "base_memory":
            memory_source_values = np.concatenate(
                (base_full_values, memory_values), axis=0
            )
        else:
            memory_source_values = memory_values
        memory_windows = make_windows(
            memory_source_values,
            args.context_length,
            args.forecast_horizon,
            stride=1,
            max_windows=_effective_limit(args.memory_candidate_max_windows),
            seed=_stable_seed(args.crop_seed, dataset_name, "memory-windows"),
        )
        memory_class = (
            ChannelMemoryBank if args.retrieval_mode == "channel" else GlobalMemoryBank
        )
        memory = memory_class(
            memory_per_channel=args.memory_per_channel,
            top_k=args.retrieval_top_k,
            temperature=args.memory_temperature,
            selection=args.memory_selection,
        ).fit(memory_windows[0], memory_windows[1], end_indices=memory_windows[2])
        memory_window_count = len(memory_windows[0])
        memory_source_points = int(len(memory_source_values))
        memory_sizes = memory.size_per_channel_.astype(int).tolist()  # type: ignore[union-attr]
        adapt_values_retrieved, retrieval_seconds["adapt"] = _query_memory(
            memory, adapt_windows, args
        )
        validation_values_retrieved, retrieval_seconds["validation"] = _query_memory(
            memory, validation_windows, args
        )
        score_validation_values_retrieved, retrieval_seconds[
            "score_validation"
        ] = _query_memory(
            memory, score_validation_windows, args
        )
        test_values_retrieved, retrieval_seconds["test"] = _query_memory(
            memory, test_windows, args
        )
        adapt_retrieval = _window_dataset(adapt_windows, adapt_values_retrieved)
        validation_retrieval = _window_dataset(
            validation_windows, validation_values_retrieved
        )
        score_validation_retrieval = _window_dataset(
            score_validation_windows, score_validation_values_retrieved
        )
        test_retrieval = _window_dataset(test_windows, test_values_retrieved)

    source_metadata = dict(source_metadata)
    source_metadata.update(
        {
            "split_points": {
                "base_total": len(base_raw),
                "base_fit": len(base_fit_raw),
                "base_internal_validation": len(base_validation_raw),
                "memory": len(memory_raw),
                "adapt": len(adapt_raw),
                "validation": len(validation_raw),
            },
            "window_counts": {
                "base": len(base_windows[0]),
                "base_internal_validation": len(base_validation_windows[0]),
                "memory_candidates": memory_window_count,
                "adapt": len(adapt_windows[0]),
                "validation": len(validation_windows[0]),
                "score_validation": len(score_validation_windows[0]),
                "test": len(test_windows[0]),
            },
            "memory_size_per_channel": memory_sizes,
            "memory_source": args.memory_source,
            "memory_source_points": memory_source_points,
            "scaler_fit_block": "full normal training series",
            "scaler_center": scaler.center_.tolist() if scaler.center_ is not None else [],
            "scaler_scale": scaler.scale_.tolist() if scaler.scale_ is not None else [],
            "crop_seed": int(args.crop_seed),
            "labeled_validation_points": int(len(score_validation_labels)),
            "heldout_test_points": int(len(evaluation_test_labels)),
            "labeled_validation_segments": [
                int(len(segment)) for segment in score_validation_label_segments
            ],
            "heldout_test_segments": [
                int(len(segment)) for segment in evaluation_test_label_segments
            ],
            "labeled_validation_block_size": int(args.score_validation_block_size),
            "test_window_policy": (
                f"all starts at stride {args.test_stride}, plus the final legal window"
            ),
            "test_labels_used_for": (
                "deterministically interleaved labeled-validation blocks tune "
                "score-combination and LoRA-blend hyperparameters; held-out "
                "block labels are used only "
                "for final evaluate_scores"
            ),
        }
    )
    return PreparedDataset(
        name=dataset_name,
        channels=int(train_raw.shape[1]),
        metadata=source_metadata,
        base=_window_dataset(base_windows),
        base_validation=_window_dataset(base_validation_windows),
        adapt_plain=_window_dataset(adapt_windows),
        validation_plain=_window_dataset(validation_windows),
        score_validation_plain=_window_dataset(score_validation_windows),
        test_plain=_window_dataset(test_windows),
        adapt_retrieval=adapt_retrieval,
        validation_retrieval=validation_retrieval,
        score_validation_retrieval=score_validation_retrieval,
        test_retrieval=test_retrieval,
        score_validation_labels=score_validation_labels,
        score_validation_segment_lengths=tuple(
            int(len(segment)) for segment in score_validation_label_segments
        ),
        test_labels=evaluation_test_labels,
        test_segment_lengths=tuple(
            int(len(segment)) for segment in evaluation_test_label_segments
        ),
        retrieval_seconds=retrieval_seconds,
    )


def _amp_for(backbone: str, args: argparse.Namespace) -> bool:
    # torch.fft.rfft on CUDA half can fail for unsupported transform lengths.
    return bool(args.amp and backbone != "period_conv")


def _load_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError(f"Invalid clean-room checkpoint: {path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    training = payload.get("training")
    return dict(training) if isinstance(training, Mapping) else {}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    training: Mapping[str, Any],
    configuration_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "training": _finite_or_none(dict(training)),
            "configuration_id": configuration_id,
            "created_at": _utc_now(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _fit_or_load_base(
    prepared: PreparedDataset,
    backbone_name: str,
    args: argparse.Namespace,
    seed: int,
    base_configuration_id: str,
    checkpoint_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    seed_all(_stable_seed(seed, prepared.name, backbone_name, "base-init"))
    base = build_backbone(
        backbone_name,
        prepared.channels,
        args.context_length,
        args.forecast_horizon,
        d_model=args.d_model,
    )
    checkpoint_path = checkpoint_dir / (
        f"base-{_slug(prepared.name)}-{_slug(backbone_name)}-seed{seed}-"
        f"{base_configuration_id}.pt"
    )
    training = _load_checkpoint(checkpoint_path, base)
    source = "checkpoint"
    if training is None:
        training = train_model(
            base,
            prepared.base,
            prepared.base_validation,
            epochs=args.base_epochs,
            batch_size=args.batch_size,
            lr=args.base_lr,
            device=args.device,
            amp=_amp_for(backbone_name, args),
            patience=args.patience,
            loss=args.loss,
            seed=_stable_seed(seed, prepared.name, backbone_name, "base-train"),
        )
        base.to("cpu")
        _save_checkpoint(checkpoint_path, base, training, base_configuration_id)
        source = "trained"
    else:
        base.to("cpu")
    return base, training, source


def _condition_arrays(
    prepared: PreparedDataset, condition: str
) -> tuple[
    NumpyWindowDataset,
    NumpyWindowDataset,
    NumpyWindowDataset,
    NumpyWindowDataset,
]:
    if condition in {"srf", "joint"}:
        if (
            prepared.adapt_retrieval is None
            or prepared.validation_retrieval is None
            or prepared.score_validation_retrieval is None
            or prepared.test_retrieval is None
        ):
            raise RuntimeError("retrieval arrays were not prepared for an SRF condition")
        return (
            prepared.adapt_retrieval,
            prepared.validation_retrieval,
            prepared.score_validation_retrieval,
            prepared.test_retrieval,
        )
    return (
        prepared.adapt_plain,
        prepared.validation_plain,
        prepared.score_validation_plain,
        prepared.test_plain,
    )


def _aggregate_candidate_distance(distance: np.ndarray) -> np.ndarray:
    """Use the nearest retained candidate for the memory anomaly signal."""

    values = np.asarray(distance, dtype=np.float64)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        return np.min(values, axis=1)
    raise ValueError(f"distance must have shape [N,C] or [N,K,C], got {values.shape}")


def _empirical_transform(
    reference: np.ndarray,
    values: np.ndarray,
    *,
    tail_score: bool,
) -> np.ndarray:
    """Per-channel empirical calibration fitted on normal validation only."""

    ref = np.asarray(reference, dtype=np.float64)
    current = np.asarray(values, dtype=np.float64)
    if ref.ndim < 2 or current.ndim < 2 or ref.shape[-1] != current.shape[-1]:
        raise ValueError("empirical calibration requires matching channel axes")
    channels = ref.shape[-1]
    ref_flat = ref.reshape(-1, channels)
    current_flat = current.reshape(-1, channels)
    transformed = np.empty_like(current_flat, dtype=np.float64)
    for channel in range(channels):
        distribution = np.sort(ref_flat[:, channel][np.isfinite(ref_flat[:, channel])])
        if distribution.size == 0:
            transformed[:, channel] = 0.0 if tail_score else 0.5
            continue
        query = np.nan_to_num(
            current_flat[:, channel],
            nan=float(distribution[len(distribution) // 2]),
            posinf=float(distribution[-1]),
            neginf=float(distribution[0]),
        )
        ranks = np.searchsorted(distribution, query, side="right")
        cdf = (ranks.astype(np.float64) + 0.5) / (distribution.size + 1.0)
        cdf = np.clip(cdf, 1e-5, 1.0 - 1e-5)
        if tail_score:
            transformed[:, channel] = np.minimum(-np.log1p(-cdf), 12.0)
        else:
            transformed[:, channel] = 1.0 - cdf
    return transformed.reshape(current.shape).astype(np.float32)


def _causal_smooth(scores: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    width = int(width)
    if width <= 1:
        return values.copy()
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(safe)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    result = np.full_like(values, np.nan)
    for index in range(len(values)):
        start = max(0, index - width + 1)
        count = counts[index + 1] - counts[start]
        if count:
            result[index] = (sums[index + 1] - sums[start]) / count
    return result


def _causal_smooth_segments(
    scores: np.ndarray, width: int, segment_lengths: Sequence[int] | None
) -> np.ndarray:
    """Causally smooth without carrying state across disjoint time blocks."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not segment_lengths:
        return _causal_smooth(values, width)
    lengths = tuple(int(length) for length in segment_lengths)
    if any(length <= 0 for length in lengths) or sum(lengths) != len(values):
        raise ValueError("segment lengths must be positive and sum to score length")
    result = np.full_like(values, np.nan)
    offset = 0
    for length in lengths:
        stop = offset + length
        result[offset:stop] = _causal_smooth(values[offset:stop], width)
        offset = stop
    return result


def _top_q_mean(channel_scores: np.ndarray, top_q: float) -> np.ndarray:
    values = np.asarray(channel_scores, dtype=np.float64)
    channels = values.shape[1]
    wanted = max(1, min(channels, int(math.ceil(float(top_q) * channels))))
    safe = np.where(np.isfinite(values), values, -np.inf)
    partitioned = np.partition(safe, kth=channels - wanted, axis=1)[:, -wanted:]
    partitioned[~np.isfinite(partitioned)] = np.nan
    finite = np.isfinite(partitioned)
    sums = np.nansum(partitioned, axis=1)
    counts = finite.sum(axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _score_channel_components(
    prediction: Mapping[str, Any],
    validation_errors: np.ndarray,
    n_points: int,
    memory_score: np.ndarray | None,
    agreement: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = dict(
        predictions=prediction["predictions"],
        targets=prediction["targets"],
        end_indices=prediction["end_indices"],
        validation_residuals=validation_errors,
        residual_kind="absolute",
        top_q=1.0,
        n_points=n_points,
    )
    base = overlap_add_scores(**common)["channel_scores"]
    residual = np.asarray(base, dtype=np.float64)
    memory_component = np.zeros_like(residual)
    agreement_component = np.zeros_like(residual)
    if memory_score is not None:
        with_memory = overlap_add_scores(
            **common, memory_distance=memory_score, memory_weight=1.0
        )["channel_scores"]
        memory_component = np.asarray(with_memory, dtype=np.float64) - residual
    if agreement is not None:
        with_agreement = overlap_add_scores(
            **common, agreement=agreement, agreement_weight=1.0
        )["channel_scores"]
        agreement_component = np.asarray(with_agreement, dtype=np.float64) - residual
    return residual, memory_component, agreement_component


def _select_score_parameters(
    labels: np.ndarray,
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    use_retrieval: bool,
    segment_lengths: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    residual, memory, agreement = components
    memory_grid = args.score_memory_grid if use_retrieval else [0.0]
    agreement_grid = args.score_agreement_grid if use_retrieval else [0.0]
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_objective = -math.inf
    for top_q in args.score_top_q_grid:
        for memory_weight in memory_grid:
            for agreement_weight in agreement_grid:
                combined = (
                    residual
                    + float(memory_weight) * memory
                    + float(agreement_weight) * agreement
                )
                point_scores = _top_q_mean(combined, float(top_q))
                for smoothing in args.score_smoothing_grid:
                    smoothed = _causal_smooth_segments(
                        point_scores, int(smoothing), segment_lengths
                    )
                    metrics = evaluate_scores(labels, smoothed)
                    aggregate_objective = metrics.get("auprc")
                    if aggregate_objective is None or not np.isfinite(aggregate_objective):
                        continue
                    block_auprc: list[float] = []
                    if args.score_selection == "macro_block" and segment_lengths:
                        offset = 0
                        for length in segment_lengths:
                            stop = offset + int(length)
                            block_metrics = evaluate_scores(
                                labels[offset:stop], smoothed[offset:stop]
                            )
                            value = block_metrics.get("auprc")
                            if value is not None and np.isfinite(value):
                                block_auprc.append(float(value))
                            offset = stop
                        if offset != len(labels):
                            raise ValueError(
                                "score-validation segment lengths do not match labels"
                            )
                    objective = (
                        float(np.mean(block_auprc))
                        if len(block_auprc) >= 2
                        else float(aggregate_objective)
                    )
                    candidate = {
                        "top_q": float(top_q),
                        "memory_weight": float(memory_weight),
                        "agreement_weight": float(agreement_weight),
                        "causal_smoothing": int(smoothing),
                    }
                    if best_metrics is None or float(objective) > best_objective:
                        best_params = candidate
                        best_metrics = dict(metrics)
                        best_metrics["selection_objective"] = float(objective)
                        best_metrics["selection_objective_kind"] = (
                            "macro_block_auprc"
                            if len(block_auprc) >= 2
                            else "aggregate_auprc"
                        )
                        best_metrics["eligible_validation_blocks"] = len(block_auprc)
                        best_objective = float(objective)
    if best_params is None or best_metrics is None:
        best_params = {
            "top_q": float(args.top_q),
            "memory_weight": 0.25 if use_retrieval else 0.0,
            "agreement_weight": 0.25 if use_retrieval else 0.0,
            "causal_smoothing": 1,
        }
        best_metrics = {"auprc": None, "auroc": None, "single_class": True}
    return best_params, best_metrics


def _apply_score_parameters(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    parameters: Mapping[str, Any],
    segment_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    residual, memory, agreement = components
    combined = (
        residual
        + float(parameters["memory_weight"]) * memory
        + float(parameters["agreement_weight"]) * agreement
    )
    point_scores = _top_q_mean(combined, float(parameters["top_q"]))
    return _causal_smooth_segments(
        point_scores, int(parameters["causal_smoothing"]), segment_lengths
    )


def _evaluate_fixed_score_parameters(
    labels: np.ndarray,
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    parameters: Mapping[str, Any],
    args: argparse.Namespace,
    segment_lengths: Sequence[int] | None,
) -> dict[str, Any]:
    """Evaluate one fixed post-processing setting on labeled validation.

    LoRA blends are compared with the fallback Native/SRF score parameters
    held fixed.  This makes the module delta identifiable and avoids multiplying
    the already-large score grid by the number of LoRA strengths.
    """

    scores = _apply_score_parameters(components, parameters, segment_lengths)
    metrics = dict(evaluate_scores(labels, scores))
    aggregate = metrics.get("auprc")
    block_auprc: list[float] = []
    if args.score_selection == "macro_block" and segment_lengths:
        offset = 0
        for length in segment_lengths:
            stop = offset + int(length)
            value = evaluate_scores(labels[offset:stop], scores[offset:stop]).get(
                "auprc"
            )
            if value is not None and np.isfinite(value):
                block_auprc.append(float(value))
            offset = stop
        if offset != len(labels):
            raise ValueError("score-validation segment lengths do not match labels")
    objective = (
        float(np.mean(block_auprc))
        if len(block_auprc) >= 2
        else (None if aggregate is None else float(aggregate))
    )
    metrics["selection_objective"] = objective
    metrics["selection_objective_kind"] = (
        "macro_block_auprc" if len(block_auprc) >= 2 else "aggregate_auprc"
    )
    metrics["eligible_validation_blocks"] = len(block_auprc)
    return metrics


def _run_condition(
    base: torch.nn.Module,
    base_training: Mapping[str, Any],
    base_source: str,
    prepared: PreparedDataset,
    backbone_name: str,
    condition: str,
    args: argparse.Namespace,
    seed: int,
    configuration_id: str,
) -> dict[str, Any]:
    # Identical re-seeding gives LoRA in lora/joint and fusion in srf/joint the
    # same initial weights wherever the corresponding module is active.
    seed_all(_stable_seed(seed, prepared.name, backbone_name, "condition-init"))
    model = ConditionModel(
        copy.deepcopy(base),
        condition=condition,
        rank=args.lora_rank,
        fusion_hidden=args.fusion_hidden,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    wrapper_parameters = count_parameters(model)
    adapter_parameters = model.lora_parameter_count()
    base_with_lora = sum(parameter.numel() for parameter in model.base.parameters())
    base_parameters = base_with_lora - adapter_parameters
    fusion_parameters = sum(parameter.numel() for parameter in model.fusion.parameters())
    deployed_parameters = base_parameters
    if condition in {"lora", "joint"}:
        deployed_parameters += adapter_parameters
    if condition in {"srf", "joint"}:
        deployed_parameters += fusion_parameters
    parameters = {
        **wrapper_parameters,
        "wrapper_total": int(wrapper_parameters["total"]),
        "deployed_total": int(deployed_parameters),
        "wrapper_overhead_vs_comparison": int(
            int(wrapper_parameters["total"]) - deployed_parameters
        ),
        "base": int(base_parameters),
        "adapter": int(adapter_parameters if condition in {"lora", "joint"} else 0),
        "fusion": int(fusion_parameters if condition in {"srf", "joint"} else 0),
    }
    (
        adapt_arrays,
        validation_arrays,
        score_validation_arrays,
        test_arrays,
    ) = _condition_arrays(prepared, condition)

    def train_stage(
        name: str,
        *,
        epochs: int,
        lr: float,
        stage_seed: str,
        include_initial: bool,
        amp_override: bool | None = None,
        use_srf_regularization: bool = False,
        parameter_groups: Sequence[
            tuple[str, Iterable[torch.nn.Parameter], float]
        ]
        | None = None,
    ) -> dict[str, Any]:
        metadata = train_model(
            model,
            adapt_arrays,
            validation_arrays,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=lr,
            device=args.device,
            amp=(
                _amp_for(backbone_name, args)
                if amp_override is None
                else bool(amp_override)
            ),
            patience=args.patience,
            loss=args.loss,
            seed=_stable_seed(seed, prepared.name, backbone_name, stage_seed),
            include_initial_checkpoint=include_initial,
            parameter_groups=parameter_groups,
            reference_consistency_weight=(
                args.reference_consistency_weight if use_srf_regularization else 0.0
            ),
            gate_regularization_weight=(
                args.gate_regularization_weight if use_srf_regularization else 0.0
            ),
            reference_quality_margin=args.reference_quality_margin,
            reference_quality_temperature=args.reference_quality_temperature,
        )
        metadata["stage"] = name
        metadata["learning_rate"] = float(lr)
        return metadata

    condition_training: dict[str, Any] | None = None
    if condition == "lora":
        model.configure_trainable("lora", phase="lora")
        if not args.lora_train_head:
            for parameter in model.base.forecast_head.parameters():
                parameter.requires_grad_(False)
        model.set_lora_blend(1.0)
        stage = train_stage(
            "lora_adaptation",
            epochs=args.lora_epochs,
            lr=args.lora_lr,
            stage_seed="lora-train",
            include_initial=True,
            parameter_groups=[
                ("lora", model.lora_parameters(), args.lora_lr),
                (
                    "prediction_head",
                    model.base.forecast_head.parameters(),
                    args.head_lr,
                ),
            ],
        )
        condition_training = {**stage, "stages": [stage]}
    elif condition == "srf":
        model.configure_trainable("srf", phase="srf")
        model.set_lora_blend(0.0)
        stage = train_stage(
            "srf_training",
            epochs=args.condition_epochs,
            lr=args.condition_lr,
            stage_seed="condition-train",
            include_initial=True,
            use_srf_regularization=True,
            parameter_groups=[
                ("fusion", model.fusion.parameters(), args.condition_lr),
                (
                    "prediction_head",
                    model.base.forecast_head.parameters(),
                    args.head_lr,
                ),
            ],
        )
        condition_training = {**stage, "stages": [stage]}
    elif condition == "joint":
        # Stage one intentionally uses the same initialization, phase, loader
        # seed, and learning rate as SRF-Only.  Stage two starts from that exact
        # solution; an immutable copy remains the blend-zero reference even
        # when the live LoRA/SRF/head branch is collaboratively fine-tuned.
        model.configure_trainable("joint", phase="srf")
        model.set_lora_blend(0.0)
        srf_stage = train_stage(
            "srf_warm_start",
            epochs=args.condition_epochs,
            lr=args.condition_lr,
            stage_seed="condition-train",
            include_initial=True,
            use_srf_regularization=True,
            parameter_groups=[
                ("fusion", model.fusion.parameters(), args.condition_lr),
                (
                    "prediction_head",
                    model.base.forecast_head.parameters(),
                    args.head_lr,
                ),
            ],
        )
        if args.joint_update_fusion_head:
            model.capture_srf_reference()
        model.configure_trainable(
            "joint",
            phase="joint" if args.joint_update_fusion_head else "lora",
        )
        model.set_lora_blend(1.0)
        joint_parameter_groups: list[
            tuple[str, Iterable[torch.nn.Parameter], float]
        ] = [("lora", model.lora_parameters(), args.joint_lora_lr)]
        if args.joint_update_fusion_head:
            joint_parameter_groups.extend(
                [
                    ("fusion", model.fusion.parameters(), args.joint_fusion_lr),
                    (
                        "prediction_head",
                        model.base.forecast_head.parameters(),
                        args.joint_head_lr,
                    ),
                ]
            )
        lora_stage = train_stage(
            (
                "collaborative_lora_srf_adaptation"
                if args.joint_update_fusion_head
                else "prefusion_lora_adaptation"
            ),
            epochs=args.lora_epochs,
            lr=args.joint_lora_lr,
            stage_seed="joint-lora-train",
            include_initial=True,
            use_srf_regularization=True,
            # Adapted latents feed several nonlinear selector/gate branches;
            # FP32 avoids the overflow observed on high-dynamic-range SWaT.
            amp_override=False,
            parameter_groups=joint_parameter_groups,
        )
        condition_training = {
            **lora_stage,
            "elapsed_seconds": float(srf_stage["elapsed_seconds"])
            + float(lora_stage["elapsed_seconds"]),
            "epochs_ran": int(srf_stage["epochs_ran"])
            + int(lora_stage["epochs_ran"]),
            "stages": [srf_stage, lora_stage],
            "joint_protocol": (
                "SRF warm-start, immutable SRF fallback, then low-rate "
                "LoRA+fusion+head collaboration"
                if args.joint_update_fusion_head
                else "SRF warm-start then pre-fusion LoRA"
            ),
            "exact_srf_fallback_snapshot": bool(args.joint_update_fusion_head),
        }

    use_memory_score = condition in {"srf", "joint"}
    blend_grid: list[float | None] = (
        [float(value) for value in args.lora_scale_grid]
        if condition in {"lora", "joint"}
        else [None]
    )
    blend_validation: list[dict[str, Any]] = []
    best_objective = -math.inf
    selected_blend: float | None = None
    validation_prediction: dict[str, Any] | None = None
    score_validation_prediction: dict[str, Any] | None = None
    score_parameters: dict[str, Any] | None = None
    score_validation_metrics: dict[str, Any] | None = None
    fallback_score_parameters: dict[str, Any] | None = None
    full_strength_validation_prediction: dict[str, Any] | None = None

    for blend in blend_grid:
        if blend is not None:
            model.set_lora_blend(blend)
        candidate_validation = predict_windows(
            model,
            validation_arrays,
            batch_size=args.batch_size,
            device=args.device,
            amp=_amp_for(backbone_name, args),
            return_aux=True,
            auxiliary_names=("gate",),
        )
        candidate_score_validation = predict_windows(
            model,
            score_validation_arrays,
            batch_size=args.batch_size,
            device=args.device,
            amp=_amp_for(backbone_name, args),
            return_aux=True,
            auxiliary_names=("gate",),
        )
        if blend == 1.0:
            # Retain the normal calibration prediction so the prespecified
            # full-strength adapter can be reported as a sensitivity result.
            # It is never used to select the deployed blend.
            full_strength_validation_prediction = candidate_validation
        candidate_validation_errors = (
            candidate_validation["predictions"]
            - candidate_validation["targets"]
        )
        candidate_memory = None
        candidate_agreement = None
        if use_memory_score:
            normal_distance = _aggregate_candidate_distance(
                candidate_validation["distance"]
            )
            candidate_memory = _empirical_transform(
                normal_distance,
                _aggregate_candidate_distance(
                    candidate_score_validation["distance"]
                ),
                tail_score=True,
            )
            normal_divergence = np.asarray(
                candidate_validation["divergence"], dtype=np.float64
            )
            candidate_agreement = _empirical_transform(
                normal_divergence,
                candidate_score_validation["divergence"],
                tail_score=False,
            )
        candidate_components = _score_channel_components(
            candidate_score_validation,
            candidate_validation_errors,
            len(prepared.score_validation_labels),
            candidate_memory,
            candidate_agreement,
        )
        if fallback_score_parameters is None:
            candidate_parameters, candidate_metrics = _select_score_parameters(
                prepared.score_validation_labels,
                candidate_components,
                args,
                use_memory_score,
                prepared.score_validation_segment_lengths,
            )
            fallback_score_parameters = dict(candidate_parameters)
        else:
            candidate_parameters = dict(fallback_score_parameters)
            candidate_metrics = _evaluate_fixed_score_parameters(
                prepared.score_validation_labels,
                candidate_components,
                candidate_parameters,
                args,
                prepared.score_validation_segment_lengths,
            )
        objective_value = candidate_metrics.get(
            "selection_objective", candidate_metrics.get("auprc")
        )
        objective = (
            float(objective_value)
            if objective_value is not None and np.isfinite(objective_value)
            else -math.inf
        )
        blend_validation.append(
            {
                "blend": blend,
                "selection_objective": None if not math.isfinite(objective) else objective,
                "auprc": candidate_metrics.get("auprc"),
                "auroc": candidate_metrics.get("auroc"),
                "score_parameters": candidate_parameters,
            }
        )
        # The grid is sorted; strict improvement keeps the smaller blend on a
        # numerical tie and therefore prefers the safer Native/SRF path.
        if validation_prediction is None or objective > best_objective + 1e-12:
            best_objective = objective
            selected_blend = blend
            validation_prediction = candidate_validation
            score_validation_prediction = candidate_score_validation
            score_parameters = candidate_parameters
            score_validation_metrics = candidate_metrics

    if (
        validation_prediction is None
        or score_validation_prediction is None
        or score_parameters is None
        or score_validation_metrics is None
        or not math.isfinite(best_objective)
    ):
        raise RuntimeError("LoRA blend/score validation produced no finite candidate")
    if selected_blend is not None:
        model.set_lora_blend(selected_blend)

    test_prediction = predict_windows(
        model,
        test_arrays,
        batch_size=args.batch_size,
        device=args.device,
        amp=_amp_for(backbone_name, args),
        return_aux=True,
        auxiliary_names=("gate",),
    )
    validation_errors = (
        validation_prediction["predictions"] - validation_prediction["targets"]
    )
    test_memory = None
    test_agreement = None
    if use_memory_score:
        normal_distance = _aggregate_candidate_distance(
            validation_prediction["distance"]
        )
        test_memory = _empirical_transform(
            normal_distance,
            _aggregate_candidate_distance(test_prediction["distance"]),
            tail_score=True,
        )
        normal_divergence = np.asarray(
            validation_prediction["divergence"], dtype=np.float64
        )
        test_agreement = _empirical_transform(
            normal_divergence,
            test_prediction["divergence"],
            tail_score=False,
        )
    test_components = _score_channel_components(
        test_prediction,
        validation_errors,
        len(prepared.test_labels),
        test_memory,
        test_agreement,
    )
    point_scores = _apply_score_parameters(
        test_components, score_parameters, prepared.test_segment_lengths
    )
    # Held-out labels are used only for final reporting.  A second,
    # prespecified full-strength LoRA sensitivity may be reported below, but no
    # held-out metric participates in blend or score-parameter selection.
    metrics = evaluate_scores(prepared.test_labels, point_scores)
    errors = test_prediction["predictions"] - test_prediction["targets"]
    forecast_loss = float(np.mean(np.square(errors, dtype=np.float64)))
    gate = test_prediction.get("gate")
    gate_mean = float(np.mean(gate, dtype=np.float64)) if gate is not None else None

    full_strength_test: dict[str, Any] | None = None
    if condition in {"lora", "joint"} and full_strength_validation_prediction is not None:
        if selected_blend == 1.0:
            full_strength_test = {
                "metrics": dict(metrics),
                "forecast_loss": forecast_loss,
                "gate_mean": gate_mean,
                "same_as_selected": True,
                "separate_latency_measured": False,
            }
        else:
            model.set_lora_blend(1.0)
            pure_prediction = predict_windows(
                model,
                test_arrays,
                batch_size=args.batch_size,
                device=args.device,
                amp=_amp_for(backbone_name, args),
                return_aux=True,
                auxiliary_names=("gate",),
            )
            pure_validation_errors = (
                full_strength_validation_prediction["predictions"]
                - full_strength_validation_prediction["targets"]
            )
            pure_memory = None
            pure_agreement = None
            if use_memory_score:
                pure_memory = _empirical_transform(
                    _aggregate_candidate_distance(
                        full_strength_validation_prediction["distance"]
                    ),
                    _aggregate_candidate_distance(pure_prediction["distance"]),
                    tail_score=True,
                )
                pure_agreement = _empirical_transform(
                    np.asarray(
                        full_strength_validation_prediction["divergence"],
                        dtype=np.float64,
                    ),
                    pure_prediction["divergence"],
                    tail_score=False,
                )
            pure_components = _score_channel_components(
                pure_prediction,
                pure_validation_errors,
                len(prepared.test_labels),
                pure_memory,
                pure_agreement,
            )
            pure_scores = _apply_score_parameters(
                pure_components, score_parameters, prepared.test_segment_lengths
            )
            pure_metrics = evaluate_scores(prepared.test_labels, pure_scores)
            pure_errors = pure_prediction["predictions"] - pure_prediction["targets"]
            pure_gate = pure_prediction.get("gate")
            full_strength_test = {
                "metrics": pure_metrics,
                "forecast_loss": float(
                    np.mean(np.square(pure_errors, dtype=np.float64))
                ),
                "gate_mean": (
                    float(np.mean(pure_gate, dtype=np.float64))
                    if pure_gate is not None
                    else None
                ),
                "same_as_selected": False,
                "reported_sensitivity_only": True,
                "separate_latency_measured": False,
            }
            model.set_lora_blend(float(selected_blend))

    lora_a_norm = math.sqrt(
        sum(
            float(parameter.detach().float().square().sum().cpu())
            for name, parameter in model.named_parameters()
            if name.endswith("lora_A")
        )
    )
    lora_b_norm = math.sqrt(
        sum(
            float(parameter.detach().float().square().sum().cpu())
            for name, parameter in model.named_parameters()
            if name.endswith("lora_B")
        )
    )
    lora_target_prefixes = sorted(
        {
            name.rsplit(".lora_", 1)[0]
            for name, _parameter in model.named_parameters()
            if name.endswith(".lora_A") or name.endswith(".lora_B")
        }
    )

    # Recount after training because collaborative Joint materialises a frozen
    # SRF fallback snapshot and because the active trainable set is phase-
    # dependent.  ``deployed_total`` remains the logical paper-condition count;
    # runtime/snapshot overhead is reported separately and never hidden.
    runtime_parameters = count_parameters(model)
    stage_trainable_parameters: list[dict[str, Any]] = []
    if condition_training is not None:
        for stage_metadata in condition_training.get("stages", []):
            groups = stage_metadata.get("optimizer_groups", [])
            stage_trainable_parameters.append(
                {
                    "stage": stage_metadata.get("stage"),
                    "parameter_count": int(
                        sum(int(group.get("parameter_count", 0)) for group in groups)
                    ),
                }
            )
    snapshot_parameters = (
        sum(parameter.numel() for parameter in model.srf_reference_fusion.parameters())
        if model.srf_reference_fusion is not None
        else 0
    )
    snapshot_buffers = sum(
        int(buffer.numel())
        for buffer in (
            model.srf_reference_head_weight,
            model.srf_reference_head_bias,
        )
        if buffer is not None
    )
    parameters.update(runtime_parameters)
    parameters.update(
        {
            "wrapper_total": int(runtime_parameters["total"]),
            "runtime_wrapper_total": int(runtime_parameters["total"]),
            "comparison_deployed_total": int(deployed_parameters),
            "wrapper_overhead_vs_comparison": int(
                int(runtime_parameters["total"]) - deployed_parameters
            ),
            "srf_fallback_snapshot_parameters": int(snapshot_parameters),
            "srf_fallback_snapshot_buffers": int(snapshot_buffers),
            "stage_trainable_parameters": stage_trainable_parameters,
            "peak_trainable_parameters": int(
                max(
                    [item["parameter_count"] for item in stage_trainable_parameters]
                    or [int(runtime_parameters["trainable"])]
                )
            ),
        }
    )

    test_windows = len(test_prediction["predictions"])
    retrieval_seconds = prepared.retrieval_seconds["test"] if use_memory_score else 0.0
    model_seconds = float(test_prediction["elapsed_seconds"])
    total_seconds = model_seconds + retrieval_seconds
    latency = {
        "model_seconds": model_seconds,
        "inference_pipeline_seconds": model_seconds,
        "retrieval_seconds": retrieval_seconds,
        "total_seconds": total_seconds,
        "model_ms_per_window": 1000.0 * model_seconds / test_windows,
        "retrieval_ms_per_window": 1000.0 * retrieval_seconds / test_windows,
        "total_ms_per_window": 1000.0 * total_seconds / test_windows,
        "end_to_end_windows_per_second": test_windows / max(total_seconds, np.finfo(float).eps),
        "retrieval_is_shared_precomputation": True,
        "measurement_scope": (
            "predict_windows pipeline: DataLoader, host-to-device transfer, "
            "forward, finite checks, and device-to-host collection"
        ),
        "variant": "validation_selected_blend",
        "selected_lora_blend": selected_blend,
    }
    return {
        "configuration_id": configuration_id,
        "timestamp_utc": _utc_now(),
        "status": "ok",
        "dataset": prepared.name,
        "backbone": backbone_name,
        "condition": condition,
        "seed": seed,
        "metrics": metrics,
        "forecast_loss": forecast_loss,
        "gate_mean": gate_mean,
        "parameters": parameters,
        "latency": latency,
        "test_windows": test_windows,
        "base_checkpoint_source": base_source,
        "base_training": dict(base_training),
        "condition_training": condition_training,
        "lora": {
            "implementation": "weight-level LoRA on architecture-specific frozen linear maps",
            "joint_integration": "pre-fusion adapted latent and forecast",
            "rank": int(args.lora_rank),
            "alpha": float(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "initialization": "A=Kaiming-uniform, B=zeros",
            "target_parameter_prefixes": lora_target_prefixes,
            "selected_blend": selected_blend,
            "blend_validation": blend_validation,
            "a_l2_norm": lora_a_norm,
            "b_l2_norm": lora_b_norm,
            "head_trained_in_lora_only": bool(args.lora_train_head),
            "full_strength_test": full_strength_test,
        },
        "amp_enabled": bool(test_prediction["amp_enabled"]),
        "score": {
            "residual": "normal-validation median/MAD calibrated absolute error",
            "memory_distance_calibration": "normal-validation empirical upper-tail score",
            "reference_agreement": "1 minus normal-validation empirical divergence CDF",
            "selected_parameters": score_parameters,
            "labeled_validation_metrics": score_validation_metrics,
            "lora_blend_selected_on_labeled_validation": selected_blend,
            "shared_fallback_score_parameters_for_lora_blends": bool(
                condition in {"lora", "joint"}
            ),
            "point_adjustment": False,
            "test_label_tuning": False,
        },
    }


def _error_row(
    configuration_id: str,
    dataset: str,
    backbone: str,
    condition: str,
    seed: int,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "timestamp_utc": _utc_now(),
        "status": "error",
        "dataset": dataset,
        "backbone": backbone,
        "condition": condition,
        "seed": seed,
        "error": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
    }


def _pending_conditions(
    configuration_id: str,
    dataset: str,
    backbone: str,
    seed: int,
    conditions: Sequence[str],
    completed: set[tuple[str, str, str, str, int]],
) -> list[str]:
    return [
        condition
        for condition in conditions
        if (configuration_id, dataset, backbone, condition, seed) not in completed
    ]


def _environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _write_manifest(
    path: Path,
    configuration_id: str,
    configuration: Mapping[str, Any],
    args: argparse.Namespace,
    dataset_metadata: Mapping[str, Any],
    results_path: Path,
) -> None:
    project_dir = Path(__file__).resolve().parent
    code_files = ("run_main.py", "data_pipeline.py", "models.py", "training.py", "full_data.py")
    rows = _read_jsonl(results_path)
    current_rows = [row for row in rows if row.get("configuration_id") == configuration_id]
    latest = {_result_key(row): row for row in current_rows}
    manifest = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "configuration_id": configuration_id,
        "configuration": dict(configuration),
        "data_root": str(Path(args.data_root).expanduser()),
        "output_dir": str(Path(args.output_dir).expanduser()),
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": _environment(),
        "code_sha256": {
            name: _file_sha256(project_dir / name) for name in code_files
        },
        "guardrails": {
            "clean_room": True,
            "existing_project_code_loaded": False,
            "existing_project_configuration_loaded": False,
            "existing_project_weights_loaded": False,
            "labels_used_for_model_training": False,
            "labeled_validation_labels_used_for_score_tuning": True,
            "labeled_validation_labels_used_for_lora_blend_selection": True,
            "heldout_test_labels_used_for_tuning": False,
            "heldout_test_labels_used_only_for_final_metrics": True,
            "point_adjustment": False,
        },
        "planned_runs": len(args.datasets)
        * len(args.backbones)
        * len(args.conditions)
        * len(args.seeds),
        "recorded_runs": len(latest),
        "successful_runs": sum(row.get("status") == "ok" for row in latest.values()),
        "error_runs": sum(row.get("status") == "error" for row in latest.values()),
        "dataset_metadata": dict(dataset_metadata),
        "artifacts": {
            "results_jsonl": str(results_path),
            "summary_csv": str(Path(args.output_dir) / "summary.csv"),
            "checkpoint_dir": str(
                Path(args.base_checkpoint_dir).expanduser()
                if args.base_checkpoint_dir
                else Path(args.output_dir) / "checkpoints"
            ),
        },
    }
    save_json(manifest, path)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"
    checkpoint_dir = (
        Path(args.base_checkpoint_dir).expanduser().resolve()
        if args.base_checkpoint_dir
        else output_dir / "checkpoints"
    )

    configuration = _configuration(args)
    base_configuration_id = _configuration_id(_base_configuration(configuration))
    configuration["base_configuration_id"] = base_configuration_id
    configuration["base_checkpoint_dir"] = str(checkpoint_dir)
    configuration_id = _configuration_id(configuration)
    existing_rows = _read_jsonl(results_path)
    if args.retry_errors:
        completed = {
            _result_key(row)
            for row in existing_rows
            if row.get("status") == "ok"
        }
    else:
        completed = {_result_key(row) for row in existing_rows}
    dataset_metadata: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous_manifest.get("configuration_id") == configuration_id:
                previous_metadata = previous_manifest.get("dataset_metadata", {})
                if isinstance(previous_metadata, Mapping):
                    dataset_metadata.update(previous_metadata)
        except (OSError, json.JSONDecodeError):
            # Results JSONL remains the source of truth; a damaged manifest is
            # safely reconstructed as the run progresses.
            pass

    _write_manifest(
        manifest_path,
        configuration_id,
        configuration,
        args,
        dataset_metadata,
        results_path,
    )
    print(
        f"configuration={configuration_id} planned="
        f"{len(args.datasets) * len(args.backbones) * len(args.conditions) * len(args.seeds)}"
    )

    for seed in args.seeds:
        for dataset_name in args.datasets:
            dataset_pending = any(
                _pending_conditions(
                    configuration_id,
                    dataset_name,
                    backbone,
                    seed,
                    args.conditions,
                    completed,
                )
                for backbone in args.backbones
            )
            if not dataset_pending:
                print(f"skip completed dataset={dataset_name} seed={seed}")
                continue
            need_retrieval = any(
                condition in {"srf", "joint"} for condition in args.conditions
            )
            try:
                print(f"prepare dataset={dataset_name} seed={seed}", flush=True)
                prepared = _prepare_dataset(dataset_name, args, seed, need_retrieval)
                dataset_metadata[
                    f"{dataset_name}/crop-seed-{args.crop_seed}"
                ] = prepared.metadata
                _write_manifest(
                    manifest_path,
                    configuration_id,
                    configuration,
                    args,
                    dataset_metadata,
                    results_path,
                )
            except Exception as exc:
                print(f"ERROR data dataset={dataset_name} seed={seed}: {exc}", file=sys.stderr)
                for backbone_name in args.backbones:
                    for condition in _pending_conditions(
                        configuration_id,
                        dataset_name,
                        backbone_name,
                        seed,
                        args.conditions,
                        completed,
                    ):
                        row = _error_row(
                            configuration_id,
                            dataset_name,
                            backbone_name,
                            condition,
                            seed,
                            "dataset_preparation",
                            exc,
                        )
                        _append_jsonl(results_path, row)
                        completed.add(_result_key(row))
                _write_summary(results_path, summary_path)
                _write_manifest(
                    manifest_path,
                    configuration_id,
                    configuration,
                    args,
                    dataset_metadata,
                    results_path,
                )
                continue

            for backbone_name in args.backbones:
                pending = _pending_conditions(
                    configuration_id,
                    dataset_name,
                    backbone_name,
                    seed,
                    args.conditions,
                    completed,
                )
                if not pending:
                    continue
                try:
                    print(
                        f"base dataset={dataset_name} backbone={backbone_name} seed={seed}",
                        flush=True,
                    )
                    base, base_training, base_source = _fit_or_load_base(
                        prepared,
                        backbone_name,
                        args,
                        seed,
                        base_configuration_id,
                        checkpoint_dir,
                    )
                except Exception as exc:
                    print(
                        f"ERROR base dataset={dataset_name} backbone={backbone_name}: {exc}",
                        file=sys.stderr,
                    )
                    for condition in pending:
                        row = _error_row(
                            configuration_id,
                            dataset_name,
                            backbone_name,
                            condition,
                            seed,
                            "base_training",
                            exc,
                        )
                        _append_jsonl(results_path, row)
                        completed.add(_result_key(row))
                    _write_summary(results_path, summary_path)
                    _write_manifest(
                        manifest_path,
                        configuration_id,
                        configuration,
                        args,
                        dataset_metadata,
                        results_path,
                    )
                    continue

                for condition in pending:
                    try:
                        print(
                            f"run dataset={dataset_name} backbone={backbone_name} "
                            f"condition={condition} seed={seed}",
                            flush=True,
                        )
                        row = _run_condition(
                            base,
                            base_training,
                            base_source,
                            prepared,
                            backbone_name,
                            condition,
                            args,
                            seed,
                            configuration_id,
                        )
                    except Exception as exc:
                        print(
                            f"ERROR condition dataset={dataset_name} backbone={backbone_name} "
                            f"condition={condition}: {exc}",
                            file=sys.stderr,
                        )
                        row = _error_row(
                            configuration_id,
                            dataset_name,
                            backbone_name,
                            condition,
                            seed,
                            "condition_training_or_evaluation",
                            exc,
                        )
                    _append_jsonl(results_path, row)
                    completed.add(_result_key(row))
                    _write_summary(results_path, summary_path)
                    _write_manifest(
                        manifest_path,
                        configuration_id,
                        configuration,
                        args,
                        dataset_metadata,
                        results_path,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    _write_summary(results_path, summary_path)
    _write_manifest(
        manifest_path,
        configuration_id,
        configuration,
        args,
        dataset_metadata,
        results_path,
    )
    latest = {
        _result_key(row): row
        for row in _read_jsonl(results_path)
        if row.get("configuration_id") == configuration_id
    }
    failures = sum(row.get("status") != "ok" for row in latest.values())
    print(
        f"finished configuration={configuration_id} recorded={len(latest)} "
        f"failures={failures} summary={summary_path}",
        flush=True,
    )
    return 1 if failures else 0


def _self_test() -> int:
    """Exercise the real model/data/training interfaces without external files."""

    seed_all(17)
    n_windows, channels = 12, 3
    contexts = np.random.default_rng(17).normal(
        size=(n_windows, CONTEXT_LENGTH, channels)
    ).astype(np.float32)
    futures = (
        contexts[:, -1:, :]
        + np.linspace(0.0, 0.2, FORECAST_HORIZON, dtype=np.float32)[None, :, None]
    )
    indices = CONTEXT_LENGTH + np.arange(n_windows, dtype=np.int64) * 16
    memory = ChannelMemoryBank(memory_per_channel=8, top_k=3, selection="farthest")
    memory.fit(contexts[:8], futures[:8], end_indices=indices[:8])
    retrieval = memory.query(contexts)
    global_memory = GlobalMemoryBank(
        memory_per_channel=8, top_k=3, selection="farthest"
    )
    global_memory.fit(contexts[:8], futures[:8], end_indices=indices[:8])
    global_retrieval = global_memory.query(contexts, return_candidates=True)
    if global_retrieval[0].shape != (
        n_windows,
        3,
        FORECAST_HORIZON,
        channels,
    ):
        raise AssertionError(
            f"global retrieval candidate shape mismatch: {global_retrieval[0].shape}"
        )
    plain = NumpyWindowDataset(contexts, futures, end_indices=indices)
    augmented = _window_dataset((contexts, futures, indices), retrieval)

    # A finite value at the end of one disjoint block must never fill the NaN
    # prefix of the next block, regardless of the selected smoothing width.
    smoothing_probe = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 9.0])
    smoothed_probe = _causal_smooth_segments(smoothing_probe, 3, (3, 3))
    if np.isfinite(smoothed_probe[3:5]).any() or smoothed_probe[5] != 9.0:
        raise AssertionError(f"segment-aware smoothing failed: {smoothed_probe}")

    shapes: dict[str, Any] = {}
    for backbone_name in DEFAULT_BACKBONES:
        base = build_backbone(
            backbone_name,
            channels,
            CONTEXT_LENGTH,
            FORECAST_HORIZON,
            d_model=8,
        )
        for condition in DEFAULT_CONDITIONS:
            model = ConditionModel(copy.deepcopy(base), condition=condition, rank=2, fusion_hidden=4)
            arrays = augmented if condition in {"srf", "joint"} else plain
            output = predict_windows(
                model,
                arrays,
                batch_size=4,
                device="cpu",
                amp=False,
                return_aux=True,
                auxiliary_names=("gate",),
            )
            expected = (n_windows, FORECAST_HORIZON, channels)
            if output["predictions"].shape != expected:
                raise AssertionError(
                    f"{backbone_name}/{condition}: expected {expected}, "
                    f"got {output['predictions'].shape}"
                )
            if condition in {"srf", "joint"} and "gate" not in output:
                raise AssertionError(f"{backbone_name}/{condition}: gate missing")
            shapes[f"{backbone_name}/{condition}"] = list(output["predictions"].shape)

    # Exercise optimizer/early-stopping plumbing on a tiny trainable model.
    trainable = build_backbone(
        "patch_transformer",
        channels,
        CONTEXT_LENGTH,
        FORECAST_HORIZON,
        d_model=8,
    )
    training = train_model(
        trainable,
        NumpyWindowDataset(contexts[:8], futures[:8]),
        NumpyWindowDataset(contexts[8:], futures[8:]),
        epochs=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        amp=False,
        patience=1,
        seed=17,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "interfaces": shapes,
                "training_best_val_loss": training["best_val_loss"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-data SRF matrix."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "RAG_TSAD_DATA_ROOT", "./data"
        ),
        help="Directory containing raw PSM/SMD dataset files (data only).",
    )
    parser.add_argument("--output-dir", default="outputs/main")
    parser.add_argument(
        "--base-checkpoint-dir",
        default=None,
        help=(
            "Optional shared cache for fitted base checkpoints.  Cache keys "
            "exclude LoRA/SRF-only hyperparameters."
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--backbones", nargs="+", choices=DEFAULT_BACKBONES, default=list(DEFAULT_BACKBONES)
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=DEFAULT_CONDITIONS, default=list(DEFAULT_CONDITIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--crop-seed",
        type=int,
        default=20_260_923,
        help="Fixed seed for contiguous crops and window subsampling across all run seeds.",
    )
    parser.add_argument(
        "--crop-mode", choices=("random", "head", "tail"), default="random"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--labeled-validation-fraction", type=float, default=0.20)
    parser.add_argument("--score-validation-block-size", type=int, default=2048)
    parser.add_argument("--max-train-points", type=int, default=0)
    parser.add_argument("--max-test-points", type=int, default=0)
    parser.add_argument("--base-max-windows", type=int, default=0)
    parser.add_argument("--condition-max-windows", type=int, default=0)
    parser.add_argument("--validation-max-windows", type=int, default=0)
    parser.add_argument("--memory-candidate-max-windows", type=int, default=0)
    parser.add_argument("--memory-per-channel", type=int, default=256)
    parser.add_argument(
        "--retrieval-mode",
        choices=("channel", "global"),
        default="channel",
        help=(
            "Retrieve independent neighbours per variable or one shared set of "
            "multivariate windows. The global mode is an ablation baseline."
        ),
    )
    parser.add_argument(
        "--memory-selection",
        choices=("uniform", "farthest", "boundary"),
        default="boundary",
    )
    parser.add_argument(
        "--memory-source",
        choices=("base_memory", "dedicated"),
        default="base_memory",
        help=(
            "Use the first 80% normal history or only the disjoint 10% memory "
            "block when constructing retrieval prototypes."
        ),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha; the injected update is scaled by alpha/rank.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-epochs",
        type=int,
        default=20,
        help="Epoch budget for LoRA-only and Joint's second adaptation stage.",
    )
    parser.add_argument("--lora-lr", type=float, default=1e-3)
    parser.add_argument(
        "--joint-lora-lr",
        type=float,
        default=2e-4,
        help="Conservative FP32 LoRA rate for Joint's post-SRF adaptation stage.",
    )
    parser.add_argument(
        "--joint-update-fusion-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "During Joint stage two, also update the live SRF and prediction "
            "head at smaller rates while retaining an immutable SRF fallback."
        ),
    )
    parser.add_argument("--joint-fusion-lr", type=float, default=5e-5)
    parser.add_argument("--joint-head-lr", type=float, default=2e-5)
    parser.add_argument(
        "--lora-scale-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help=(
            "Validation-only blend between the exact Native/SRF safety path "
            "and the adapted LoRA path."
        ),
    )
    parser.add_argument(
        "--lora-train-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also update the prediction head in LoRA-only, matching the manuscript.",
    )
    parser.add_argument("--fusion-hidden", type=int, default=32)
    parser.add_argument("--base-epochs", type=int, default=25)
    parser.add_argument("--condition-epochs", type=int, default=15)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--condition-lr", type=float, default=5e-4)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Separate conservative learning rate for the full-rank prediction head.",
    )
    parser.add_argument(
        "--reference-consistency-weight",
        type=float,
        default=0.0,
        help="Weight of the quality-masked SRF analogue consistency loss.",
    )
    parser.add_argument(
        "--gate-regularization-weight",
        type=float,
        default=0.0,
        help="Weight of the quality-aware SRF gate calibration loss.",
    )
    parser.add_argument("--reference-quality-margin", type=float, default=0.10)
    parser.add_argument("--reference-quality-temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--memory-temperature", type=float, default=0.15)
    parser.add_argument("--top-q", type=float, default=0.2)
    parser.add_argument(
        "--score-selection",
        choices=("aggregate", "macro_block"),
        default="aggregate",
        help=(
            "Select score hyperparameters by AUPRC on the complete labeled "
            "validation partition (paper protocol) or by macro block AUPRC."
        ),
    )
    parser.add_argument("--loss", choices=("mse", "huber"), default="huber")
    parser.add_argument(
        "--score-memory-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--score-agreement-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--score-top-q-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.4, 1.0],
    )
    parser.add_argument(
        "--score-smoothing-grid",
        nargs="+",
        type=int,
        default=[1, 3, 5, 9],
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.max_train_points = min(args.max_train_points or 2400, 2_400)
        args.max_test_points = min(args.max_test_points or 1600, 1_600)
        args.base_max_windows = min(args.base_max_windows or 256, 256)
        args.condition_max_windows = min(args.condition_max_windows or 192, 192)
        args.validation_max_windows = min(args.validation_max_windows or 128, 128)
        args.memory_candidate_max_windows = min(args.memory_candidate_max_windows or 256, 256)
        args.base_epochs = min(args.base_epochs, 2)
        args.condition_epochs = min(args.condition_epochs, 2)
        args.lora_epochs = min(args.lora_epochs, 2)
        args.patience = min(args.patience, 1)
        args.d_model = min(args.d_model, 16)
        args.batch_size = min(args.batch_size, 32)
        args.context_length = min(args.context_length, 48)
        args.forecast_horizon = min(args.forecast_horizon, 8)
        args.test_stride = min(args.test_stride, 4)
        args.memory_per_channel = min(args.memory_per_channel, 32)
        args.retrieval_top_k = min(args.retrieval_top_k, 3)
        args.score_validation_block_size = min(args.score_validation_block_size, 400)

    if not args.self_test:
        args.datasets = discover_datasets(args.data_root, args.datasets)
    for limit in ("max_train_points", "max_test_points", "base_max_windows", "condition_max_windows", "validation_max_windows", "memory_candidate_max_windows"):
        if getattr(args, limit) < 0:
            parser.error(f"--{limit.replace('_', '-')} cannot be negative")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets contains duplicates")
    if len(set(args.backbones)) != len(args.backbones):
        parser.error("--backbones contains duplicates")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions contains duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicates")
    positive_names = (
        "d_model",
        "lora_rank",
        "lora_epochs",
        "fusion_hidden",
        "base_epochs",
        "condition_epochs",
        "batch_size",
        "context_length",
        "forecast_horizon",
        "test_stride",
        "memory_per_channel",
        "retrieval_top_k",
        "score_validation_block_size",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    if (
        args.base_lr <= 0
        or args.condition_lr <= 0
        or args.lora_lr <= 0
        or args.joint_lora_lr <= 0
        or args.joint_fusion_lr <= 0
        or args.joint_head_lr <= 0
        or args.head_lr <= 0
    ):
        parser.error("learning rates must be positive")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must lie in [0, 1)")
    if args.reference_consistency_weight < 0 or args.gate_regularization_weight < 0:
        parser.error("SRF auxiliary-loss weights must be non-negative")
    if args.reference_quality_temperature <= 0:
        parser.error("--reference-quality-temperature must be positive")
    if not args.lora_scale_grid:
        parser.error("--lora-scale-grid must not be empty")
    if len(set(args.lora_scale_grid)) != len(args.lora_scale_grid):
        parser.error("--lora-scale-grid contains duplicates")
    if any(value < 0.0 or value > 1.0 for value in args.lora_scale_grid):
        parser.error("--lora-scale-grid values must lie in [0, 1]")
    args.lora_scale_grid = sorted(float(value) for value in args.lora_scale_grid)
    if 0.0 not in args.lora_scale_grid:
        parser.error("--lora-scale-grid must include 0 for the Native/SRF fallback")
    if 1.0 not in args.lora_scale_grid:
        parser.error(
            "--lora-scale-grid must include 1 to report full-strength LoRA"
        )
    if args.memory_temperature <= 0:
        parser.error("--memory-temperature must be positive")
    if not 0.05 <= args.labeled_validation_fraction <= 0.5:
        parser.error("--labeled-validation-fraction must lie in [0.05, 0.5]")
    if not 0 < args.top_q <= 1:
        parser.error("--top-q must lie in (0, 1]")
    if any(not 0 < value <= 1 for value in args.score_top_q_grid):
        parser.error("--score-top-q-grid values must lie in (0,1]")
    if any(value < 0 for value in args.score_memory_grid + args.score_agreement_grid):
        parser.error("score weights must be non-negative")
    if any(value <= 0 for value in args.score_smoothing_grid):
        parser.error("score smoothing widths must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
