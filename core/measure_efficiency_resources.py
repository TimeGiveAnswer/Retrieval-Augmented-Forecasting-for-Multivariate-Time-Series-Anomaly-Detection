"""Independent resource microbenchmark for the clean-room TSAD reproduction.

This script deliberately does not change or train the experiment implementation.
It combines two kinds of evidence that must not be conflated:

* parameter counts and training wall times are read from completed ``run_main``
  result rows; and
* latency, throughput, CUDA peak allocation, and retained-memory bytes are
  measured in a reproducible PSM-shaped PatchTST microbenchmark.

The model weights used by the microbenchmark are deterministically initialised.
Weights do not change tensor shapes or the exact retrieval algorithm, so this is
appropriate for resource measurement, but the output is not a model-accuracy
result.  ``--mode real`` uses the same PSM crop/scaler/memory construction as the
main experiment.  ``--mode synthetic`` is an explicit smoke-test mode, while
``--mode static`` only materialises fields already present in result JSONL files.

Progress is appended to ``resource_measurements.jsonl`` after every completed
condition or sensitivity point.  Rerunning the same benchmark id resumes it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from data_pipeline import ChannelMemoryBank, RobustScaler, load_series, make_windows
from models import (
    ConditionModel,
    ForecastBackbone,
    SelectiveRetrievalFusion,
    build_backbone,
)


CONDITIONS = ("native", "lora", "srf", "joint")
CONDITION_LABELS = {
    "native": "Native",
    "lora": "LoRA-Only",
    "srf": "SRF-Only",
    "joint": "SRF+LoRA",
}
RETRIEVAL_CONDITIONS = frozenset({"srf", "joint"})

RESOURCE_COLUMNS = (
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
    "retrieval_top_k",
    "memory_per_channel",
    "deployed_lora_blend",
    "deployed_parameters",
    "adaptation_trainable_parameters",
    "deployed_parameter_bytes_fp32",
    "memory_bank_payload_bytes",
    "memory_bank_object_bytes_approx",
    "psm_base_training_seconds_observed",
    "psm_adaptation_training_seconds_observed",
    "psm_total_training_seconds_observed",
    "five_family_mean_total_training_seconds_observed",
    "seven_physical_median_total_training_seconds_observed",
    "single_model_latency_ms_median",
    "single_retrieval_latency_ms_median",
    "single_e2e_latency_ms_median",
    "single_e2e_latency_ms_p95",
    "batch_size_for_throughput",
    "batch_model_windows_per_second",
    "batch_retrieval_windows_per_second",
    "batch_e2e_windows_per_second",
    "cuda_inference_peak_allocated_mib_batch1",
    "cuda_inference_peak_allocated_mib_batch",
    "cuda_inference_peak_reserved_mib_batch",
    "cuda_training_peak_allocated_mib_batch",
    "microbenchmark_weights",
    "timing_scope",
)

SENSITIVITY_COLUMNS = (
    "benchmark_id",
    "measurement_status",
    "dataset",
    "backbone",
    "condition",
    "seed",
    "device",
    "channels",
    "context_length",
    "forecast_horizon",
    "retrieval_top_k",
    "memory_per_channel",
    "memory_candidates",
    "memory_fit_seconds",
    "memory_bank_payload_bytes",
    "memory_bank_object_bytes_approx",
    "single_retrieval_latency_ms_median",
    "single_model_latency_ms_median",
    "single_e2e_latency_ms_median",
    "batch_size_for_throughput",
    "batch_retrieval_windows_per_second",
    "batch_e2e_windows_per_second",
)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    return (int(seed) ^ (zlib.crc32(payload) & 0x7FFFFFFF)) & 0x7FFFFFFF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _discover_results(explicit: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        candidates = list(explicit)
    else:
        candidates = [Path("outputs/main/results.jsonl")]
    for candidate in candidates:
        path = candidate / "results.jsonl" if candidate.is_dir() else candidate
        if path.exists() and path not in paths:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(
            "No result JSONL files found; pass --results or run from the project root"
        )
    return paths


def _read_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                key = (
                    str(row.get("dataset")),
                    str(row.get("backbone")),
                    str(row.get("condition")),
                    int(row.get("seed", -1)),
                )
                if key in latest:
                    raise ValueError(f"duplicate result {key} at {path}:{line_number}")
                latest[key] = row
    return list(latest.values())


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _family(dataset: str) -> str:
    return "SMD" if dataset.startswith("SMD-machine-") else dataset.split("@", 1)[0]


def _finite_mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else None


def _observed_summaries(
    rows: Sequence[dict[str, Any]], seed: int
) -> dict[str, dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("backbone") == "patch_transformer" and int(row.get("seed", -1)) == seed
    ]
    output: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        current = [row for row in selected if row.get("condition") == condition]
        psm = [row for row in current if row.get("dataset") == "PSM"]
        if len(psm) != 1:
            raise ValueError(f"expected one PSM/PatchTST/{condition}/seed{seed} result, got {len(psm)}")
        psm_row = psm[0]

        total_by_physical: list[float] = []
        total_by_family: defaultdict[str, list[float]] = defaultdict(list)
        for row in current:
            base = _nested(row, "base_training", "elapsed_seconds")
            adaptation = _nested(row, "condition_training", "elapsed_seconds")
            if not isinstance(base, (int, float)):
                continue
            adaptation_value = 0.0 if condition == "native" else float(adaptation)
            total = float(base) + adaptation_value
            total_by_physical.append(total)
            total_by_family[_family(str(row["dataset"]))].append(total)
        family_means = [statistics.fmean(values) for values in total_by_family.values()]

        psm_base = float(_nested(psm_row, "base_training", "elapsed_seconds"))
        psm_adaptation = (
            0.0
            if condition == "native"
            else float(_nested(psm_row, "condition_training", "elapsed_seconds"))
        )
        output[condition] = {
            "deployed_parameters": int(_nested(psm_row, "parameters", "deployed_total")),
            "adaptation_trainable_parameters": int(_nested(psm_row, "parameters", "trainable")),
            "psm_base_training_seconds_observed": psm_base,
            "psm_adaptation_training_seconds_observed": psm_adaptation,
            "psm_total_training_seconds_observed": psm_base + psm_adaptation,
            "five_family_mean_total_training_seconds_observed": (
                statistics.fmean(family_means) if family_means else None
            ),
            "seven_physical_median_total_training_seconds_observed": (
                statistics.median(total_by_physical) if total_by_physical else None
            ),
            "n_families_for_training_summary": len(family_means),
            "n_physical_for_training_summary": len(total_by_physical),
            "deployed_lora_blend": (
                _nested(psm_row, "lora", "selected_blend")
                if condition in {"lora", "joint"}
                else None
            ),
        }
    return output


def _synthetic_series(points: int, channels: int, seed: int) -> np.ndarray:
    if points < 1024:
        raise ValueError("--synthetic-points must be at least 1024")
    rng = np.random.default_rng(seed)
    time_axis = np.arange(points, dtype=np.float64)
    values = np.empty((points, channels), dtype=np.float64)
    for channel in range(channels):
        period = 31.0 + 7.0 * (channel % 11)
        slow_period = 281.0 + 13.0 * (channel % 7)
        signal = np.sin(2.0 * np.pi * time_axis / period + 0.1 * channel)
        signal += 0.35 * np.cos(2.0 * np.pi * time_axis / slow_period)
        noise = rng.normal(0.0, 0.08 + 0.002 * channel, size=points)
        values[:, channel] = signal + noise + 0.00002 * (channel + 1) * time_axis
    return values.astype(np.float32)


def _prepare_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if args.mode == "real":
        train, _, _, metadata = load_series(
            "PSM",
            args.data_root,
            max_train_points=args.max_train_points,
            max_test_points=args.max_test_points,
            seed=args.crop_seed,
            crop_mode="random",
        )
        source_kind = "PSM train crop"
    else:
        train = _synthetic_series(args.synthetic_points, args.channels, args.seed)
        metadata = {"dataset": "synthetic-PSM-shape", "channels": args.channels}
        source_kind = "deterministic synthetic PSM-shape smoke input"

    channels = int(train.shape[1])
    if args.mode == "real" and channels != args.channels:
        raise ValueError(f"PSM has {channels} channels, expected --channels={args.channels}")
    scaler = RobustScaler().fit(train)
    scaled = scaler.transform(train)
    memory_stop = int(math.floor(0.80 * len(scaled)))
    query_stop = int(math.floor(0.90 * len(scaled)))
    memory_values = scaled[:memory_stop]
    query_values = scaled[memory_stop:query_stop]
    memory_windows = make_windows(
        memory_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=(None if args.memory_candidate_windows == 0 else min(args.memory_candidate_windows, len(memory_values) - args.context_length - args.forecast_horizon + 1)),
        seed=_stable_seed(args.crop_seed, "PSM", "memory-windows"),
    )
    query_windows = make_windows(
        query_values,
        args.context_length,
        args.forecast_horizon,
        stride=1,
        max_windows=max(args.throughput_batch_size, args.query_windows),
        seed=_stable_seed(args.crop_seed, "PSM", "resource-query-windows"),
    )
    preparation = {
        "source_kind": source_kind,
        "train_points": int(len(train)),
        "channels": channels,
        "memory_source_points": int(len(memory_values)),
        "memory_candidates": int(len(memory_windows[0])),
        "query_windows": int(len(query_windows[0])),
        "source_metadata": metadata,
    }
    return memory_windows, query_windows[0], query_windows[1], preparation


def _array_payload_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_array_payload_bytes(key, seen) + _array_payload_bytes(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(_array_payload_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return _array_payload_bytes(vars(value), seen)
    return 0


def _deep_object_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Approximate retained object footprint including NumPy-owned buffers.

    CPython's ``ndarray.__sizeof__`` already includes an owning array's data
    buffer.  Returning ``sys.getsizeof`` directly avoids double-counting the
    payload that is also reported separately by :func:`_array_payload_bytes`.
    """
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return int(size)
    if isinstance(value, Mapping):
        return int(size + sum(_deep_object_bytes(key, seen) + _deep_object_bytes(item, seen) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        return int(size + sum(_deep_object_bytes(item, seen) for item in value))
    if hasattr(value, "__dict__"):
        return int(size + _deep_object_bytes(vars(value), seen))
    return int(size)


def _fit_memory(
    memory_windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    memory_per_channel: int,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[ChannelMemoryBank, float]:
    start = time.perf_counter()
    bank = ChannelMemoryBank(
        memory_per_channel=memory_per_channel,
        top_k=top_k,
        temperature=args.memory_temperature,
        selection=args.memory_selection,
    ).fit(memory_windows[0], memory_windows[1], end_indices=memory_windows[2])
    return bank, time.perf_counter() - start


class _RetrievalDeploymentModel(nn.Module):
    """Condition-specific SRF deployment graph without dormant LoRA modules.

    ``ConditionModel`` intentionally contains every optional component so that
    one training implementation can run all four ablations.  Resource results,
    however, report only the components deployed by a condition.  Building SRF
    directly from the fitted backbone and fusion module keeps both the counted
    parameters and the timed forward graph faithful to ``base + SRF``.
    """

    def __init__(
        self,
        base: ForecastBackbone,
        *,
        fusion_hidden: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.fusion = SelectiveRetrievalFusion(
            base.d_model,
            base.horizon,
            hidden=fusion_hidden,
            initial_gate=0.10,
        )
        self.configure_trainable()

    def configure_trainable(self) -> "_RetrievalDeploymentModel":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(True)
        for parameter in self.base.forecast_head.parameters():
            parameter.requires_grad_(True)
        self.base.eval()
        return self

    def train(self, mode: bool = True) -> "_RetrievalDeploymentModel":
        super().train(mode)
        # The adaptation protocol freezes the fitted backbone.  Keeping it in
        # eval mode also prevents PatchTST dropout from adding timing noise.
        self.base.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        analog_future: torch.Tensor,
        distance: torch.Tensor,
        divergence: torch.Tensor,
    ) -> torch.Tensor:
        latent, stats = self.base.encode_with_stats(x)
        native_forecast = self.base.decode_with_stats(latent, stats)
        forecast, _gate = self.fusion(
            native_forecast,
            analog_future,
            distance,
            divergence,
            latent,
        )
        return forecast


def _build_model(
    condition: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    lora_blend: float | None = None,
) -> nn.Module:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    base = build_backbone(
        "patch_transformer",
        channels=args.channels,
        context=args.context_length,
        horizon=args.forecast_horizon,
        d_model=args.d_model,
    )
    if condition == "native":
        model: nn.Module = base
    elif condition == "srf":
        model = _RetrievalDeploymentModel(
            base,
            fusion_hidden=args.fusion_hidden,
        )
    else:
        wrapper = ConditionModel(
            base,
            condition=condition,
            rank=args.lora_rank,
            fusion_hidden=args.fusion_hidden,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        # LoRA-Only does not deploy retrieval.  Joint keeps both true weight-
        # level LoRA and SRF, so no module is removed from that graph.
        if condition == "lora":
            wrapper.fusion = nn.Identity()
        if lora_blend is not None:
            wrapper.set_lora_blend(float(lora_blend))
        model = wrapper
    return model.to(device)


def _count_parameter_bytes(model: nn.Module) -> tuple[int, int]:
    count = sum(parameter.numel() for parameter in model.parameters())
    byte_count = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    byte_count += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return int(count), int(byte_count)


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _forward(
    model: nn.Module,
    condition: str,
    x: torch.Tensor,
    retrieval: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    with _autocast(device, amp):
        if condition in RETRIEVAL_CONDITIONS:
            if retrieval is None:
                raise RuntimeError("retrieval inputs missing")
            output = model(x, retrieval[0], retrieval[1], retrieval[2])
        else:
            output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    return output


def _retrieval_numpy(
    bank: ChannelMemoryBank, contexts: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return bank.query(contexts, top_k=top_k, return_candidates=True)


def _benchmark_retrieval(
    bank: ChannelMemoryBank,
    contexts: np.ndarray,
    top_k: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for index in range(warmup):
        start = (index * len(contexts)) % len(contexts)
        _retrieval_numpy(bank, contexts[start : start + 1], top_k)
    samples: list[float] = []
    for index in range(repeats):
        start = (index * 17) % len(contexts)
        begin = time.perf_counter()
        _retrieval_numpy(bank, contexts[start : start + 1], top_k)
        samples.append(1000.0 * (time.perf_counter() - begin))
    return float(statistics.median(samples)), float(np.percentile(samples, 95.0))


def _benchmark_model(
    model: nn.Module,
    condition: str,
    context: np.ndarray,
    retrieval_numpy: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    device: torch.device,
    amp: bool,
    warmup: int,
    repeats: int,
) -> float:
    x = _to_tensor(context, device)
    retrieval = None
    if retrieval_numpy is not None:
        retrieval = tuple(_to_tensor(value, device) for value in retrieval_numpy)  # type: ignore[assignment]
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, condition, x, retrieval, device, amp)
        _synchronize(device)
        samples: list[float] = []
        if device.type == "cuda":
            pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward(model, condition, x, retrieval, device, amp)
                stop.record()
                pairs.append((start, stop))
            _synchronize(device)
            samples = [float(start.elapsed_time(stop)) for start, stop in pairs]
        else:
            for _ in range(repeats):
                begin = time.perf_counter()
                _forward(model, condition, x, retrieval, device, amp)
                samples.append(1000.0 * (time.perf_counter() - begin))
    return float(statistics.median(samples))


def _one_e2e(
    model: nn.Module,
    condition: str,
    bank: ChannelMemoryBank | None,
    contexts: np.ndarray,
    top_k: int,
    device: torch.device,
    amp: bool,
) -> None:
    retrieval_numpy = None
    if condition in RETRIEVAL_CONDITIONS:
        if bank is None:
            raise RuntimeError("retrieval condition has no memory bank")
        retrieval_numpy = _retrieval_numpy(bank, contexts, top_k)
    x = _to_tensor(contexts, device)
    retrieval = None
    if retrieval_numpy is not None:
        retrieval = tuple(_to_tensor(value, device) for value in retrieval_numpy)  # type: ignore[assignment]
    with torch.inference_mode():
        _forward(model, condition, x, retrieval, device, amp)
    _synchronize(device)


def _benchmark_e2e(
    model: nn.Module,
    condition: str,
    bank: ChannelMemoryBank | None,
    contexts: np.ndarray,
    top_k: int,
    device: torch.device,
    amp: bool,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    model.eval()
    for index in range(warmup):
        start = (index * len(contexts)) % len(contexts)
        _one_e2e(model, condition, bank, contexts[start : start + 1], top_k, device, amp)
    samples: list[float] = []
    for index in range(repeats):
        start = (index * 17) % len(contexts)
        begin = time.perf_counter()
        _one_e2e(model, condition, bank, contexts[start : start + 1], top_k, device, amp)
        samples.append(1000.0 * (time.perf_counter() - begin))
    return float(statistics.median(samples)), float(np.percentile(samples, 95.0))


def _benchmark_batch(
    model: nn.Module,
    condition: str,
    bank: ChannelMemoryBank | None,
    contexts: np.ndarray,
    top_k: int,
    device: torch.device,
    amp: bool,
    warmup: int,
    repeats: int,
) -> dict[str, float | None]:
    model.eval()
    batch = contexts
    retrieval_samples: list[float] = []
    model_samples: list[float] = []
    e2e_samples: list[float] = []
    for iteration in range(warmup + repeats):
        begin_total = time.perf_counter()
        retrieval_numpy = None
        retrieval_seconds = 0.0
        if condition in RETRIEVAL_CONDITIONS:
            if bank is None:
                raise RuntimeError("retrieval condition has no memory bank")
            begin_retrieval = time.perf_counter()
            retrieval_numpy = _retrieval_numpy(bank, batch, top_k)
            retrieval_seconds = time.perf_counter() - begin_retrieval
        x = _to_tensor(batch, device)
        retrieval = None
        if retrieval_numpy is not None:
            retrieval = tuple(_to_tensor(value, device) for value in retrieval_numpy)  # type: ignore[assignment]
        _synchronize(device)
        begin_model = time.perf_counter()
        with torch.inference_mode():
            _forward(model, condition, x, retrieval, device, amp)
        _synchronize(device)
        model_seconds = time.perf_counter() - begin_model
        total_seconds = time.perf_counter() - begin_total
        if iteration >= warmup:
            retrieval_samples.append(retrieval_seconds)
            model_samples.append(model_seconds)
            e2e_samples.append(total_seconds)
    windows = len(batch)
    retrieval_median = _median(retrieval_samples)
    model_median = _median(model_samples)
    e2e_median = _median(e2e_samples)
    return {
        "batch_model_windows_per_second": windows / model_median if model_median else None,
        "batch_retrieval_windows_per_second": (
            windows / retrieval_median if retrieval_median and condition in RETRIEVAL_CONDITIONS else None
        ),
        "batch_e2e_windows_per_second": windows / e2e_median if e2e_median else None,
    }


def _measure_inference_peak(
    model: nn.Module,
    condition: str,
    bank: ChannelMemoryBank | None,
    contexts: np.ndarray,
    top_k: int,
    device: torch.device,
    amp: bool,
) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _one_e2e(model, condition, bank, contexts, top_k, device, amp)
    allocated = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0**2)
    return float(allocated), float(reserved)


def _measure_training_peak(
    condition: str,
    contexts: np.ndarray,
    futures: np.ndarray,
    retrieval_numpy: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None

    # Joint is trained sequentially: SRF/head warm-up, followed by LoRA-only
    # adaptation in FP32.  Measuring all three modules as simultaneously
    # trainable would overstate optimizer/gradient memory and would not reflect
    # any actual training step.  Report the larger observed peak of its two
    # real stages instead.
    phases = ("srf", "lora") if condition == "joint" else (condition,)
    peaks: list[float] = []
    for phase in phases:
        gc.collect()
        torch.cuda.empty_cache()
        model = _build_model(condition, args, device, lora_blend=1.0)
        if condition == "native":
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        elif condition == "joint":
            if not isinstance(model, ConditionModel):
                raise TypeError("Joint training peak requires ConditionModel")
            model.configure_trainable("joint", phase=phase)
            model.set_lora_blend(0.0 if phase == "srf" else 1.0)
        elif condition == "srf":
            if not isinstance(model, _RetrievalDeploymentModel):
                raise TypeError("SRF training peak requires retrieval deployment model")
            model.configure_trainable()
        elif condition == "lora":
            if not isinstance(model, ConditionModel):
                raise TypeError("LoRA training peak requires ConditionModel")
            model.configure_trainable("lora", phase="lora")
            model.set_lora_blend(1.0)

        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError(f"{condition}/{phase}: no trainable parameters")
        optimizer = torch.optim.AdamW(trainable, lr=4e-4)
        x = _to_tensor(contexts, device)
        y = _to_tensor(futures, device)
        retrieval = None
        if retrieval_numpy is not None:
            retrieval = tuple(  # type: ignore[assignment]
                _to_tensor(value, device) for value in retrieval_numpy
            )
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # The production Joint LoRA stage is explicitly FP32 to avoid the
        # high-dynamic-range overflow observed on SWaT.
        phase_amp = bool(args.amp and not (condition == "joint" and phase == "lora"))
        prediction = _forward(model, condition, x, retrieval, device, phase_amp)
        loss = (prediction.float() - y).square().mean()
        loss.backward()
        optimizer.step()
        _synchronize(device)
        peaks.append(float(torch.cuda.max_memory_allocated(device) / (1024.0**2)))
        del optimizer, model, x, y, retrieval, prediction, loss
        gc.collect()
        torch.cuda.empty_cache()
    return max(peaks)


def _append_state(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_state(path: Path, benchmark_id: str) -> dict[tuple[str, ...], dict[str, Any]]:
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark_id") != benchmark_id:
                continue
            if row.get("record_type") == "resource":
                condition_key = row.get("condition_key")
                if condition_key is None:
                    reverse_labels = {label: key for key, label in CONDITION_LABELS.items()}
                    condition_key = reverse_labels.get(str(row.get("condition")))
                if condition_key not in CONDITIONS:
                    continue
                key = ("resource", str(condition_key))
            elif row.get("record_type") == "sensitivity":
                key = (
                    "sensitivity",
                    str(row["memory_per_channel"]),
                    str(row["retrieval_top_k"]),
                )
            else:
                continue
            records[key] = row
    return records


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _jsonable(row.get(column)) for column in columns})
    temporary.replace(path)


def _benchmark_id(args: argparse.Namespace, result_paths: Sequence[Path], device: torch.device) -> str:
    script = Path(__file__).resolve()
    script_dir = script.parent
    payload = {
        "schema": 1,
        "script_sha256": _sha256(script),
        "core_sha256": {
            name: _sha256(script_dir / name)
            for name in ("data_pipeline.py", "models.py")
        },
        "result_sha256": {str(path): _sha256(path) for path in result_paths},
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "shape": [args.context_length, args.forecast_horizon, args.channels, args.d_model],
        "adapter": [
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            args.fusion_hidden,
        ],
        "memory": [
            args.memory_candidate_windows,
            args.memory_per_channel,
            args.retrieval_top_k,
            args.memory_selection,
            args.memory_temperature,
        ],
        "grids": [args.k_values, args.memory_values],
        "timing": [
            args.warmup,
            args.single_repeats,
            args.batch_repeats,
            args.sensitivity_repeats,
            args.throughput_batch_size,
        ],
        "amp": args.amp,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _base_row(
    args: argparse.Namespace,
    benchmark_id: str,
    condition: str,
    device: torch.device,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    row: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "measurement_status": "static_only" if args.mode == "static" else args.mode,
        "dataset": "PSM" if args.mode != "synthetic" else "synthetic-PSM-shape",
        "backbone": "PatchTST",
        "condition": CONDITION_LABELS[condition],
        "condition_key": condition,
        "seed": args.seed,
        "device": str(device),
        "gpu": gpu,
        "context_length": args.context_length,
        "forecast_horizon": args.forecast_horizon,
        "channels": args.channels,
        "retrieval_top_k": args.retrieval_top_k if condition in RETRIEVAL_CONDITIONS else 0,
        "memory_per_channel": args.memory_per_channel if condition in RETRIEVAL_CONDITIONS else 0,
        "deployed_lora_blend": observed.get("deployed_lora_blend"),
        "deployed_parameters": observed["deployed_parameters"],
        "adaptation_trainable_parameters": observed["adaptation_trainable_parameters"],
        "deployed_parameter_bytes_fp32": 4 * int(observed["deployed_parameters"]),
        "memory_bank_payload_bytes": 0,
        "memory_bank_object_bytes_approx": 0,
        "batch_size_for_throughput": args.throughput_batch_size,
        "microbenchmark_weights": "deterministic random initialization; no accuracy claim",
        "timing_scope": (
            "batch=1 median wall-clock end-to-end; CUDA-event device-only model; "
            "fixed-batch median throughput; CPU exact retrieval included for SRF/Joint"
        ),
    }
    row.update(observed)
    return row


def _run_resource_condition(
    condition: str,
    args: argparse.Namespace,
    benchmark_id: str,
    observed: Mapping[str, Any],
    bank: ChannelMemoryBank,
    query_contexts: np.ndarray,
    query_futures: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    row = _base_row(args, benchmark_id, condition, device, observed)
    selected_blend = observed.get("deployed_lora_blend")
    model = _build_model(
        condition,
        args,
        device,
        lora_blend=(
            1.0
            if condition in {"lora", "joint"} and selected_blend is None
            else selected_blend
        ),
    )
    parameter_count, parameter_bytes = _count_parameter_bytes(model)
    if parameter_count != int(observed["deployed_parameters"]):
        raise RuntimeError(
            f"{condition}: constructed {parameter_count} deployed parameters, "
            f"result row reports {observed['deployed_parameters']}"
        )
    row["deployed_parameter_bytes_fp32"] = parameter_bytes
    use_retrieval = condition in RETRIEVAL_CONDITIONS
    if use_retrieval:
        row["memory_bank_payload_bytes"] = _array_payload_bytes(bank)
        row["memory_bank_object_bytes_approx"] = _deep_object_bytes(bank)

    single_context = query_contexts[:1]
    single_retrieval = (
        _retrieval_numpy(bank, single_context, args.retrieval_top_k)
        if use_retrieval
        else None
    )
    row["single_model_latency_ms_median"] = _benchmark_model(
        model,
        condition,
        single_context,
        single_retrieval,
        device,
        args.amp,
        args.warmup,
        args.single_repeats,
    )
    if use_retrieval:
        retrieval_median, _ = _benchmark_retrieval(
            bank,
            query_contexts,
            args.retrieval_top_k,
            args.warmup,
            args.single_repeats,
        )
        row["single_retrieval_latency_ms_median"] = retrieval_median
    else:
        row["single_retrieval_latency_ms_median"] = 0.0
    e2e_median, e2e_p95 = _benchmark_e2e(
        model,
        condition,
        bank if use_retrieval else None,
        query_contexts,
        args.retrieval_top_k,
        device,
        args.amp,
        args.warmup,
        args.single_repeats,
    )
    row["single_e2e_latency_ms_median"] = e2e_median
    row["single_e2e_latency_ms_p95"] = e2e_p95

    batch_contexts = query_contexts[: args.throughput_batch_size]
    row.update(
        _benchmark_batch(
            model,
            condition,
            bank if use_retrieval else None,
            batch_contexts,
            args.retrieval_top_k,
            device,
            args.amp,
            args.warmup,
            args.batch_repeats,
        )
    )
    peak1, _ = _measure_inference_peak(
        model,
        condition,
        bank if use_retrieval else None,
        query_contexts[:1],
        args.retrieval_top_k,
        device,
        args.amp,
    )
    peak_batch, reserved_batch = _measure_inference_peak(
        model,
        condition,
        bank if use_retrieval else None,
        batch_contexts,
        args.retrieval_top_k,
        device,
        args.amp,
    )
    row["cuda_inference_peak_allocated_mib_batch1"] = peak1
    row["cuda_inference_peak_allocated_mib_batch"] = peak_batch
    row["cuda_inference_peak_reserved_mib_batch"] = reserved_batch

    training_retrieval = (
        _retrieval_numpy(bank, batch_contexts, args.retrieval_top_k)
        if use_retrieval
        else None
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    row["cuda_training_peak_allocated_mib_batch"] = _measure_training_peak(
        condition,
        batch_contexts,
        query_futures[: args.throughput_batch_size],
        training_retrieval,
        args,
        device,
    )
    row["record_type"] = "resource"
    return row


def _run_sensitivity_point(
    memory_per_channel: int,
    top_k: int,
    bank: ChannelMemoryBank,
    fit_seconds: float,
    args: argparse.Namespace,
    benchmark_id: str,
    query_contexts: np.ndarray,
    device: torch.device,
    memory_candidates: int,
) -> dict[str, Any]:
    condition = "srf"
    model = _build_model(condition, args, device)
    single_context = query_contexts[:1]
    single_retrieval = _retrieval_numpy(bank, single_context, top_k)
    retrieval_ms, _ = _benchmark_retrieval(
        bank,
        query_contexts,
        top_k,
        args.warmup,
        args.sensitivity_repeats,
    )
    model_ms = _benchmark_model(
        model,
        condition,
        single_context,
        single_retrieval,
        device,
        args.amp,
        args.warmup,
        args.sensitivity_repeats,
    )
    e2e_ms, _ = _benchmark_e2e(
        model,
        condition,
        bank,
        query_contexts,
        top_k,
        device,
        args.amp,
        args.warmup,
        args.sensitivity_repeats,
    )
    batch = query_contexts[: args.throughput_batch_size]
    batch_metrics = _benchmark_batch(
        model,
        condition,
        bank,
        batch,
        top_k,
        device,
        args.amp,
        args.warmup,
        max(1, args.batch_repeats // 2),
    )
    row = {
        "record_type": "sensitivity",
        "benchmark_id": benchmark_id,
        "measurement_status": args.mode,
        "dataset": "PSM" if args.mode == "real" else "synthetic-PSM-shape",
        "backbone": "PatchTST",
        "condition": "SRF-Only",
        "seed": args.seed,
        "device": str(device),
        "channels": args.channels,
        "context_length": args.context_length,
        "forecast_horizon": args.forecast_horizon,
        "retrieval_top_k": top_k,
        "memory_per_channel": memory_per_channel,
        "memory_candidates": memory_candidates,
        "memory_fit_seconds": fit_seconds,
        "memory_bank_payload_bytes": _array_payload_bytes(bank),
        "memory_bank_object_bytes_approx": _deep_object_bytes(bank),
        "single_retrieval_latency_ms_median": retrieval_ms,
        "single_model_latency_ms_median": model_ms,
        "single_e2e_latency_ms_median": e2e_ms,
        "batch_size_for_throughput": args.throughput_batch_size,
        "batch_retrieval_windows_per_second": batch_metrics["batch_retrieval_windows_per_second"],
        "batch_e2e_windows_per_second": batch_metrics["batch_e2e_windows_per_second"],
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    benchmark_id: str,
    device: torch.device,
    results: Sequence[Path],
    preparation: Mapping[str, Any] | None,
) -> None:
    metadata = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at_unix": time.time(),
        "mode": args.mode,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "results": [str(path.resolve()) for path in results],
        "preparation": preparation,
        "measurement_contract": {
            "training_time": "observed completed run_main wall time; base plus per-condition adaptation",
            "deployment_graph": (
                "Native=base; LoRA=base plus architecture-specific weight-level LoRA; "
                "SRF=base plus retrieval fusion without dormant LoRA; Joint=base plus LoRA plus retrieval fusion"
            ),
            "lora_inference_blend": (
                "PSM validation-selected blend read from each result row; full-strength 1.0 only when legacy rows omit it"
            ),
            "single_model_latency": "batch=1 device-only forward median; CUDA event on CUDA",
            "single_retrieval_latency": "batch=1 CPU exact ChannelMemoryBank.query median",
            "single_e2e_latency": "batch=1 wall clock including retrieval, host-to-device copies, forward, and synchronization",
            "throughput": "median fixed-batch wall-clock throughput, not reciprocal of batch=1 latency",
            "cuda_peak": "PyTorch allocator max_memory_allocated; excludes CUDA driver/context and non-PyTorch allocations",
            "memory_bank_payload": "exact retained NumPy ndarray nbytes after fitting",
            "memory_bank_object": "sys.getsizeof-based approximation including Python containers and NumPy payload",
            "weights": "deterministic random initialization; resource result only",
        },
        "limitations": [
            "One RTX/process/software stack; latency is descriptive rather than hardware independent.",
            "CPU retrieval and GPU forecasting are sequential and not asynchronously pipelined.",
            "CUDA peak allocation is not nvidia-smi process memory and excludes driver/context memory.",
            "The microbenchmark does not reload trained checkpoints because resource shapes do not depend on learned values.",
        ],
        "arguments": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "synthetic", "static"), default="real")
    parser.add_argument("--data-root", default=os.environ.get("RAG_TSAD_DATA_ROOT", "./data"))
    parser.add_argument("--results", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/efficiency_resources_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-seed", type=int, default=20_260_923)
    parser.add_argument("--max-train-points", type=int, default=0)
    parser.add_argument("--max-test-points", type=int, default=0)
    parser.add_argument("--synthetic-points", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=25)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--fusion-hidden", type=int, default=32)
    parser.add_argument("--memory-candidate-windows", type=int, default=0)
    parser.add_argument("--memory-per-channel", type=int, default=256)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--memory-selection", choices=("uniform", "farthest", "boundary"), default="boundary")
    parser.add_argument("--memory-temperature", type=float, default=0.15)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--memory-values", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--query-windows", type=int, default=512)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--single-repeats", type=int, default=50)
    parser.add_argument("--batch-repeats", type=int, default=10)
    parser.add_argument("--sensitivity-repeats", type=int, default=30)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Ignore same-id completion records and append replacements")
    parser.add_argument("--smoke", action="store_true", help="Cap repetitions and sensitivity grid for a fast synthetic/static check")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.mode == "real":
            parser.error("--smoke requires --mode synthetic or --mode static")
        args.warmup = min(args.warmup, 1)
        args.single_repeats = min(args.single_repeats, 2)
        args.batch_repeats = min(args.batch_repeats, 1)
        args.sensitivity_repeats = min(args.sensitivity_repeats, 2)
        args.query_windows = min(args.query_windows, 32)
        args.throughput_batch_size = min(args.throughput_batch_size, 8)
        args.memory_candidate_windows = min(args.memory_candidate_windows, 512)
        args.k_values = [1, 3]
        args.memory_values = [32, 64]

    positive = (
        "synthetic_points", "channels",
        "context_length", "forecast_horizon", "d_model", "lora_rank",
        "fusion_hidden", "memory_per_channel",
        "retrieval_top_k", "query_windows", "throughput_batch_size",
        "single_repeats", "batch_repeats", "sensitivity_repeats",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.memory_temperature <= 0:
        parser.error("--memory-temperature must be positive")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must lie in [0, 1)")
    if any(value <= 0 for value in args.k_values + args.memory_values):
        parser.error("K and memory grids must contain positive integers")
    args.k_values = sorted(set(args.k_values))
    args.memory_values = sorted(set(args.memory_values))
    if max(args.k_values) > min(args.memory_values):
        parser.error("every K must be no larger than the smallest memory grid value")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result_paths = _discover_results(args.results)
    observed = _observed_summaries(_read_results(result_paths), args.seed)
    device = _resolve_device(args.device)
    benchmark_id = _benchmark_id(args, result_paths, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "resource_measurements.jsonl"
    state = _read_state(state_path, benchmark_id)

    if args.mode == "static":
        for condition in CONDITIONS:
            key = ("resource", condition)
            if key in state and not args.force:
                continue
            row = _base_row(args, benchmark_id, condition, device, observed[condition])
            row["record_type"] = "resource"
            _append_state(state_path, row)
        state = _read_state(state_path, benchmark_id)
        resources = [state[("resource", condition)] for condition in CONDITIONS]
        _write_csv(args.output_dir / "efficiency_resources.csv", RESOURCE_COLUMNS, resources)
        _write_csv(args.output_dir / "sensitivity_latency.csv", SENSITIVITY_COLUMNS, [])
        _write_metadata(args.output_dir / "efficiency_resources_metadata.json", args, benchmark_id, device, result_paths, None)
        print(f"static output complete: benchmark_id={benchmark_id}")
        return 0

    memory_windows, query_contexts, query_futures, preparation = _prepare_arrays(args)
    # A fitted main bank is shared by SRF and Joint; Native/LoRA record zero
    # deployed memory bytes even though this benchmarking helper retains it.
    main_bank, main_fit_seconds = _fit_memory(
        memory_windows, args.memory_per_channel, args.retrieval_top_k, args
    )
    bank_cache: dict[int, tuple[ChannelMemoryBank, float]] = {
        args.memory_per_channel: (main_bank, main_fit_seconds)
    }

    for condition in CONDITIONS:
        key = ("resource", condition)
        if key in state and not args.force:
            print(f"resume: skip completed resource row {condition}")
            continue
        print(f"measure resource row: {condition}", flush=True)
        row = _run_resource_condition(
            condition,
            args,
            benchmark_id,
            observed[condition],
            main_bank,
            query_contexts,
            query_futures,
            device,
        )
        _append_state(state_path, row)

    for memory_per_channel in args.memory_values:
        if memory_per_channel not in bank_cache:
            print(f"fit memory M={memory_per_channel}", flush=True)
            bank_cache[memory_per_channel] = _fit_memory(
                memory_windows, memory_per_channel, min(args.k_values), args
            )
        bank, fit_seconds = bank_cache[memory_per_channel]
        for top_k in args.k_values:
            key = ("sensitivity", str(memory_per_channel), str(top_k))
            if key in state and not args.force:
                print(f"resume: skip completed sensitivity M={memory_per_channel}, K={top_k}")
                continue
            print(f"measure sensitivity: M={memory_per_channel}, K={top_k}", flush=True)
            row = _run_sensitivity_point(
                memory_per_channel,
                top_k,
                bank,
                fit_seconds,
                args,
                benchmark_id,
                query_contexts,
                device,
                int(len(memory_windows[0])),
            )
            _append_state(state_path, row)

    state = _read_state(state_path, benchmark_id)
    resources = [state[("resource", condition)] for condition in CONDITIONS]
    sensitivity = [
        state[("sensitivity", str(memory), str(top_k))]
        for memory in args.memory_values
        for top_k in args.k_values
    ]
    _write_csv(args.output_dir / "efficiency_resources.csv", RESOURCE_COLUMNS, resources)
    _write_csv(args.output_dir / "sensitivity_latency.csv", SENSITIVITY_COLUMNS, sensitivity)
    _write_metadata(
        args.output_dir / "efficiency_resources_metadata.json",
        args,
        benchmark_id,
        device,
        result_paths,
        preparation,
    )
    print(
        f"complete: benchmark_id={benchmark_id}, resources={len(resources)}, "
        f"sensitivity={len(sensitivity)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
