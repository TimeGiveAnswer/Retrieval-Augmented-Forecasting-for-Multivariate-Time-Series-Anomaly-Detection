#!/usr/bin/env python3
"""Generate audited retrieval-success and retrieval-failure mechanism cases.

The script is an independent consumer of the clean-room experiment modules.  It
does not modify model, data, training, or runner source.  It loads a compatible
PatchTST base checkpoint when available, otherwise retrains a base model in its
own output directory, trains or resumes one SRF condition model, freezes all
predictions/scores, and only then uses held-out labels for post-hoc case choice.

Case-selection protocol (fixed before labels are inspected)
-----------------------------------------------------------
1. Enumerate every anomaly event independently inside each held-out segment.
2. Form a fixed-radius event-centred score interval, retaining candidates with
   both classes and finite Native/SRF scores.
3. Define event effect = raw local AUPRC(SRF) - raw local AUPRC(Native).
4. Success is the largest effect; failure is the smallest *distinct* effect.
   "Failure" therefore means the relative worst case and is not claimed to be
   an absolute degradation unless its recorded effect is negative.
5. Inside the selected event, choose the overlapping forecast window/channel
   with respectively largest/smallest SRF-vs-Native squared-error reduction.

The protocol is written to selection_protocol.json before model inference.
Held-out labels never affect training, score weights, retrieval, prediction, or
the candidate selector; the existing labeled-validation split still selects
score-only hyperparameters exactly as in run_main.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "1.0.0"
DEFAULT_DATASETS = ("PSM", "SMD-machine-3-1")
AUTO_SOURCE_DIRS = {}
CORE_FILES = ("run_main.py", "data_pipeline.py", "models.py", "training.py")

SELECTION_PROTOCOL = {
    "version": "event-local-auprc-v1",
    "candidate_unit": "contiguous anomaly event inside one held-out segment",
    "candidate_interval": "event plus fixed --selection-radius points on both sides, clipped to its segment",
    "candidate_requirements": "both labels present and Native/SRF scores finite",
    "effect": "raw local AUPRC(SRF) minus raw local AUPRC(Native)",
    "success": "maximum effect; ties broken by segment then event start",
    "failure": "minimum effect among a distinct event; ties broken by segment then event start",
    "failure_semantics": "relative worst case; absolute failure only when effect < 0",
    "window_rule_success": "overlapping forecast window/channel with maximum Native-MSE minus SRF-MSE",
    "window_rule_failure": "overlapping forecast window/channel with minimum Native-MSE minus SRF-MSE",
    "heldout_label_role": "post-hoc event enumeration and case selection only after models, score parameters, predictions, and scores are frozen",
}


@dataclass
class CoreModules:
    torch: Any
    run_main: Any
    data_pipeline: Any
    models: Any
    training: Any


@dataclass
class SourceRun:
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]
    args: Any
    configuration_id: str
    code_hashes: dict[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in columns})
    os.replace(temporary, path)


def import_core() -> CoreModules:
    try:
        import torch
        import data_pipeline
        import models
        import run_main
        import training
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "The real generator requires the project's PyTorch environment. "
            "Run with the same conda environment used by run_main.py. "
            f"Import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return CoreModules(torch, run_main, data_pipeline, models, training)


def current_core_hashes(project_dir: Path) -> dict[str, str]:
    return {name: sha256_file(project_dir / name) for name in CORE_FILES}


def parse_source_map(entries: Sequence[str], project_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--source-map entries must be DATASET=PATH, got {entry!r}")
        dataset, raw_path = entry.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        mapping[dataset.strip()] = path.resolve()
    return mapping


def locate_source_dir(
    dataset: str,
    project_dir: Path,
    explicit_dir: Path | None,
    source_map: Mapping[str, Path],
    n_datasets: int,
) -> Path:
    if dataset in source_map:
        return source_map[dataset]
    if explicit_dir is not None:
        if n_datasets != 1:
            raise ValueError("--source-run-dir can only be used with one dataset; use --source-map for multiple datasets")
        return explicit_dir.expanduser().resolve()
    if dataset not in AUTO_SOURCE_DIRS:
        raise ValueError(f"No automatic source directory for {dataset}; pass --source-map {dataset}=PATH")
    return Path(AUTO_SOURCE_DIRS[dataset]).resolve()


def load_source_run(
    core: CoreModules,
    dataset: str,
    directory: Path,
    device_override: str | None,
    current_hashes: Mapping[str, str],
    allow_code_mismatch: bool,
) -> SourceRun:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = manifest.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"Source manifest has no reusable run_main command: {manifest_path}")
    script_index = next(
        (i for i, token in enumerate(command) if Path(str(token)).name == "run_main.py"),
        None,
    )
    if script_index is None:
        raise ValueError("Source manifest command does not invoke run_main.py")
    run_args = core.run_main._parse_args([str(token) for token in command[script_index + 1 :]])
    if dataset not in run_args.datasets:
        raise ValueError(f"Dataset {dataset} is not present in source run {directory}")
    if "patch_transformer" not in run_args.backbones:
        raise ValueError("Source run does not contain the PatchTST-compatible patch_transformer backbone")
    if device_override:
        run_args.device = device_override
    configuration_id = str(manifest.get("configuration_id") or "")
    if not configuration_id:
        configuration_id = core.run_main._configuration_id(core.run_main._configuration(run_args))
    recorded = manifest.get("code_sha256") or manifest.get("configuration", {}).get("implementation_sha256") or {}
    mismatches = {
        name: {"recorded": recorded.get(name), "current": current_hashes.get(name)}
        for name in CORE_FILES
        if recorded.get(name) and recorded.get(name) != current_hashes.get(name)
    }
    if mismatches and not allow_code_mismatch:
        raise RuntimeError(
            "Core source differs from the source run. Refusing an unaudited checkpoint load; "
            "use --allow-code-mismatch only if you have independently checked compatibility. "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return SourceRun(
        directory=directory,
        manifest_path=manifest_path,
        manifest=manifest,
        args=run_args,
        configuration_id=configuration_id,
        code_hashes=dict(recorded),
    )


def prepare_with_captured_memory(
    core: CoreModules, dataset: str, args: Any, seed: int
) -> tuple[Any, Any]:
    if args.retrieval_mode != "channel":
        raise ValueError("Mechanism export currently requires channel-wise retrieval")
    original_class = core.run_main.ChannelMemoryBank
    capture: dict[str, Any] = {}

    class CapturingMemoryBank(original_class):
        def fit(self, contexts, futures, normal_mask=None, end_indices=None):  # type: ignore[override]
            result = super().fit(contexts, futures, normal_mask=normal_mask, end_indices=end_indices)
            context_array = np.asarray(contexts, dtype=np.float32)
            future_array = np.asarray(futures, dtype=np.float32)
            source = np.asarray(end_indices, dtype=np.int64).reshape(-1)
            lookup = {int(value): index for index, value in enumerate(source)}
            self.display_contexts_ = []
            self.display_futures_ = []
            for channel, selected_sources in enumerate(self.source_end_indices_):
                positions = np.asarray([lookup[int(value)] for value in selected_sources], dtype=np.int64)
                self.display_contexts_.append(context_array[positions, :, channel].copy())
                self.display_futures_.append(future_array[positions, :, channel].copy())
            capture["memory"] = self
            return result

    core.run_main.ChannelMemoryBank = CapturingMemoryBank
    try:
        prepared = core.run_main._prepare_dataset(dataset, args, seed, True)
    finally:
        core.run_main.ChannelMemoryBank = original_class
    if "memory" not in capture:
        raise RuntimeError("Retrieval memory was not captured during dataset preparation")
    return prepared, capture["memory"]


def build_or_load_base(
    core: CoreModules,
    prepared: Any,
    source: SourceRun,
    seed: int,
    output_dir: Path,
    explicit_checkpoint: Path | None,
    retrain_if_needed: bool,
) -> tuple[Any, dict[str, Any], str, Path | None]:
    args = source.args
    core.training.seed_all(core.run_main._stable_seed(seed, prepared.name, "patch_transformer", "base-init"))
    base = core.models.build_backbone(
        "patch_transformer",
        prepared.channels,
        args.context_length,
        args.forecast_horizon,
        d_model=args.d_model,
    )
    expected_name = (
        f"base-{core.run_main._slug(prepared.name)}-patch-transformer-"
        f"seed{seed}-{source.configuration_id}.pt"
    )
    candidates: list[Path] = []
    if explicit_checkpoint is not None:
        candidates.append(explicit_checkpoint.expanduser().resolve())
    candidates.append(source.directory / "checkpoints" / expected_name)
    artifact_checkpoint_dir = source.manifest.get("artifacts", {}).get("checkpoint_dir")
    if artifact_checkpoint_dir:
        artifact_path = Path(str(artifact_checkpoint_dir)).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = source.directory.parent.parent / artifact_path
        candidates.append(artifact_path / expected_name)
    seen: set[str] = set()
    load_errors: list[str] = []
    for checkpoint in candidates:
        key = str(checkpoint)
        if key in seen or not checkpoint.is_file():
            continue
        seen.add(key)
        try:
            try:
                payload = core.torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = core.torch.load(checkpoint, map_location="cpu")
            payload_configuration = payload.get("configuration_id") if isinstance(payload, Mapping) else None
            if payload_configuration and str(payload_configuration) != source.configuration_id:
                raise ValueError(
                    f"checkpoint configuration_id={payload_configuration} does not match source {source.configuration_id}"
                )
            training = core.run_main._load_checkpoint(checkpoint, base)
            if training is not None:
                base.to("cpu")
                return base, training, "compatible_checkpoint", checkpoint
        except Exception as exc:
            load_errors.append(f"{checkpoint}: {type(exc).__name__}: {exc}")
    if not retrain_if_needed:
        suffix = " Load errors: " + " | ".join(load_errors) if load_errors else ""
        raise FileNotFoundError(f"No compatible base checkpoint found for {prepared.name}.{suffix}")
    checkpoint_dir = output_dir / "checkpoints" / "base"
    base, training, origin = core.run_main._fit_or_load_base(
        prepared,
        "patch_transformer",
        args,
        seed,
        source.configuration_id,
        checkpoint_dir,
    )
    generated = checkpoint_dir / expected_name
    return base, dict(training), f"{origin}_in_mechanism_output", generated if generated.exists() else None


def condition_fingerprint(
    dataset: str,
    seed: int,
    source: SourceRun,
    current_hashes: Mapping[str, str],
) -> dict[str, Any]:
    args = source.args
    return {
        "dataset": dataset,
        "seed": seed,
        "backbone": "patch_transformer",
        "condition": "srf",
        "source_configuration_id": source.configuration_id,
        "core_hashes": dict(current_hashes),
        "context_length": args.context_length,
        "forecast_horizon": args.forecast_horizon,
        "d_model": args.d_model,
        "fusion_hidden": args.fusion_hidden,
        "condition_epochs": args.condition_epochs,
        "condition_lr": args.condition_lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss": args.loss,
        "memory_per_channel": args.memory_per_channel,
        "retrieval_top_k": args.retrieval_top_k,
        "memory_selection": args.memory_selection,
        "memory_source": args.memory_source,
        "memory_temperature": args.memory_temperature,
    }


def train_or_load_srf(
    core: CoreModules,
    base: Any,
    prepared: Any,
    source: SourceRun,
    seed: int,
    output_dir: Path,
    current_hashes: Mapping[str, str],
    force_retrain: bool,
) -> tuple[Any, dict[str, Any], str, Path]:
    args = source.args
    core.training.seed_all(core.run_main._stable_seed(seed, prepared.name, "patch_transformer", "condition-init"))
    model = core.models.ConditionModel(
        copy.deepcopy(base),
        condition="srf",
        rank=args.lora_rank,
        fusion_hidden=args.fusion_hidden,
    )
    fingerprint = condition_fingerprint(prepared.name, seed, source, current_hashes)
    fingerprint_id = stable_id(fingerprint)
    checkpoint = output_dir / "checkpoints" / f"srf-{slug(prepared.name)}-patchtst-seed{seed}-{fingerprint_id}.pt"
    if checkpoint.is_file() and not force_retrain:
        try:
            try:
                payload = core.torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = core.torch.load(checkpoint, map_location="cpu")
            if payload.get("fingerprint_id") != fingerprint_id:
                raise ValueError("condition checkpoint fingerprint mismatch")
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to("cpu")
            return model, dict(payload.get("training") or {}), "condition_checkpoint", checkpoint
        except Exception as exc:
            print(f"WARNING: SRF checkpoint could not be reused; retraining: {exc}", file=sys.stderr)
    if prepared.adapt_retrieval is None or prepared.validation_retrieval is None:
        raise RuntimeError("Prepared dataset has no SRF adaptation arrays")
    training = core.training.train_model(
        model,
        prepared.adapt_retrieval,
        prepared.validation_retrieval,
        epochs=args.condition_epochs,
        batch_size=args.batch_size,
        lr=args.condition_lr,
        device=args.device,
        amp=core.run_main._amp_for("patch_transformer", args),
        patience=args.patience,
        loss=args.loss,
        seed=core.run_main._stable_seed(seed, prepared.name, "patch_transformer", "condition-train"),
    )
    model.to("cpu")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    core.torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": json_safe(training),
            "fingerprint": fingerprint,
            "fingerprint_id": fingerprint_id,
            "created_at": utc_now(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return model, dict(training), "trained_in_mechanism_output", checkpoint


def predict_and_score_condition(
    core: CoreModules,
    model: Any,
    prepared: Any,
    args: Any,
    use_retrieval: bool,
) -> dict[str, Any]:
    condition = "srf" if use_retrieval else "native"
    adapt, validation_arrays, score_validation_arrays, test_arrays = core.run_main._condition_arrays(prepared, condition)
    del adapt
    predict_kwargs = {
        "batch_size": args.batch_size,
        "device": args.device,
        "amp": core.run_main._amp_for("patch_transformer", args),
        "return_aux": True,
    }
    validation = core.training.predict_windows(model, validation_arrays, **predict_kwargs)
    score_validation = core.training.predict_windows(model, score_validation_arrays, **predict_kwargs)
    test = core.training.predict_windows(model, test_arrays, **predict_kwargs)
    validation_errors = validation["predictions"] - validation["targets"]
    validation_memory = score_validation_memory = test_memory = None
    score_validation_agreement = test_agreement = None
    if use_retrieval:
        normal_distance = core.run_main._aggregate_candidate_distance(validation["distance"])
        score_validation_memory = core.run_main._empirical_transform(
            normal_distance,
            core.run_main._aggregate_candidate_distance(score_validation["distance"]),
            tail_score=True,
        )
        test_memory = core.run_main._empirical_transform(
            normal_distance,
            core.run_main._aggregate_candidate_distance(test["distance"]),
            tail_score=True,
        )
        normal_divergence = np.asarray(validation["divergence"], dtype=np.float64)
        score_validation_agreement = core.run_main._empirical_transform(
            normal_divergence, score_validation["divergence"], tail_score=False
        )
        test_agreement = core.run_main._empirical_transform(
            normal_divergence, test["divergence"], tail_score=False
        )
    score_validation_components = core.run_main._score_channel_components(
        score_validation,
        validation_errors,
        len(prepared.score_validation_labels),
        score_validation_memory,
        score_validation_agreement,
    )
    parameters, validation_metrics = core.run_main._select_score_parameters(
        prepared.score_validation_labels,
        score_validation_components,
        args,
        use_retrieval,
        prepared.score_validation_segment_lengths,
    )
    test_components = core.run_main._score_channel_components(
        test,
        validation_errors,
        len(prepared.test_labels),
        test_memory,
        test_agreement,
    )
    scores = core.run_main._apply_score_parameters(
        test_components, parameters, prepared.test_segment_lengths
    )
    return {
        "validation_prediction": validation,
        "score_validation_prediction": score_validation,
        "test_prediction": test,
        "score_parameters": parameters,
        "score_validation_metrics": validation_metrics,
        "point_scores": np.asarray(scores, dtype=np.float64),
    }


def raw_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[keep], s[keep]
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    cumulative_true = np.cumsum(y)
    cumulative_false = np.cumsum(1 - y)
    group_ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    true_positive = cumulative_true[group_ends]
    false_positive = cumulative_false[group_ends]
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def segment_bounds(lengths: Sequence[int]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    offset = 0
    for index, length in enumerate(lengths):
        stop = offset + int(length)
        result.append((index, offset, stop))
        offset = stop
    return result


def enumerate_event_candidates(
    labels: np.ndarray,
    native_scores: np.ndarray,
    srf_scores: np.ndarray,
    segment_lengths: Sequence[int],
    radius: int,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    native_scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
    srf_scores = np.asarray(srf_scores, dtype=np.float64).reshape(-1)
    if not (len(labels) == len(native_scores) == len(srf_scores) == sum(map(int, segment_lengths))):
        raise ValueError("Label/score/segment lengths do not match")
    candidates: list[dict[str, Any]] = []
    event_id = 0
    for segment_index, start, stop in segment_bounds(segment_lengths):
        segment_labels = labels[start:stop]
        transitions = np.diff(np.r_[0, segment_labels > 0, 0].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        for local_start, local_stop in zip(starts, stops):
            event_start, event_stop = start + int(local_start), start + int(local_stop)
            interval_start = max(start, event_start - radius)
            interval_stop = min(stop, event_stop + radius)
            local_labels = labels[interval_start:interval_stop]
            finite = np.isfinite(native_scores[interval_start:interval_stop]) & np.isfinite(
                srf_scores[interval_start:interval_stop]
            )
            if finite.sum() < 2 or len(np.unique(local_labels[finite])) < 2:
                continue
            native_ap = raw_average_precision(
                local_labels[finite], native_scores[interval_start:interval_stop][finite]
            )
            srf_ap = raw_average_precision(
                local_labels[finite], srf_scores[interval_start:interval_stop][finite]
            )
            if not np.isfinite(native_ap) or not np.isfinite(srf_ap):
                continue
            candidates.append(
                {
                    "candidate_id": event_id,
                    "segment_index": segment_index,
                    "segment_start": start,
                    "segment_stop": stop,
                    "event_start": event_start,
                    "event_stop": event_stop,
                    "interval_start": interval_start,
                    "interval_stop": interval_stop,
                    "interval_points": interval_stop - interval_start,
                    "positive_points": int(local_labels.sum()),
                    "native_local_auprc": native_ap,
                    "srf_local_auprc": srf_ap,
                    "effect_auprc": srf_ap - native_ap,
                }
            )
            event_id += 1
    return candidates


def choose_success_failure(candidates: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(candidates) < 2:
        raise RuntimeError(
            "At least two eligible held-out anomaly events are required for distinct success/failure cases; "
            f"found {len(candidates)}. Increase --selection-radius or choose another dataset."
        )
    ordered_high = sorted(
        candidates,
        key=lambda row: (-float(row["effect_auprc"]), int(row["segment_index"]), int(row["event_start"])),
    )
    success = dict(ordered_high[0])
    remaining = [row for row in candidates if int(row["candidate_id"]) != int(success["candidate_id"])]
    failure = dict(
        sorted(
            remaining,
            key=lambda row: (float(row["effect_auprc"]), int(row["segment_index"]), int(row["event_start"])),
        )[0]
    )
    return {"success": success, "failure": failure}


def attach_window_and_channel(
    selected: dict[str, dict[str, Any]],
    native_prediction: Mapping[str, Any],
    srf_prediction: Mapping[str, Any],
    horizon: int,
) -> None:
    native = np.asarray(native_prediction["predictions"], dtype=np.float64)
    srf = np.asarray(srf_prediction["predictions"], dtype=np.float64)
    targets = np.asarray(native_prediction["targets"], dtype=np.float64)
    if native.shape != srf.shape or native.shape != targets.shape:
        raise ValueError("Native/SRF prediction shapes differ")
    if not np.allclose(targets, np.asarray(srf_prediction["targets"]), atol=0, rtol=0):
        raise ValueError("Native/SRF test targets differ")
    ends = np.asarray(native_prediction["end_indices"], dtype=np.int64)
    improvement = np.mean((native - targets) ** 2 - (srf - targets) ** 2, axis=1)
    for outcome, row in selected.items():
        overlap = np.flatnonzero((ends < int(row["event_stop"])) & (ends + horizon > int(row["event_start"])))
        if not len(overlap):
            overlap = np.asarray([int(np.argmin(np.abs(ends - int(row["event_start"]))))])
        values = improvement[overlap]
        flat_index = int(np.argmax(values) if outcome == "success" else np.argmin(values))
        window_position, channel = np.unravel_index(flat_index, values.shape)
        window_index = int(overlap[window_position])
        row["window_index"] = window_index
        row["channel"] = int(channel)
        row["forecast_start"] = int(ends[window_index])
        row["forecast_stop"] = int(ends[window_index] + horizon)
        row["window_channel_mse_improvement"] = float(improvement[window_index, channel])


def compute_selector_weights(
    core: CoreModules,
    model: Any,
    contexts: np.ndarray,
    candidate_futures: np.ndarray,
    distances: np.ndarray,
    divergence: np.ndarray,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay one *inference batch* and expose selector intermediates.

    The input may be one sample or a complete batch.  Mechanism export passes
    the exact sequential batch boundaries used by ``predict_windows``.  This
    matters under CUDA AMP: Transformer/GEMM kernels can produce measurably
    different rounding for batch=1 versus the original batch (normally 128),
    even though model weights and the selected sample are identical.
    """
    torch = core.torch
    device = next(model.parameters()).device
    context_array = np.asarray(contexts, dtype=np.float32)
    candidate_array = np.asarray(candidate_futures, dtype=np.float32)
    distance_array = np.asarray(distances, dtype=np.float32)
    divergence_array = np.asarray(divergence, dtype=np.float32)
    if context_array.ndim == 2:
        context_array = context_array[None]
    if candidate_array.ndim == 3:
        candidate_array = candidate_array[None]
    if distance_array.ndim == 2:
        distance_array = distance_array[None]
    if divergence_array.ndim == 2:
        divergence_array = divergence_array[None]
    batch = len(context_array)
    if not (
        candidate_array.shape[0] == distance_array.shape[0] == divergence_array.shape[0] == batch
    ):
        raise ValueError("Selector replay arrays must have the same batch dimension")
    tensors = {
        "x": torch.as_tensor(context_array, dtype=torch.float32, device=device),
        "analog": torch.as_tensor(candidate_array, dtype=torch.float32, device=device),
        "distance": torch.as_tensor(distance_array, dtype=torch.float32, device=device),
        "divergence": torch.as_tensor(divergence_array, dtype=torch.float32, device=device),
    }
    captured: list[Any] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.fusion.selector.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            with core.training._autocast_context(device, bool(use_amp and device.type == "cuda")):
                internal_latent, internal_stats = model.base.encode_with_stats(tensors["x"])
                internal_native = model.base.decode_with_stats(internal_latent, internal_stats)
                forecast, gate = model(
                    tensors["x"],
                    analog_future=tensors["analog"],
                    distance=tensors["distance"],
                    divergence=tensors["divergence"],
                    return_gate=True,
                )
    finally:
        handle.remove()
        model.train(was_training)
    if len(captured) != 1:
        raise RuntimeError(f"Expected one selector activation, got {len(captured)}")
    raw = captured[0].squeeze(-1)
    log_distance = model.fusion._safe_log(tensors["distance"].permute(0, 2, 1))
    logits = raw - log_distance
    valid = torch.isfinite(tensors["distance"].permute(0, 2, 1))
    logits = logits.masked_fill(~valid, -1.0e4)
    weights = torch.softmax(logits, dim=-1)
    return (
        weights.detach().cpu().numpy(),
        forecast.detach().cpu().numpy(),
        gate.detach().cpu().numpy(),
        internal_native.detach().cpu().numpy(),
    )


def assert_replay_close(
    name: str,
    expected: np.ndarray,
    replayed: np.ndarray,
    *,
    rtol: float = 2e-4,
    atol: float = 5e-5,
) -> dict[str, float]:
    """Strictly validate replay and return auditable numerical diagnostics."""
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(replayed, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(f"{name} replay shape mismatch: {left.shape} versus {right.shape}")
    difference = np.abs(left - right)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), atol)
    diagnostics = {
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_relative_error": float((difference / scale).max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
        raise RuntimeError(
            f"{name} exact-batch replay does not match batched inference: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    return diagnostics


def retrieved_context_details(
    core: CoreModules,
    memory: Any,
    context: np.ndarray,
    channel: int,
    top_k: int,
) -> dict[str, np.ndarray]:
    raw_features = core.data_pipeline._context_features(
        np.asarray(context, dtype=np.float64)[None], memory.waveform_points, memory.eps
    )
    center = memory.feature_center_[channel].astype(np.float64, copy=False)
    scale = memory.feature_scale_[channel].astype(np.float64, copy=False)
    feature = (raw_features[0, channel] - center) / scale
    k = min(int(top_k), len(memory.features_[channel]))
    distances, indices = memory._topk(feature[None], channel, k, None, 0)
    index = indices[0]
    if np.any(index < 0):
        raise RuntimeError("Memory lookup returned an invalid candidate index")
    return {
        "retrieved": np.asarray(memory.display_contexts_[channel][index], dtype=np.float32),
        "retrieved_raw_future": np.asarray(memory.display_futures_[channel][index], dtype=np.float32),
        "retrieved_source_end_indices": np.asarray(memory.source_end_indices_[channel][index], dtype=np.int64),
        "recomputed_candidate_distances": np.asarray(distances[0], dtype=np.float32),
    }


def distance_softmax_weights(distance: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values < np.finfo(np.float32).max)
    weights = np.zeros_like(values)
    if valid.any():
        shifted = values[valid] - values[valid].min()
        logits = -shifted / float(temperature)
        logits -= logits.max()
        current = np.exp(logits)
        weights[valid] = current / current.sum()
    return weights.astype(np.float32)


def export_case(
    core: CoreModules,
    dataset: str,
    outcome: str,
    row: Mapping[str, Any],
    prepared: Any,
    memory: Any,
    native_result: Mapping[str, Any],
    srf_result: Mapping[str, Any],
    srf_model: Any,
    source: SourceRun,
    seed: int,
    output_dir: Path,
    current_hashes: Mapping[str, str],
    base_origin: str,
    base_checkpoint: Path | None,
    srf_origin: str,
    srf_checkpoint: Path,
) -> dict[str, Any]:
    args = source.args
    index = int(row["window_index"])
    channel = int(row["channel"])
    test_data = prepared.test_retrieval
    if test_data is None or test_data.analog is None or test_data.distance is None or test_data.divergence is None:
        raise RuntimeError("SRF test retrieval arrays are unavailable")
    context_all = np.asarray(test_data.contexts[index], dtype=np.float32)
    future_all = np.asarray(test_data.futures[index], dtype=np.float32)
    candidate_all = np.asarray(test_data.analog[index], dtype=np.float32)
    distance_all = np.asarray(test_data.distance[index], dtype=np.float32)
    divergence_all = np.asarray(test_data.divergence[index], dtype=np.float32)
    # Replay the exact sequential DataLoader batch that produced this window.
    # Batch=1 is not numerically equivalent under CUDA AMP for Transformer/GEMM
    # kernels and caused false consistency failures in the first exporter.
    batch_start = (index // int(args.batch_size)) * int(args.batch_size)
    batch_stop = min(batch_start + int(args.batch_size), len(test_data))
    local_index = index - batch_start
    learned_batch, forecast_batch, gate_batch, internal_native_batch = compute_selector_weights(
        core,
        srf_model,
        test_data.contexts[batch_start:batch_stop],
        test_data.analog[batch_start:batch_stop],
        test_data.distance[batch_start:batch_stop],
        test_data.divergence[batch_start:batch_stop],
        core.run_main._amp_for("patch_transformer", args),
    )
    learned_weights = learned_batch[local_index]
    recomputed_forecast = forecast_batch[local_index]
    recomputed_gate = gate_batch[local_index]
    internal_native = internal_native_batch[local_index]
    memory_details = retrieved_context_details(
        core, memory, context_all, channel, args.retrieval_top_k
    )
    candidate_distances = distance_all[:, channel]
    if not np.allclose(
        memory_details["recomputed_candidate_distances"],
        candidate_distances[: len(memory_details["recomputed_candidate_distances"])],
        rtol=2e-4,
        atol=2e-5,
    ):
        raise RuntimeError("Recomputed memory candidates do not match the candidates supplied to SRF")
    selector_weights = learned_weights[channel].astype(np.float32)
    candidate_futures = candidate_all[:, :, channel].astype(np.float32)
    retrieval_forecast = np.einsum("k,kh->h", selector_weights, candidate_futures).astype(np.float32)
    native_prediction = np.asarray(native_result["test_prediction"]["predictions"][index], dtype=np.float32)
    srf_prediction = np.asarray(srf_result["test_prediction"]["predictions"][index], dtype=np.float32)
    gate = np.asarray(srf_result["test_prediction"]["gate"][index, :, channel], dtype=np.float32)
    forecast_replay = assert_replay_close(
        "Selected-window SRF forecast", srf_prediction, recomputed_forecast
    )
    gate_replay = assert_replay_close(
        "Selected-window SRF gate", gate, recomputed_gate[:, channel]
    )
    trace_start, trace_stop = int(row["interval_start"]), int(row["interval_stop"])
    native_score = np.asarray(native_result["point_scores"][trace_start:trace_stop], dtype=np.float32)
    srf_score = np.asarray(srf_result["point_scores"][trace_start:trace_stop], dtype=np.float32)
    labels = np.asarray(prepared.test_labels[trace_start:trace_stop], dtype=np.int64)
    case_id = f"{slug(dataset)}-{outcome}-seed{seed}"
    arrays = {
        "query": context_all[:, channel].astype(np.float32),
        "retrieved": memory_details["retrieved"],
        "retrieved_raw_future": memory_details["retrieved_raw_future"],
        "retrieved_source_end_indices": memory_details["retrieved_source_end_indices"],
        "true_future": future_all[:, channel].astype(np.float32),
        "candidate_futures": candidate_futures,
        "candidate_distances": candidate_distances.astype(np.float32),
        "weights": selector_weights,
        "retrieval_distance_weights": distance_softmax_weights(candidate_distances, args.memory_temperature),
        "gate": gate,
        "native_forecast": native_prediction[:, channel],
        "srf_internal_native_forecast": internal_native[:, channel].astype(np.float32),
        "retrieval_forecast": retrieval_forecast,
        "final_forecast": srf_prediction[:, channel],
        "native_score": native_score,
        "srf_score": srf_score,
        "anomaly_score": srf_score,
        "labels": labels,
    }
    npz_path = output_dir / f"{case_id}.npz"
    json_path = npz_path.with_suffix(".json")
    metadata = {
        "case_id": case_id,
        "dataset": dataset,
        "outcome": outcome,
        "failure_is_absolute_degradation": bool(outcome == "failure" and float(row["effect_auprc"]) < 0),
        "seed": seed,
        "backbone": "PatchTST clean-room surrogate",
        "condition": "SRF-Only",
        "channel": channel,
        "source_channel": channel,
        "top_k_to_show": min(5, args.retrieval_top_k),
        "signal_scale": "training-history robust scaled",
        "weights_definition": "trained SRF candidate-selector softmax for source_channel; includes learned selector logit and negative log-distance bias",
        "selector_replay": {
            "batch_start": batch_start,
            "batch_stop": batch_stop,
            "batch_size": batch_stop - batch_start,
            "local_index": local_index,
            "amp_enabled": bool(core.run_main._amp_for("patch_transformer", args)),
            "forecast_validation": forecast_replay,
            "gate_validation": gate_replay,
        },
        "retrieved_definition": "stored normal memory contexts for source_channel in the same candidate order as candidate_futures",
        "candidate_futures_definition": "level-aligned normal futures supplied to SRF",
        "score_trace_coordinate": "concatenated held-out segments; causal smoothing reset at segment boundaries",
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_record": dict(row),
        "native_score_parameters": native_result["score_parameters"],
        "srf_score_parameters": srf_result["score_parameters"],
        "native_score_validation_metrics": native_result["score_validation_metrics"],
        "srf_score_validation_metrics": srf_result["score_validation_metrics"],
        "heldout_labels_used_for": "post-hoc event/case choice and display only",
        "base_origin": base_origin,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
        "srf_origin": srf_origin,
        "srf_checkpoint": str(srf_checkpoint),
        "source_run_dir": str(source.directory),
        "source_manifest_sha256": sha256_file(source.manifest_path),
        "source_configuration_id": source.configuration_id,
        "core_sha256": dict(current_hashes),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "created_at": utc_now(),
    }
    atomic_npz(npz_path, arrays)
    atomic_json(json_path, metadata)
    return {
        "dataset": dataset,
        "outcome": outcome,
        "case_id": case_id,
        "effect_auprc": row["effect_auprc"],
        "native_local_auprc": row["native_local_auprc"],
        "srf_local_auprc": row["srf_local_auprc"],
        "absolute_failure": metadata["failure_is_absolute_degradation"],
        "window_index": index,
        "source_channel": channel,
        "npz": str(npz_path),
        "json": str(json_path),
        "npz_sha256": sha256_file(npz_path),
        "json_sha256": sha256_file(json_path),
    }


def process_dataset(
    core: CoreModules,
    dataset: str,
    source_dir: Path,
    options: argparse.Namespace,
    project_dir: Path,
    current_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"load source dataset={dataset} dir={source_dir}", flush=True)
    source = load_source_run(
        core,
        dataset,
        source_dir,
        options.device,
        current_hashes,
        options.allow_code_mismatch,
    )
    if options.seed not in source.args.seeds:
        print(
            f"WARNING: seed {options.seed} was not listed in source manifest; a new compatible base/SRF may be trained.",
            file=sys.stderr,
        )
    print(f"prepare dataset={dataset}", flush=True)
    prepared, memory = prepare_with_captured_memory(core, dataset, source.args, options.seed)
    explicit_base = options.base_checkpoint.expanduser().resolve() if options.base_checkpoint else None
    base, base_training, base_origin, base_checkpoint = build_or_load_base(
        core,
        prepared,
        source,
        options.seed,
        options.output_dir,
        explicit_base,
        options.retrain_base_if_needed,
    )
    print(f"base dataset={dataset} origin={base_origin}", flush=True)
    srf_model, srf_training, srf_origin, srf_checkpoint = train_or_load_srf(
        core,
        base,
        prepared,
        source,
        options.seed,
        options.output_dir,
        current_hashes,
        options.force_retrain_srf,
    )
    print(f"SRF dataset={dataset} origin={srf_origin}", flush=True)
    native_model = core.models.ConditionModel(
        copy.deepcopy(base),
        condition="native",
        rank=source.args.lora_rank,
        fusion_hidden=source.args.fusion_hidden,
    )
    # Freeze predictions and score parameters before passing held-out labels to
    # enumerate_event_candidates below.
    native_result = predict_and_score_condition(core, native_model, prepared, source.args, False)
    srf_result = predict_and_score_condition(core, srf_model, prepared, source.args, True)
    candidates = enumerate_event_candidates(
        prepared.test_labels,
        native_result["point_scores"],
        srf_result["point_scores"],
        prepared.test_segment_lengths,
        options.selection_radius,
    )
    selected = choose_success_failure(candidates)
    attach_window_and_channel(
        selected,
        native_result["test_prediction"],
        srf_result["test_prediction"],
        source.args.forecast_horizon,
    )
    selected_ids = {outcome: int(row["candidate_id"]) for outcome, row in selected.items()}
    audit_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = dict(candidate)
        current["dataset"] = dataset
        current["selected_as"] = next(
            (outcome for outcome, candidate_id in selected_ids.items() if candidate_id == int(candidate["candidate_id"])),
            "",
        )
        audit_rows.append(current)
    cases = [
        export_case(
            core,
            dataset,
            outcome,
            row,
            prepared,
            memory,
            native_result,
            srf_result,
            srf_model,
            source,
            options.seed,
            options.output_dir,
            current_hashes,
            base_origin,
            base_checkpoint,
            srf_origin,
            srf_checkpoint,
        )
        for outcome, row in selected.items()
    ]
    dataset_record = {
        "dataset": dataset,
        "source_run": str(source.directory),
        "base_origin": base_origin,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
        "base_training": base_training,
        "srf_origin": srf_origin,
        "srf_checkpoint": str(srf_checkpoint),
        "srf_training": srf_training,
        "eligible_events": len(candidates),
        "selected": selected,
        "native_score_parameters": native_result["score_parameters"],
        "srf_score_parameters": srf_result["score_parameters"],
    }
    atomic_json(options.output_dir / f"{slug(dataset)}-generation-record.json", dataset_record)
    return cases, audit_rows


def self_test() -> int:
    rng = np.random.default_rng(20260924)
    n = 1000
    labels = np.zeros(n, dtype=np.int64)
    labels[190:220] = 1
    labels[590:625] = 1
    labels[820:840] = 1
    native = rng.normal(0.15, 0.03, n)
    native[labels == 1] += 0.18
    native[150:260] += 0.18
    native[190:220] -= 0.16
    srf = native.copy()
    srf[190:220] += 0.5
    first_region = np.arange(150, 260)
    srf[first_region[labels[first_region] == 0]] -= 0.16
    srf[590:625] -= 0.22
    second_region = np.arange(550, 670)
    srf[second_region[labels[second_region] == 0]] += 0.24
    candidates = enumerate_event_candidates(labels, native, srf, [n], 80)
    selected = choose_success_failure(candidates)
    assert selected["success"]["effect_auprc"] > selected["failure"]["effect_auprc"]
    with tempfile.TemporaryDirectory(prefix="mechanism-case-selftest-") as temporary_dir:
        out = Path(temporary_dir)
        arrays = {
            "query": rng.normal(size=48).astype(np.float32),
            "retrieved": rng.normal(size=(3, 48)).astype(np.float32),
            "true_future": rng.normal(size=8).astype(np.float32),
            "candidate_futures": rng.normal(size=(3, 8)).astype(np.float32),
            "weights": np.asarray([0.6, 0.3, 0.1], dtype=np.float32),
            "gate": np.full(8, 0.1, dtype=np.float32),
            "native_forecast": rng.normal(size=8).astype(np.float32),
            "final_forecast": rng.normal(size=8).astype(np.float32),
            "native_score": native[110:300].astype(np.float32),
            "srf_score": srf[110:300].astype(np.float32),
            "anomaly_score": srf[110:300].astype(np.float32),
            "labels": labels[110:300],
        }
        npz = out / "synthetic-success.npz"
        atomic_npz(npz, arrays)
        atomic_json(npz.with_suffix(".json"), {"case_id": "synthetic-success", "dataset": "synthetic", "outcome": "success", "channel": 0})
        with np.load(npz, allow_pickle=False) as loaded:
            assert set(["query", "retrieved", "candidate_futures", "native_score", "srf_score"]).issubset(loaded.files)
            assert loaded["weights"].shape == (3,)
            assert np.isclose(loaded["weights"].sum(), 1.0)
        try:
            from analyze_extended_experiments import validate_mechanism_case

            _, _, validation_errors = validate_mechanism_case(npz)
            assert not validation_errors, validation_errors
        except ImportError:
            # The generator itself only requires NumPy for --self-test. The
            # full analyser may be unavailable in a minimal deployment.
            pass
    print(
        json.dumps(
            {
                "status": "ok",
                "eligible_events": len(candidates),
                "success_effect": selected["success"]["effect_auprc"],
                "failure_effect": selected["failure"]["effect_auprc"],
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audited Native-vs-SRF success/failure mechanism cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--source-run-dir", type=Path, default=None, help="Source run for a single dataset.")
    parser.add_argument(
        "--source-map",
        nargs="*",
        default=[],
        metavar="DATASET=PATH",
        help="Per-dataset source run override; useful when generating both defaults.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mechanism_cases_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override source run device, e.g. cuda or cpu.")
    parser.add_argument("--selection-radius", type=int, default=128)
    parser.add_argument("--base-checkpoint", type=Path, default=None, help="Explicit base checkpoint; one dataset only.")
    parser.add_argument(
        "--retrain-base-if-needed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain the base in the mechanism output if no strict-compatible source checkpoint exists.",
    )
    parser.add_argument("--force-retrain-srf", action="store_true")
    parser.add_argument("--allow-code-mismatch", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    if options.self_test:
        return self_test()
    if options.selection_radius <= 0:
        raise SystemExit("--selection-radius must be positive")
    if len(set(options.datasets)) != len(options.datasets):
        raise SystemExit("--datasets contains duplicates")
    if options.base_checkpoint is not None and len(options.datasets) != 1:
        raise SystemExit("--base-checkpoint can only be used with one dataset")
    project_dir = Path(__file__).resolve().parent
    options.output_dir = options.output_dir.expanduser().resolve()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    core = import_core()
    hashes = current_core_hashes(project_dir)
    source_map = parse_source_map(options.source_map, project_dir)
    # Persist the rule before any held-out labels are inspected by this script.
    protocol_record = {
        "written_at": utc_now(),
        "written_before_model_scoring": True,
        "script_version": SCRIPT_VERSION,
        "selection_radius": options.selection_radius,
        "datasets": list(options.datasets),
        "seed": options.seed,
        "protocol": SELECTION_PROTOCOL,
        "core_sha256": hashes,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(options.output_dir / "selection_protocol.json", protocol_record)
    all_cases: list[dict[str, Any]] = []
    all_audit: list[dict[str, Any]] = []
    started = time.perf_counter()
    for dataset in options.datasets:
        source_dir = locate_source_dir(
            dataset,
            project_dir,
            options.source_run_dir,
            source_map,
            len(options.datasets),
        )
        cases, audit = process_dataset(core, dataset, source_dir, options, project_dir, hashes)
        all_cases.extend(cases)
        all_audit.extend(audit)
    write_csv(options.output_dir / "mechanism_case_index.csv", all_cases)
    write_csv(options.output_dir / "case_selection_audit.csv", all_audit)
    manifest = {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "datasets": list(options.datasets),
        "seed": options.seed,
        "selection_protocol": str(options.output_dir / "selection_protocol.json"),
        "case_index": str(options.output_dir / "mechanism_case_index.csv"),
        "selection_audit": str(options.output_dir / "case_selection_audit.csv"),
        "cases": all_cases,
        "core_sha256": hashes,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(options.output_dir / "manifest.json", manifest)
    print(f"generated {len(all_cases)} mechanism cases in {options.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
