"""Independent extension runner for module ablations and sensitivity sweeps.

This file deliberately leaves ``run_main.py`` and the clean-room model/data
implementation untouched.  It orchestrates the public runner, installs small
runtime-only ablation shims in isolated worker processes, reuses only compatible
base-forecast checkpoints, and rebuilds tidy JSONL/CSV outputs after every run.

The controlled reference system is PatchTransformer + SRF with channel-wise
retrieval, boundary-aware memory, a learned Top-K selector, a continuous gate,
and both calibrated retrieval score components.  All supplied SMD machines are run
physically and then averaged within seed so that the statistical unit remains
the SMD dataset family.

Typical remote usage (inside the existing ``v12`` environment)::

    python run_extended_ablations.py plan \
      --output-dir outputs/extended_ablations_v1
    python run_extended_ablations.py run \
      --output-dir outputs/extended_ablations_v1
    python run_extended_ablations.py summarize \
      --output-dir outputs/extended_ablations_v1

The ``run`` command is restart-safe.  Each worker delegates persistence to the
existing runner's JSONL checkpoint protocol; completed seed/dataset/model cells
are skipped on a subsequent invocation.  Do not run two executors against the
same output directory concurrently because the shared base-checkpoint cache is
intentionally single-writer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROJECT_DIR = Path(__file__).resolve().parent
from full_data import discover_datasets, family_of
DEFAULT_DATA_ROOT = os.environ.get(
    "RAG_TSAD_DATA_ROOT", "./data"
)
FAMILY_ORDER = ("PSM", "MSL", "SMAP", "SMD", "SWaT")
# Every family uses the same full-data protocol; prototype budgets are model
# parameters, not truncation/subsampling limits on the input histories.
DATASET_PROTOCOLS = {family: dict(crop_mode="head", max_train_points=0, max_test_points=0,
    base_max_windows=0, condition_max_windows=0, validation_max_windows=0,
    memory_candidate_max_windows=0, patience=5) for family in FAMILY_ORDER}

MODULE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "level": "full",
        "description": "All six retrieval mechanisms enabled.",
        "overrides": {},
        "patches": [],
    },
    {
        "level": "no_boundary_samples",
        "description": "Replace 80/20 boundary-aware memory with farthest-point memory.",
        "overrides": {"memory_selection": "farthest"},
        "patches": [],
    },
    {
        "level": "no_channelwise_retrieval",
        "description": "Use one global multivariate neighbour set instead of per-variable sets.",
        "overrides": {"retrieval_mode": "global"},
        "patches": [],
    },
    {
        "level": "no_candidate_selection",
        "description": "Use uniform weights inside the retrieved Top-K shortlist.",
        "overrides": {},
        "patches": ["uniform_candidate_weights"],
    },
    {
        "level": "no_continuous_gate",
        "description": "Fix gamma to one, applying the complete retrieval correction.",
        "overrides": {},
        "patches": ["fixed_unit_gate"],
    },
    {
        "level": "no_memory_distance",
        "description": (
            "Retain Top-K neighbour identities but remove distance magnitude from "
            "fusion and force its anomaly-score weight to zero."
        ),
        "overrides": {"score_memory_grid": [0.0]},
        "patches": ["zero_distance_feature"],
    },
    {
        "level": "no_reference_divergence",
        "description": (
            "Zero the reference-divergence fusion input and force its score weight to zero."
        ),
        "overrides": {"score_agreement_grid": [0.0]},
        "patches": ["zero_divergence_feature"],
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _append_event(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _family_for(dataset: str) -> str:
    return family_of(dataset)


def _base_core_settings(options: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_root": str(Path(options.data_root).expanduser()),
        "conditions": ["srf"],
        "crop_seed": int(options.crop_seed),
        "device": options.device,
        "amp": bool(options.amp),
        "context_length": options.context_length,
        "forecast_horizon": options.forecast_horizon,
        "test_stride": 1,
        "labeled_validation_fraction": 0.20,
        "score_validation_block_size": options.score_validation_block_size,
        "memory_per_channel": 256,
        "retrieval_mode": "channel",
        "memory_selection": "boundary",
        "memory_source": "base_memory",
        "retrieval_top_k": 5,
        "d_model": options.d_model,
        "lora_rank": 8,
        "fusion_hidden": 32,
        "base_epochs": options.base_epochs,
        "condition_epochs": options.condition_epochs,
        "base_lr": 3e-4,
        "condition_lr": 4e-4,
        "batch_size": options.batch_size,
        "memory_temperature": 0.15,
        "score_selection": "aggregate",
        "loss": "huber",
        "score_memory_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "score_agreement_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        "score_top_q_grid": [0.05, 0.1, 0.2, 0.4, 1.0],
        "score_smoothing_grid": [1, 3, 5, 9, 17, 33],
    }


def _dataset_settings(dataset: str) -> dict[str, Any]:
    return dict(DATASET_PROTOCOLS[_family_for(dataset)])


def _analysis(
    suite: str,
    factor: str,
    level: str | int | float,
    description: str,
) -> dict[str, Any]:
    return {
        "analysis_id": _stable_id([suite, factor, str(level)]),
        "suite": suite,
        "factor": factor,
        "level": str(level),
        "description": description,
    }


def _add_candidate(
    candidates: list[dict[str, Any]],
    *,
    analysis: dict[str, Any],
    settings: Mapping[str, Any],
    patches: Sequence[str],
    seeds: Sequence[int],
    datasets: Sequence[str],
    backbones: Sequence[str],
) -> None:
    for dataset in datasets:
        for backbone in backbones:
            core = dict(settings)
            core.update(_dataset_settings(dataset))
            core.update(
                {
                    "datasets": [dataset],
                    "backbones": [backbone],
                    "seeds": [int(seed) for seed in seeds],
                }
            )
            candidates.append(
                {
                    "core_settings": core,
                    "patches": sorted(set(patches)),
                    "dataset": dataset,
                    "family": _family_for(dataset),
                    "backbone": backbone,
                    "analyses": [analysis],
                }
            )


def _build_plan(options: argparse.Namespace) -> dict[str, Any]:
    families = list(options.datasets)
    physical = discover_datasets(options.data_root, families)
    baseline = _base_core_settings(options)
    candidates: list[dict[str, Any]] = []

    if "module" in options.suites:
        for variant in MODULE_VARIANTS:
            settings = dict(baseline)
            settings.update(variant["overrides"])
            _add_candidate(
                candidates,
                analysis=_analysis(
                    "module",
                    "component",
                    variant["level"],
                    variant["description"],
                ),
                settings=settings,
                patches=variant["patches"],
                seeds=options.module_seeds,
                datasets=physical,
                backbones=options.backbones,
            )

    if "sensitivity" in options.suites:
        factor_definitions: list[tuple[str, Sequence[Any], str]] = []
        if "k" in options.sensitivity_factors:
            factor_definitions.append(
                ("retrieval_top_k", options.k_values, "Top-K retrieved candidates")
            )
        if "memory" in options.sensitivity_factors:
            factor_definitions.append(
                ("memory_per_channel", options.memory_values, "Memory exemplars per channel")
            )
        if "context" in options.sensitivity_factors:
            factor_definitions.append(
                ("context_length", options.context_values, "Input window length L")
            )
        if "score_memory_weight" in options.sensitivity_factors:
            factor_definitions.append(
                (
                    "score_memory_weight",
                    options.score_memory_weight_values,
                    "Fixed calibrated memory-distance score weight",
                )
            )
        for factor, values, description in factor_definitions:
            for value in values:
                settings = dict(baseline)
                if factor == "score_memory_weight":
                    settings["score_memory_grid"] = [float(value)]
                else:
                    settings[factor] = int(value)
                _add_candidate(
                    candidates,
                    analysis=_analysis("sensitivity", factor, value, description),
                    settings=settings,
                    patches=[],
                    seeds=options.sensitivity_seeds,
                    datasets=physical,
                    backbones=options.backbones,
                )

    # De-duplicate identical executions (e.g. K=5, M=256, L=192) while retaining
    # all analysis memberships needed to draw each parameter curve.
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        execution_signature = {
            "core_settings": candidate["core_settings"],
            "patches": candidate["patches"],
        }
        execution_id = _stable_id(execution_signature, length=20)
        if execution_id not in merged:
            item = dict(candidate)
            item["execution_id"] = execution_id
            merged[execution_id] = item
        else:
            known = {
                member["analysis_id"] for member in merged[execution_id]["analyses"]
            }
            merged[execution_id]["analyses"].extend(
                member
                for member in candidate["analyses"]
                if member["analysis_id"] not in known
            )

    jobs = []
    for execution_id in sorted(merged):
        job = merged[execution_id]
        primary = job["analyses"][0]
        job["job_id"] = execution_id
        job["relative_output_dir"] = str(
            Path("jobs")
            / primary["suite"]
            / primary["factor"]
            / primary["level"]
            / job["dataset"]
            / job["backbone"]
            / execution_id
        )
        jobs.append(job)

    code_files = (
        "run_extended_ablations.py",
        "run_main.py",
        "data_pipeline.py",
        "models.py",
        "training.py",
    )
    plan_core = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": {
            name: _sha256(PROJECT_DIR / name) for name in code_files
        },
        "design": {
            "families": families,
            "physical_datasets": physical,
            "backbones": list(options.backbones),
            "module_seeds": list(options.module_seeds),
            "sensitivity_seeds": list(options.sensitivity_seeds),
            "suites": list(options.suites),
            "sensitivity_factors": list(options.sensitivity_factors),
            "smd_aggregation": "equal mean of all supplied entities within each seed",
            "point_adjustment": False,
            "test_label_tuning": False,
            "controlled_baseline": baseline,
        },
        "jobs": jobs,
    }
    plan_core["plan_id"] = _stable_id(plan_core, length=24)
    return plan_core


def _ensure_plan(options: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    output_dir = Path(options.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    proposed = _build_plan(options)
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        if existing.get("plan_id") != proposed.get("plan_id"):
            raise RuntimeError(
                f"{plan_path} contains a different plan. Use a new output directory "
                "instead of mixing protocols."
            )
        return existing, output_dir
    _atomic_json(plan_path, proposed)
    return proposed, output_dir


def _cli_tokens(settings: Mapping[str, Any], output_dir: Path, retry_errors: bool) -> list[str]:
    tokens: list[str] = ["--output-dir", str(output_dir)]
    for key, value in settings.items():
        option = "--" + key.replace("_", "-")
        if key == "amp":
            tokens.append("--amp" if value else "--no-amp")
        elif isinstance(value, bool):
            if value:
                tokens.append(option)
        elif isinstance(value, (list, tuple)):
            tokens.append(option)
            tokens.extend(str(item) for item in value)
        else:
            tokens.extend((option, str(value)))
    if retry_errors:
        tokens.append("--retry-errors")
    return tokens


def _base_cache_id(args: Any, dataset: str, backbone: str) -> str:
    relevant = {
        "dataset": dataset,
        "backbone": backbone,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "crop_seed": args.crop_seed,
        "crop_mode": args.crop_mode,
        "max_train_points": args.max_train_points,
        "context_length": args.context_length,
        "forecast_horizon": args.forecast_horizon,
        "base_max_windows": args.base_max_windows,
        "validation_max_windows": args.validation_max_windows,
        "d_model": args.d_model,
        "base_epochs": args.base_epochs,
        "base_lr": args.base_lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss": args.loss,
        "amp": args.amp,
        "core_sha256": {
            name: _sha256(PROJECT_DIR / name)
            for name in ("run_main.py", "data_pipeline.py", "models.py", "training.py")
        },
    }
    return "extbase-" + _stable_id(relevant, length=20)


def _install_runtime_patches(patches: Sequence[str]) -> None:
    """Install auditable, worker-local ablations without editing core files."""

    import torch
    from torch import nn

    import models

    patch_set = set(patches)
    allowed = {
        "uniform_candidate_weights",
        "fixed_unit_gate",
        "zero_distance_feature",
        "zero_divergence_feature",
    }
    unknown = patch_set - allowed
    if unknown:
        raise ValueError(f"Unknown runtime patches: {sorted(unknown)}")
    if not patch_set:
        return

    original_init = models.SelectiveRetrievalFusion.__init__
    original_forward = models.SelectiveRetrievalFusion.forward

    class UniformCandidateWeights(nn.Module):
        def forward(self, selector_input: torch.Tensor) -> torch.Tensor:
            # The penultimate selector input is log-distance.  Returning it
            # exactly cancels the core forward's subsequent '- log_distance',
            # yielding equal logits for every finite Top-K candidate.
            return selector_input[..., -2:-1]

    class FixedUnitGate(nn.Module):
        def forward(self, control: torch.Tensor) -> torch.Tensor:
            # sigmoid(80) is exactly one at the supported float precisions.
            return torch.full_like(control[..., :1], 80.0)

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if "uniform_candidate_weights" in patch_set:
            self.selector = UniformCandidateWeights()
        if "fixed_unit_gate" in patch_set:
            self.gate_network = FixedUnitGate()

    def patched_forward(
        self: Any,
        native_forecast: Any,
        candidate_futures: Any,
        distance: Any,
        divergence: Any,
        query_latent: Any,
        return_aux: bool = False,
    ) -> Any:
        if "zero_distance_feature" in patch_set:
            # Keep non-finite sentinels invalid while removing every finite
            # distance magnitude from selector/control features.
            distance = torch.where(
                torch.isfinite(distance) & (distance < 0.5 * torch.finfo(torch.float32).max), torch.zeros_like(distance), distance
            )
        if "zero_divergence_feature" in patch_set:
            divergence = torch.zeros_like(divergence)
        return original_forward(
            self,
            native_forecast,
            candidate_futures,
            distance,
            divergence,
            query_latent,
            return_aux=return_aux,
        )

    models.SelectiveRetrievalFusion.__init__ = patched_init
    models.SelectiveRetrievalFusion.forward = patched_forward


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    output_root = Path(spec["output_root"]).expanduser().resolve()
    core_output = output_root / spec["relative_output_dir"]
    core_output.mkdir(parents=True, exist_ok=True)

    import run_main

    _install_runtime_patches(spec.get("patches", []))
    original_configuration = run_main._configuration
    original_fit_or_load_base = run_main._fit_or_load_base
    runner_hash = _sha256(Path(__file__).resolve())

    def extended_configuration(args: Any) -> dict[str, Any]:
        value = original_configuration(args)
        value["independent_extension"] = {
            "schema_version": SCHEMA_VERSION,
            "runner_sha256": runner_hash,
            "execution_id": spec["execution_id"],
            "patches": list(spec.get("patches", [])),
            "analyses": list(spec["analyses"]),
        }
        return value

    def shared_base_fit(
        prepared: Any,
        backbone_name: str,
        args: Any,
        seed: int,
        configuration_id: str,
        checkpoint_dir: Path,
    ) -> Any:
        del configuration_id, checkpoint_dir
        base_id = _base_cache_id(args, prepared.name, backbone_name)
        shared_dir = output_root / "shared_base_checkpoints"
        return original_fit_or_load_base(
            prepared, backbone_name, args, seed, base_id, shared_dir
        )

    run_main._configuration = extended_configuration
    run_main._fit_or_load_base = shared_base_fit
    core_tokens = _cli_tokens(
        spec["core_settings"], core_output, bool(spec.get("retry_errors", False))
    )
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "started_at": _utc_now(),
        "execution_id": spec["execution_id"],
        "analyses": spec["analyses"],
        "patches": spec.get("patches", []),
        "core_tokens": core_tokens,
        "runner_sha256": runner_hash,
        "core_sha256": {
            name: _sha256(PROJECT_DIR / name)
            for name in ("run_main.py", "data_pipeline.py", "models.py", "training.py")
        },
    }
    try:
        args = run_main._parse_args(core_tokens)
        code = int(run_main.run(args))
        sidecar["exit_code"] = code
        sidecar["finished_at"] = _utc_now()
        _atomic_json(core_output / "extension_worker.json", sidecar)
        return code
    except BaseException as exc:
        sidecar["exit_code"] = 1
        sidecar["finished_at"] = _utc_now()
        sidecar["worker_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        _atomic_json(core_output / "extension_worker.json", sidecar)
        raise


def _latest_core_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (
            row.get("configuration_id"),
            row.get("dataset"),
            row.get("backbone"),
            row.get("condition"),
            row.get("seed"),
        )
        latest[key] = row
    return list(latest.values())


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _flatten_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis_id": row.get("analysis_id"),
        "suite": row.get("suite"),
        "factor": row.get("factor"),
        "level": row.get("level"),
        "execution_id": row.get("execution_id"),
        "status": row.get("status"),
        "family": row.get("family"),
        "dataset": row.get("dataset"),
        "backbone": row.get("backbone"),
        "condition": row.get("condition"),
        "seed": row.get("seed"),
        "auprc": _nested(row, "metrics", "auprc"),
        "auroc": _nested(row, "metrics", "auroc"),
        "forecast_loss": row.get("forecast_loss"),
        "gate_mean": row.get("gate_mean"),
        "selected_memory_weight": _nested(
            row, "score", "selected_parameters", "memory_weight"
        ),
        "selected_agreement_weight": _nested(
            row, "score", "selected_parameters", "agreement_weight"
        ),
        "selected_top_q": _nested(row, "score", "selected_parameters", "top_q"),
        "selected_smoothing": _nested(
            row, "score", "selected_parameters", "causal_smoothing"
        ),
        "latency_ms_per_window": _nested(row, "latency", "total_ms_per_window"),
        "windows_per_second": _nested(
            row, "latency", "end_to_end_windows_per_second"
        ),
        "evaluated_points": _nested(row, "metrics", "n_evaluated"),
        "positive_points": _nested(row, "metrics", "positives"),
        "error_type": _nested(row, "error", "type"),
        "error_message": _nested(row, "error", "message"),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def summarize(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    plan_path = output_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    expanded: list[dict[str, Any]] = []
    job_status: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        job_dir = output_dir / job["relative_output_dir"]
        rows = _latest_core_rows(job_dir / "results.jsonl")
        expected = len(job["core_settings"]["seeds"])
        ok_count = sum(row.get("status") == "ok" for row in rows)
        error_count = sum(row.get("status") != "ok" for row in rows)
        job_status.append(
            {
                "execution_id": job["execution_id"],
                "dataset": job["dataset"],
                "family": job["family"],
                "backbone": job["backbone"],
                "expected_rows": expected,
                "recorded_rows": len(rows),
                "ok_rows": ok_count,
                "error_rows": error_count,
                "complete": int(len(rows) == expected and error_count == 0),
                "relative_output_dir": job["relative_output_dir"],
            }
        )
        for raw in rows:
            for analysis in job["analyses"]:
                annotated = dict(raw)
                annotated.update(analysis)
                annotated["execution_id"] = job["execution_id"]
                annotated["family"] = job["family"]
                annotated["patches"] = job["patches"]
                annotated["core_settings"] = job["core_settings"]
                expanded.append(annotated)

    expanded.sort(
        key=lambda row: (
            str(row.get("suite")),
            str(row.get("factor")),
            str(row.get("level")),
            str(row.get("family")),
            str(row.get("dataset")),
            str(row.get("backbone")),
            int(row.get("seed", -1)),
        )
    )
    _atomic_jsonl(output_dir / "extended_results.jsonl", expanded)
    flat_rows = [_flatten_result(row) for row in expanded]
    flat_columns = list(flat_rows[0]) if flat_rows else [
        "analysis_id", "suite", "factor", "level", "status"
    ]
    _atomic_csv(output_dir / "extended_results.csv", flat_rows, flat_columns)
    _atomic_csv(
        output_dir / "job_status.csv",
        job_status,
        list(job_status[0]) if job_status else ["execution_id", "complete"],
    )

    selected_families = list(plan["design"]["families"])
    expected_physical = {
        family: sum(_family_for(dataset) == family
                    for dataset in plan["design"]["physical_datasets"])
        for family in selected_families
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        if row.get("status") != "ok":
            continue
        key = (
            row["analysis_id"],
            row["suite"],
            row["factor"],
            row["level"],
            row["family"],
            row["backbone"],
            int(row["seed"]),
        )
        grouped[key].append(row)

    family_seed_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        analysis_id, suite, factor, level, family, backbone, seed = key
        datasets = sorted({str(row["dataset"]) for row in rows})
        expected_count = expected_physical[family]
        family_seed_rows.append(
            {
                "analysis_id": analysis_id,
                "suite": suite,
                "factor": factor,
                "level": level,
                "family": family,
                "backbone": backbone,
                "seed": seed,
                "auprc": _mean([float(_nested(row, "metrics", "auprc")) for row in rows]),
                "auroc": _mean([float(_nested(row, "metrics", "auroc")) for row in rows]),
                "physical_count": len(datasets),
                "expected_physical_count": expected_count,
                "complete": int(len(datasets) == expected_count),
                "physical_datasets": ";".join(datasets),
            }
        )

    # Add an equal-family macro only when all selected families are complete for
    # the same analysis/backbone/seed.  This prevents SMD's three machines from
    # receiving triple weight.
    macro_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in family_seed_rows:
        if row["complete"]:
            key = (
                row["analysis_id"], row["suite"], row["factor"], row["level"],
                row["backbone"], row["seed"],
            )
            macro_groups[key].append(row)
    for key, rows in macro_groups.items():
        present = {row["family"] for row in rows}
        if present != set(selected_families):
            continue
        analysis_id, suite, factor, level, backbone, seed = key
        family_seed_rows.append(
            {
                "analysis_id": analysis_id,
                "suite": suite,
                "factor": factor,
                "level": level,
                "family": "MACRO5" if len(selected_families) == 5 else "MACRO",
                "backbone": backbone,
                "seed": seed,
                "auprc": _mean([float(row["auprc"]) for row in rows]),
                "auroc": _mean([float(row["auroc"]) for row in rows]),
                "physical_count": len(rows),
                "expected_physical_count": len(selected_families),
                "complete": 1,
                "physical_datasets": ";".join(selected_families),
            }
        )

    family_seed_rows.sort(
        key=lambda row: (
            row["suite"], row["factor"], row["level"], row["family"],
            row["backbone"], row["seed"],
        )
    )
    seed_columns = list(family_seed_rows[0]) if family_seed_rows else [
        "analysis_id", "suite", "factor", "level", "family", "backbone", "seed"
    ]
    _atomic_csv(output_dir / "family_seed_results.csv", family_seed_rows, seed_columns)

    summary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in family_seed_rows:
        if row["complete"]:
            summary_groups[
                (
                    row["analysis_id"], row["suite"], row["factor"], row["level"],
                    row["family"], row["backbone"],
                )
            ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, rows in summary_groups.items():
        analysis_id, suite, factor, level, family, backbone = key
        auprc_values = [float(row["auprc"]) for row in rows]
        auroc_values = [float(row["auroc"]) for row in rows]
        summary_rows.append(
            {
                "analysis_id": analysis_id,
                "suite": suite,
                "factor": factor,
                "level": level,
                "family": family,
                "backbone": backbone,
                "seed_count": len(rows),
                "auprc_mean": _mean(auprc_values),
                "auprc_std": _sample_std(auprc_values),
                "auroc_mean": _mean(auroc_values),
                "auroc_std": _sample_std(auroc_values),
                "seeds": ";".join(str(row["seed"]) for row in rows),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            row["suite"], row["factor"], row["level"], row["family"], row["backbone"]
        )
    )
    summary_columns = list(summary_rows[0]) if summary_rows else [
        "analysis_id", "suite", "factor", "level", "family", "backbone"
    ]
    _atomic_csv(output_dir / "family_summary.csv", summary_rows, summary_columns)
    _atomic_csv(
        output_dir / "module_ablation_summary.csv",
        [row for row in summary_rows if row["suite"] == "module"],
        summary_columns,
    )
    _atomic_csv(
        output_dir / "sensitivity_summary.csv",
        [row for row in summary_rows if row["suite"] == "sensitivity"],
        summary_columns,
    )

    module_full = {
        (row["family"], row["backbone"]): row
        for row in summary_rows
        if row["suite"] == "module" and row["level"] == "full"
    }
    module_effects: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["suite"] != "module" or row["level"] == "full":
            continue
        full = module_full.get((row["family"], row["backbone"]))
        if full is None:
            continue
        ablated_auprc = float(row["auprc_mean"])
        ablated_auroc = float(row["auroc_mean"])
        full_auprc = float(full["auprc_mean"])
        full_auroc = float(full["auroc_mean"])
        module_effects.append(
            {
                "removed_component": row["level"],
                "family": row["family"],
                "backbone": row["backbone"],
                "seed_count": min(int(row["seed_count"]), int(full["seed_count"])),
                "full_auprc": full_auprc,
                "ablated_auprc": ablated_auprc,
                "auprc_full_minus_ablated": full_auprc - ablated_auprc,
                "auprc_relative_contribution_percent": (
                    100.0 * (full_auprc / ablated_auprc - 1.0)
                    if ablated_auprc != 0.0 else None
                ),
                "full_auroc": full_auroc,
                "ablated_auroc": ablated_auroc,
                "auroc_full_minus_ablated": full_auroc - ablated_auroc,
                "auroc_relative_contribution_percent": (
                    100.0 * (full_auroc / ablated_auroc - 1.0)
                    if ablated_auroc != 0.0 else None
                ),
            }
        )
    effect_columns = list(module_effects[0]) if module_effects else [
        "removed_component", "family", "backbone"
    ]
    _atomic_csv(output_dir / "module_effects.csv", module_effects, effect_columns)

    report = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "plan_id": plan["plan_id"],
        "planned_jobs": len(plan["jobs"]),
        "complete_jobs": sum(int(row["complete"]) for row in job_status),
        "incomplete_jobs": sum(not bool(row["complete"]) for row in job_status),
        "expanded_result_rows": len(expanded),
        "successful_result_rows": sum(row.get("status") == "ok" for row in expanded),
        "error_result_rows": sum(row.get("status") == "error" for row in expanded),
        "artifacts": {
            "raw_jsonl": "extended_results.jsonl",
            "flat_csv": "extended_results.csv",
            "family_seed_csv": "family_seed_results.csv",
            "family_summary_csv": "family_summary.csv",
            "module_csv": "module_ablation_summary.csv",
            "module_effects_csv": "module_effects.csv",
            "sensitivity_csv": "sensitivity_summary.csv",
            "job_status_csv": "job_status.csv",
        },
    }
    _atomic_json(output_dir / "summary_manifest.json", report)
    return report


def _run_plan(options: argparse.Namespace) -> int:
    plan, output_dir = _ensure_plan(options)
    events_path = output_dir / "executor_events.jsonl"
    selected = plan["jobs"]
    if options.only_job:
        selected = [job for job in selected if job["job_id"] == options.only_job]
        if not selected:
            raise ValueError(f"Unknown --only-job {options.only_job!r}")
    if options.max_jobs is not None:
        selected = selected[: options.max_jobs]
    failures = 0
    for position, job in enumerate(selected, start=1):
        spec = dict(job)
        spec["output_root"] = str(output_dir)
        spec["retry_errors"] = bool(options.retry_errors)
        spec_path = output_dir / "job_specs" / f"{job['job_id']}.json"
        _atomic_json(spec_path, spec)
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--spec", str(spec_path)]
        log_path = output_dir / "logs" / f"{job['job_id']}.log"
        print(
            f"[{position}/{len(selected)}] {job['job_id']} "
            f"{job['dataset']} {job['backbone']} "
            f"analyses={','.join(member['level'] for member in job['analyses'])}",
            flush=True,
        )
        _append_event(
            events_path,
            {"timestamp_utc": _utc_now(), "event": "start", "job_id": job["job_id"]},
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n[{_utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        _append_event(
            events_path,
            {
                "timestamp_utc": _utc_now(),
                "event": "finish",
                "job_id": job["job_id"],
                "exit_code": completed.returncode,
                "log": str(log_path),
            },
        )
        if completed.returncode:
            failures += 1
            print(f"  FAILED exit={completed.returncode}; see {log_path}", file=sys.stderr)
            if options.fail_fast:
                break
        summarize(output_dir)
    report = summarize(output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--suites", nargs="+", choices=("module", "sensitivity"),
        default=["module", "sensitivity"],
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=FAMILY_ORDER, default=list(FAMILY_ORDER),
        help="Dataset families; SMD/MSL/SMAP expand to all supplied entities.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("patch_transformer", "period_conv", "selective_ssm"),
        default=["patch_transformer"],
    )
    parser.add_argument("--module-seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--sensitivity-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--crop-seed", type=int, default=20_260_923)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--forecast-horizon", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--base-epochs", type=int, default=15)
    parser.add_argument("--condition-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-validation-block-size", type=int, default=2048)
    parser.add_argument(
        "--sensitivity-factors",
        nargs="+",
        choices=("k", "memory", "context", "score_memory_weight"),
        default=["k", "memory", "context", "score_memory_weight"],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--memory-values", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--context-values", nargs="+", type=int, default=[96, 192, 384])
    parser.add_argument(
        "--score-memory-weight-values",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 1.0, 4.0],
    )


def _validate_options(options: argparse.Namespace) -> None:
    for name in ("datasets", "backbones", "module_seeds", "sensitivity_seeds", "suites"):
        values = list(getattr(options, name))
        if len(values) != len(set(values)):
            raise ValueError(f"--{name.replace('_', '-')} contains duplicates")
    if any(value < 0 for values in (options.module_seeds, options.sensitivity_seeds)
           for value in values):
        raise ValueError("Seeds must be non-negative")
    if any(value <= 0 for values in
           (options.k_values, options.memory_values, options.context_values)
           for value in values):
        raise ValueError("K, memory sizes, and context lengths must be positive")
    if any(value < 0 for value in options.score_memory_weight_values):
        raise ValueError("Score weights must be non-negative")
    if max(options.k_values) > min(options.memory_values):
        raise ValueError("Every K value must be no larger than the smallest memory size")


def _self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="rag-tsad-extension-") as temporary:
        parser = _parser()
        options = parser.parse_args(
            [
                "plan", "--output-dir", temporary, "--datasets", "PSM",
                "--suites", "module", "--module-seeds", "42", "52",
            ]
        )
        _validate_options(options)
        plan = _build_plan(options)
        if len(plan["jobs"]) != len(MODULE_VARIANTS):
            raise AssertionError("module plan cardinality mismatch")
        levels = {
            member["level"]
            for job in plan["jobs"]
            for member in job["analyses"]
        }
        if levels != {variant["level"] for variant in MODULE_VARIANTS}:
            raise AssertionError("module plan levels mismatch")
        _atomic_json(Path(temporary) / "plan.json", plan)
        report = summarize(Path(temporary))
        if report["planned_jobs"] != len(MODULE_VARIANTS):
            raise AssertionError("empty-plan summary mismatch")
    print("extended runner self-test: ok")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run five-family SRF module ablations and parameter sensitivity."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="Write/validate the immutable run plan.")
    _add_plan_arguments(plan_parser)
    run_parser = subparsers.add_parser("run", help="Execute or resume the planned jobs.")
    _add_plan_arguments(run_parser)
    run_parser.add_argument("--retry-errors", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--max-jobs", type=int)
    run_parser.add_argument("--only-job")
    summary_parser = subparsers.add_parser("summarize", help="Rebuild all aggregate files.")
    summary_parser.add_argument("--output-dir", required=True)
    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--spec", required=True)
    subparsers.add_parser("self-test", help="Run static planner/aggregation checks.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(argv)
    if options.command in {"plan", "run"}:
        _validate_options(options)
    if options.command == "plan":
        plan, output_dir = _ensure_plan(options)
        print(
            json.dumps(
                {
                    "plan_id": plan["plan_id"],
                    "jobs": len(plan["jobs"]),
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if options.command == "run":
        return _run_plan(options)
    if options.command == "summarize":
        print(json.dumps(summarize(Path(options.output_dir)), ensure_ascii=False, indent=2))
        return 0
    if options.command == "_worker":
        return _worker(Path(options.spec))
    if options.command == "self-test":
        return _self_test()
    parser.error(f"Unhandled command: {options.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
