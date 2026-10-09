"""Full-data, mechanism-matched plugin comparison; all reference rows run live.

Style variants preserve the original lightweight implementations, not complete
published architectures. No archived metrics/checkpoints/manifests are bundled.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np



CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))
from full_data import discover_datasets, family_of

@dataclass
class RetrievalArrays:
    analog: np.ndarray
    distance: np.ndarray
    divergence: np.ndarray


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def import_runtime(source_dir: Path):
    sys.path.insert(0, str(source_dir))
    import run_main as runtime  # type: ignore

    return runtime


def load_profile_args(runtime, source_dir: Path, dataset: str, options):
    args = runtime._parse_args([
        "--data-root", options.data_root, "--datasets", dataset,
        "--output-dir", str(options.output_dir), "--device", options.device,
        "--seeds", str(options.seed), *options.runtime_args,
    ])
    args.backbones = ["patch_transformer"]
    args.conditions = ["native"]
    args.retry_errors = False
    configuration = runtime._configuration(args)
    configuration_id = runtime._configuration_id(configuration)
    return args, options.output_dir / "checkpoints", None, configuration_id


def as_retrieval(dataset) -> RetrievalArrays:
    if dataset.analog is None or dataset.distance is None or dataset.divergence is None:
        raise RuntimeError("retrieval arrays are incomplete")
    return RetrievalArrays(
        analog=np.asarray(dataset.analog, dtype=np.float32),
        distance=np.asarray(dataset.distance, dtype=np.float32),
        divergence=np.asarray(dataset.divergence, dtype=np.float32),
    )


def candidate_weights(distance: np.ndarray, mode: str, temperature: float = 0.15) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"candidate distances must be [N,K,C], got {values.shape}")
    sentinel = 0.5 * np.finfo(np.float32).max
    valid = np.isfinite(values) & (values < sentinel)
    safe = np.where(valid, values, 0.0)
    if mode == "nearest":
        weights = np.zeros_like(safe)
        first = np.argmax(valid, axis=1)
        any_valid = valid.any(axis=1)
        n_index, c_index = np.indices(any_valid.shape)
        weights[n_index[any_valid], first[any_valid], c_index[any_valid]] = 1.0
    elif mode == "uniform":
        weights = valid.astype(np.float64)
    elif mode == "idw":
        nearest = np.where(valid, safe, np.inf).min(axis=1, keepdims=True)
        scale = np.where(np.isfinite(nearest), np.maximum(nearest, 1e-3), 1.0)
        weights = np.where(valid, 1.0 / (1e-4 + safe / scale), 0.0)
    elif mode in {"softmax", "sparse"}:
        temperature = max(float(temperature), 1e-6)
        shifted = safe - np.where(valid, safe, np.inf).min(axis=1, keepdims=True)
        logits = np.where(valid, -shifted / temperature, -np.inf)
        maxima = np.max(logits, axis=1, keepdims=True)
        maxima[~np.isfinite(maxima)] = 0.0
        weights = np.where(valid, np.exp(logits - maxima), 0.0)
        if mode == "sparse":
            preliminary = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            threshold = 0.5 / max(values.shape[1], 1)
            weights = np.where(preliminary >= threshold, preliminary, 0.0)
    else:
        raise ValueError(f"unknown candidate weighting mode {mode!r}")
    denominator = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)


def candidate_forecast(
    native: np.ndarray,
    retrieval: RetrievalArrays,
    mode: str,
    temperature: float = 0.15,
) -> np.ndarray:
    candidates = np.asarray(retrieval.analog, dtype=np.float64)
    if candidates.ndim == 3:
        candidates = candidates[:, None, :, :]
    if candidates.ndim != 4:
        raise ValueError(f"candidate futures must be [N,K,H,C], got {candidates.shape}")
    weights = candidate_weights(retrieval.distance, mode, temperature)
    forecast = np.einsum("nkc,nkhc->nhc", weights, candidates)
    has_reference = weights.sum(axis=1) > 0
    return np.where(has_reference[:, None, :], forecast, native).astype(np.float32)


def mean_absolute_error(prediction: np.ndarray, target: np.ndarray) -> float:
    values = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    finite = np.isfinite(values)
    return float(values[finite].mean()) if np.any(finite) else math.inf


def choose_temperature(
    native: np.ndarray,
    target: np.ndarray,
    retrieval: RetrievalArrays,
    mode: str,
) -> tuple[float, dict[str, float]]:
    losses: dict[str, float] = {}
    for temperature in (0.05, 0.15, 0.50, 1.00):
        forecast = candidate_forecast(native, retrieval, mode, temperature)
        losses[f"{temperature:.2f}"] = mean_absolute_error(forecast, target)
    best = min(losses, key=lambda key: (losses[key], float(key)))
    return float(best), losses


def make_prediction(predictions: np.ndarray, dataset, retrieval: RetrievalArrays | None):
    result = {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "targets": np.asarray(dataset.futures, dtype=np.float32),
        "end_indices": np.asarray(dataset.end_indices, dtype=np.int64),
    }
    if retrieval is not None:
        result["distance"] = retrieval.distance
        result["divergence"] = retrieval.divergence
    return result


def score_plugin(
    runtime,
    prepared,
    args,
    forecast: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    retrieval: Mapping[str, RetrievalArrays] | None,
    *,
    use_memory_score: bool,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    prediction = {
        split: make_prediction(
            forecast[split], arrays[split], None if retrieval is None else retrieval[split]
        )
        for split in ("normal", "score", "test")
    }
    validation_errors = prediction["normal"]["predictions"] - prediction["normal"]["targets"]
    score_memory = None
    test_memory = None
    if use_memory_score:
        if retrieval is None:
            raise ValueError("memory scoring requested without retrieval arrays")
        normal_distance = runtime._aggregate_candidate_distance(retrieval["normal"].distance)
        score_memory = runtime._empirical_transform(
            normal_distance,
            runtime._aggregate_candidate_distance(retrieval["score"].distance),
            tail_score=True,
        )
        test_memory = runtime._empirical_transform(
            normal_distance,
            runtime._aggregate_candidate_distance(retrieval["test"].distance),
            tail_score=True,
        )
    score_components = runtime._score_channel_components(
        prediction["score"],
        validation_errors,
        len(prepared.score_validation_labels),
        score_memory,
        None,
    )
    parameters, validation_metrics = runtime._select_score_parameters(
        prepared.score_validation_labels,
        score_components,
        args,
        use_memory_score,
        prepared.score_validation_segment_lengths,
    )
    test_components = runtime._score_channel_components(
        prediction["test"],
        validation_errors,
        len(prepared.test_labels),
        test_memory,
        None,
    )
    test_scores = runtime._apply_score_parameters(
        test_components, parameters, prepared.test_segment_lengths
    )
    metrics = runtime.evaluate_scores(prepared.test_labels, test_scores)
    normal_loss = mean_absolute_error(
        prediction["normal"]["predictions"], prediction["normal"]["targets"]
    )
    return metrics, {"parameters": parameters, "metrics": validation_metrics}, normal_loss


def fit_adaptive_mixer(
    native: np.ndarray,
    analogue: np.ndarray,
    target: np.ndarray,
    retrieval: RetrievalArrays,
    seed: int,
):
    nearest = np.min(np.asarray(retrieval.distance, dtype=np.float64), axis=1)
    nearest = np.nan_to_num(nearest, nan=1e6, posinf=1e6, neginf=0.0)
    divergence = np.mean(np.asarray(retrieval.divergence, dtype=np.float64), axis=1)
    disagreement = np.mean(np.abs(analogue - native), axis=1)
    native_error = np.mean(np.abs(native - target), axis=1)
    analogue_error = np.mean(np.abs(analogue - target), axis=1)
    features = np.stack(
        [np.log1p(np.clip(nearest, 0.0, 1e6)), np.log1p(np.clip(divergence, 0.0, 1e6)), disagreement],
        axis=-1,
    ).reshape(-1, 3)
    labels = (analogue_error < native_error).reshape(-1).astype(np.int8)
    finite = np.isfinite(features).all(axis=1)
    features = features[finite]
    labels = labels[finite]
    if len(np.unique(labels)) < 2:
        return {"constant": float(labels.mean()) if len(labels) else 0.0}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=200,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(features, labels)
    return model


def adaptive_mix(
    model,
    native: np.ndarray,
    analogue: np.ndarray,
    retrieval: RetrievalArrays,
) -> tuple[np.ndarray, float]:
    nearest = np.min(np.asarray(retrieval.distance, dtype=np.float64), axis=1)
    nearest = np.nan_to_num(nearest, nan=1e6, posinf=1e6, neginf=0.0)
    divergence = np.mean(np.asarray(retrieval.divergence, dtype=np.float64), axis=1)
    disagreement = np.mean(np.abs(analogue - native), axis=1)
    features = np.stack(
        [np.log1p(np.clip(nearest, 0.0, 1e6)), np.log1p(np.clip(divergence, 0.0, 1e6)), disagreement],
        axis=-1,
    )
    shape = features.shape[:2]
    if isinstance(model, Mapping):
        probability = np.full(shape, float(model["constant"]), dtype=np.float64)
    else:
        probability = model.predict_proba(features.reshape(-1, 3))[:, 1].reshape(shape)
    mixed = native + probability[:, None, :] * (analogue - native)
    return mixed.astype(np.float32), float(np.mean(probability))


def fit_prototype_bank(runtime, prepared, args):
    memory_per_channel = min(64, int(args.memory_per_channel))
    return runtime.ChannelMemoryBank(
        memory_per_channel=memory_per_channel,
        top_k=int(args.retrieval_top_k),
        temperature=float(args.memory_temperature),
        selection="farthest",
    ).fit(
        prepared.base.contexts,
        prepared.base.futures,
        end_indices=prepared.base.end_indices,
    )


def query_prototype_bank(bank, dataset) -> RetrievalArrays:
    analog, distance, divergence = bank.query(
        dataset.contexts,
        top_k=bank.top_k,
        return_candidates=True,
    )
    return RetrievalArrays(analog=analog, distance=distance, divergence=divergence)


def predict_base(runtime, base, dataset, args) -> np.ndarray:
    output = runtime.predict_windows(
        base,
        dataset,
        batch_size=int(args.batch_size),
        device=args.device,
        amp=runtime._amp_for("patch_transformer", args),
        return_aux=False,
    )
    return np.asarray(output["predictions"], dtype=np.float32)


def run_dataset(runtime, source_dir: Path, dataset: str, options) -> list[dict[str, Any]]:
    started = time.perf_counter()
    args, checkpoint_dir, manifest_path, base_configuration_id = load_profile_args(
        runtime, source_dir, dataset, options
    )
    prepared = runtime._prepare_dataset(dataset, args, int(options.seed), need_retrieval=True)
    base, base_training, base_source = runtime._fit_or_load_base(
        prepared,
        "patch_transformer",
        args,
        int(options.seed),
        base_configuration_id,
        checkpoint_dir,
    )
    arrays = {
        "adapt": prepared.adapt_plain,
        "normal": prepared.validation_plain,
        "score": prepared.score_validation_plain,
        "test": prepared.test_plain,
    }
    retrieval_datasets = {
        "adapt": prepared.adapt_retrieval,
        "normal": prepared.validation_retrieval,
        "score": prepared.score_validation_retrieval,
        "test": prepared.test_retrieval,
    }
    if any(value is None for value in retrieval_datasets.values()):
        raise RuntimeError("prepared dataset did not include every retrieval split")
    retrieval = {key: as_retrieval(value) for key, value in retrieval_datasets.items()}
    native = {key: predict_base(runtime, base, value, args) for key, value in arrays.items()}

    plugin_forecasts: dict[str, dict[str, np.ndarray]] = {"Native audit": dict(native)}
    plugin_retrieval: dict[str, Mapping[str, RetrievalArrays] | None] = {"Native audit": None}
    plugin_metadata: dict[str, dict[str, Any]] = {"Native audit": {"trainable": False}}

    for name, mode in (
        ("RAFT-style 1NN", "nearest"),
        ("RAFT-style UniformK", "uniform"),
        ("kNN-IDW", "idw"),
    ):
        plugin_forecasts[name] = {
            key: candidate_forecast(native[key], retrieval[key], mode) for key in arrays
        }
        plugin_retrieval[name] = retrieval
        plugin_metadata[name] = {"weighting": mode, "top_k": int(args.retrieval_top_k)}

    soft_temperature, soft_losses = choose_temperature(
        native["normal"], arrays["normal"].futures, retrieval["normal"], "softmax"
    )
    soft_forecast = {
        key: candidate_forecast(native[key], retrieval[key], "softmax", soft_temperature)
        for key in arrays
    }
    plugin_forecasts["RAFT-style SoftK"] = soft_forecast
    plugin_retrieval["RAFT-style SoftK"] = retrieval
    plugin_metadata["RAFT-style SoftK"] = {
        "weighting": "softmax",
        "temperature": soft_temperature,
        "normal_validation_temperature_losses": soft_losses,
    }

    sparse_temperature, sparse_losses = choose_temperature(
        native["normal"], arrays["normal"].futures, retrieval["normal"], "sparse"
    )
    plugin_forecasts["MemAE-style SparseAttn"] = {
        key: candidate_forecast(native[key], retrieval[key], "sparse", sparse_temperature)
        for key in arrays
    }
    plugin_retrieval["MemAE-style SparseAttn"] = retrieval
    plugin_metadata["MemAE-style SparseAttn"] = {
        "weighting": "hard-shrunk sparse softmax",
        "temperature": sparse_temperature,
        "normal_validation_temperature_losses": sparse_losses,
    }

    mixer = fit_adaptive_mixer(
        native["adapt"],
        soft_forecast["adapt"],
        arrays["adapt"].futures,
        retrieval["adapt"],
        int(options.seed),
    )
    mixer_forecasts: dict[str, np.ndarray] = {}
    gate_means: dict[str, float] = {}
    for key in arrays:
        mixer_forecasts[key], gate_means[key] = adaptive_mix(
            mixer, native[key], soft_forecast[key], retrieval[key]
        )
    plugin_forecasts["TS-RAG-style ARM"] = mixer_forecasts
    plugin_retrieval["TS-RAG-style ARM"] = retrieval
    plugin_metadata["TS-RAG-style ARM"] = {
        "mixer": "normal-only logistic adaptive retrieval gate",
        "gate_means": gate_means,
        "temperature": soft_temperature,
    }

    prototype_bank = fit_prototype_bank(runtime, prepared, args)
    prototype_retrieval = {
        key: query_prototype_bank(prototype_bank, value) for key, value in arrays.items()
    }
    prototype_temperature, prototype_losses = choose_temperature(
        native["normal"],
        arrays["normal"].futures,
        prototype_retrieval["normal"],
        "softmax",
    )
    plugin_forecasts["ProtoMem-style Coreset"] = {
        key: candidate_forecast(
            native[key], prototype_retrieval[key], "softmax", prototype_temperature
        )
        for key in arrays
    }
    plugin_retrieval["ProtoMem-style Coreset"] = prototype_retrieval
    plugin_metadata["ProtoMem-style Coreset"] = {
        "memory": "64-per-channel farthest-point normal prototypes",
        "temperature": prototype_temperature,
        "normal_validation_temperature_losses": prototype_losses,
    }

    # Matrix Profile/discord-style distance augmentation retains the base
    # forecast and uses nearest-normal-subsequence distance as the plugin score.
    plugin_forecasts["MP-style NN-Score"] = dict(native)
    plugin_retrieval["MP-style NN-Score"] = retrieval
    plugin_metadata["MP-style NN-Score"] = {
        "forecast": "unchanged Native forecast",
        "score": "normal-validation-calibrated nearest-neighbour distance",
    }

    rows: list[dict[str, Any]] = []
    for plugin, forecasts in plugin_forecasts.items():
        use_memory_score = plugin != "Native audit"
        metrics, selection, normal_loss = score_plugin(
            runtime,
            prepared,
            args,
            forecasts,
            arrays,
            plugin_retrieval[plugin],
            use_memory_score=use_memory_score,
        )
        rows.append(
            {
                "family": family_of(dataset),
                "dataset": dataset,
                "backbone": "patch_transformer",
                "plugin": plugin,
                "seed": int(options.seed),
                "auprc": metrics.get("auprc"),
                "auroc": metrics.get("auroc"),
                "normal_validation_mae": normal_loss,
                "score_top_q": selection["parameters"].get("top_q"),
                "score_memory_weight": selection["parameters"].get("memory_weight"),
                "score_smoothing": selection["parameters"].get("causal_smoothing"),
                "labeled_validation_auprc": selection["metrics"].get("auprc"),
                "base_checkpoint_source": base_source,
                "base_configuration_id": base_configuration_id,
                "full_data": not args.quick,
                "plugin_metadata_json": json.dumps(
                    json_safe(plugin_metadata[plugin]), ensure_ascii=False, sort_keys=True
                ),
                "elapsed_dataset_seconds": time.perf_counter() - started,
                "point_adjustment": False,
                "test_label_tuning": False,
            }
        )
        print(
            f"[{dataset}] {plugin}: AUPRC={metrics.get('auprc'):.6f} "
            f"AUROC={metrics.get('auroc'):.6f}",
            flush=True,
        )
    for condition, name in (("srf", "SRF-Only"), ("joint", "SRF+LoRA")):
        result = runtime._run_condition(
            base, base_training, base_source, prepared, "patch_transformer",
            condition, args, int(options.seed), base_configuration_id,
        )
        rows.append({
            "family": family_of(dataset), "dataset": dataset,
            "backbone": "patch_transformer", "plugin": name,
            "seed": int(options.seed), "auprc": result["metrics"]["auprc"],
            "auroc": result["metrics"]["auroc"], "detail": result,
            "point_adjustment": False, "test_label_tuning": False,
        })
    return rows



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("RAG_TSAD_DATA_ROOT", "./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--datasets", nargs="+", default=["PSM", "MSL", "SMAP", "SMD", "SWaT"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime-args", nargs=argparse.REMAINDER, default=[],
                        help="Optional run_main arguments, e.g. --base-epochs 25.")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = import_runtime(CORE_DIR)
    datasets = discover_datasets(options.data_root, options.datasets)
    rows = []
    results = options.output_dir / "results.jsonl"
    if results.exists():
        rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {(r["dataset"], int(r["seed"])) for r in rows}
    for seed in options.seeds:
        options.seed = seed
        for dataset in datasets:
            if (dataset, seed) in completed:
                continue
            current = run_dataset(runtime, CORE_DIR, dataset, options)
            # One atomic dataset/seed unit: no partial row counts as completed.
            rows.extend(current)
            temporary = results.with_suffix(".tmp")
            temporary.write_text("\n".join(json.dumps(json_safe(row), ensure_ascii=False) for row in rows) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, results)
    table = [{k:r.get(k) for k in ("family", "dataset", "backbone", "plugin", "seed", "auprc", "auroc")} for r in rows]
    write_csv(options.output_dir / "comparison.csv", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
