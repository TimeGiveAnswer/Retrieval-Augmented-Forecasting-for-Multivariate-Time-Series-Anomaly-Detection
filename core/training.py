"""Training, inference, and point-wise evaluation utilities.

The functions in this module deliberately know nothing about a particular
backbone.  A model only needs to implement the following interface::

    model(x, analog=None, distance=None, divergence=None, return_aux=False)

where ``x`` has shape ``[batch, context_length, channels]`` and the returned
forecast has shape ``[batch, horizon, channels]``.  This keeps the experiment
runner identical for Native, SRF-only, LoRA-only, and SRF+LoRA conditions.

No point adjustment is performed anywhere in this module.  Window forecasts
are mapped back to their original timestamps by ordinary overlap-add averaging.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ArrayLike = np.ndarray | Tensor


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Experiment seed.
    deterministic:
        If true, request deterministic PyTorch kernels where they are
        available.  PyTorch is run in ``warn_only`` mode because a fully
        deterministic implementation is not available for every GPU kernel.
    """

    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if deterministic and hasattr(torch.backends, "cuda"):
        # Flash/memory-efficient attention may use nondeterministic backward
        # kernels.  The math backend is slower but makes the single-seed run
        # repeatable on the target RTX 4090.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


class NumpyWindowDataset(Dataset[dict[str, Tensor]]):
    """A zero-copy-friendly dataset for precomputed forecasting windows.

    Parameters
    ----------
    contexts, futures:
        Float arrays shaped ``[N, L, C]`` and ``[N, H, C]``.
    analog:
        Optional retrieved future/reference array.  Its first dimension must
        be ``N``; the model decides how to interpret the remaining dimensions.
    distance, divergence:
        Optional retrieval diagnostics.  Typical shapes are ``[N, C]`` or
        ``[N, H, C]``.
    end_indices:
        Optional integer array shaped ``[N]``.  In this project an end index is
        the *first forecast timestamp* (the context end-exclusive index), so a
        horizon-H forecast covers ``[end_index, end_index + H)``.

    Notes
    -----
    Arrays are converted to NumPy views where possible and individual samples
    are converted to tensors lazily.  This avoids an unnecessary full-dataset
    tensor copy for the reduced experiments.
    """

    def __init__(
        self,
        contexts: ArrayLike,
        futures: ArrayLike,
        *,
        analog: ArrayLike | None = None,
        distance: ArrayLike | None = None,
        divergence: ArrayLike | None = None,
        end_indices: ArrayLike | None = None,
    ) -> None:
        self.contexts = _as_numpy(contexts, dtype=np.float32)
        self.futures = _as_numpy(futures, dtype=np.float32)

        if self.contexts.ndim != 3:
            raise ValueError(
                f"contexts must have shape [N, L, C], got {self.contexts.shape}"
            )
        if self.futures.ndim != 3:
            raise ValueError(
                f"futures must have shape [N, H, C], got {self.futures.shape}"
            )
        if len(self.contexts) != len(self.futures):
            raise ValueError("contexts and futures must contain the same N windows")
        if self.contexts.shape[-1] != self.futures.shape[-1]:
            raise ValueError("contexts and futures must have the same channel count")

        self.analog = self._optional_float_array("analog", analog)
        self.distance = self._optional_float_array("distance", distance)
        self.divergence = self._optional_float_array("divergence", divergence)
        self.end_indices = (
            None
            if end_indices is None
            else _as_numpy(end_indices, dtype=np.int64).reshape(-1)
        )
        if self.end_indices is not None and len(self.end_indices) != len(self):
            raise ValueError("end_indices must contain one value per window")

    def _optional_float_array(
        self, name: str, value: ArrayLike | None
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = _as_numpy(value, dtype=np.float32)
        if array.ndim == 0 or len(array) != len(self.contexts):
            raise ValueError(f"{name} must have first dimension N={len(self.contexts)}")
        return array

    def __len__(self) -> int:
        return int(self.contexts.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item: dict[str, Tensor] = {
            "x": torch.tensor(self.contexts[index], dtype=torch.float32),
            "y": torch.tensor(self.futures[index], dtype=torch.float32),
        }
        for name in ("analog", "distance", "divergence"):
            value = getattr(self, name)
            if value is not None:
                item[name] = torch.as_tensor(value[index], dtype=torch.float32)
        if self.end_indices is not None:
            item["end_indices"] = torch.as_tensor(
                self.end_indices[index], dtype=torch.long
            )
        return item


def train_model(
    model: nn.Module,
    train_arrays: NumpyWindowDataset | Mapping[str, ArrayLike] | Sequence[ArrayLike],
    val_arrays: NumpyWindowDataset | Mapping[str, ArrayLike] | Sequence[ArrayLike],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    amp: bool,
    patience: int,
    *,
    loss: str = "mse",
    huber_delta: float = 1.0,
    weight_decay: float = 1e-4,
    grad_clip: float | None = 1.0,
    min_delta: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
    include_initial_checkpoint: bool = False,
    parameter_groups: Sequence[tuple[str, Iterable[nn.Parameter], float]] | None = None,
    reference_consistency_weight: float = 0.0,
    gate_regularization_weight: float = 0.0,
    reference_quality_margin: float = 0.10,
    reference_quality_temperature: float = 0.05,
) -> dict[str, Any]:
    """Train a forecasting model and restore its best validation checkpoint.

    ``train_arrays`` and ``val_arrays`` may be :class:`NumpyWindowDataset`
    objects, mappings with ``contexts``/``futures`` (or ``x``/``y``) keys, or
    tuples ordered as ``(contexts, futures, analog, distance, divergence,
    end_indices)``.  Missing optional tuple elements are allowed.

    The function mutates ``model`` in place.  The returned dictionary contains
    only JSON-serializable training metadata; model weights are restored from an
    in-memory copy of the best epoch.  Validation loss is computed without
    gradient tracking.  Non-finite target entries are ignored, but a batch with
    no finite targets is rejected.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if patience < 0:
        raise ValueError("patience must be non-negative")
    if reference_consistency_weight < 0 or gate_regularization_weight < 0:
        raise ValueError("auxiliary-loss weights must be non-negative")
    if reference_quality_temperature <= 0:
        raise ValueError("reference_quality_temperature must be positive")

    seed_all(seed)
    train_dataset = _coerce_dataset(train_arrays)
    val_dataset = _coerce_dataset(val_arrays)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("training and validation datasets must not be empty")

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    # Do not let gradients retained by a previous training stage influence
    # clipping in this stage.  This is especially important after deepcopying
    # a freshly fitted base model into several condition wrappers.
    model.zero_grad(set_to_none=True)
    use_amp = bool(amp and resolved_device.type == "cuda")
    scaler = _make_grad_scaler(use_amp)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(
            "model has no trainable parameters; evaluate this condition directly or "
            "enable its adapter/fusion parameters before calling train_model"
        )
    optimizer_groups: list[dict[str, Any]] | None = None
    if parameter_groups is not None:
        optimizer_groups = []
        seen: set[int] = set()
        for group_name, group_parameters, group_lr in parameter_groups:
            current = [
                parameter
                for parameter in group_parameters
                if parameter.requires_grad
            ]
            if not current:
                continue
            if float(group_lr) <= 0:
                raise ValueError(f"parameter-group LR for {group_name!r} must be positive")
            for parameter in current:
                identifier = id(parameter)
                if identifier in seen:
                    raise ValueError("a parameter appears in more than one optimizer group")
                seen.add(identifier)
            optimizer_groups.append(
                {
                    "params": current,
                    "lr": float(group_lr),
                    "group_name": str(group_name),
                }
            )
        missing = [parameter for parameter in trainable_parameters if id(parameter) not in seen]
        if missing:
            raise ValueError(
                f"optimizer parameter groups omit {len(missing)} trainable tensors"
            )
        if not optimizer_groups:
            raise ValueError("optimizer parameter groups contain no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters if optimizer_groups is None else optimizer_groups,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    minimum_lrs = [
        float(group["lr"]) * 0.02 for group in optimizer.param_groups
    ]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, int(patience) // 2),
        threshold=max(float(min_delta), 1e-5),
        min_lr=minimum_lrs,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": resolved_device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    loss_name = loss.strip().lower()
    if loss_name not in {"mse", "huber", "smooth_l1"}:
        raise ValueError("loss must be 'mse' or 'huber'/'smooth_l1'")
    use_srf_auxiliary_loss = bool(
        reference_consistency_weight > 0 or gate_regularization_weight > 0
    )

    def evaluate_batch(
        batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        prediction, auxiliary = _forward_model(
            model, batch, return_aux=use_srf_auxiliary_loss
        )
        prediction_loss, valid_count = _masked_forecast_loss(
            prediction,
            batch["y"],
            loss_name=loss_name,
            huber_delta=float(huber_delta),
        )
        reference_loss, gate_loss = _srf_auxiliary_losses(
            auxiliary,
            batch,
            loss_name=loss_name,
            huber_delta=float(huber_delta),
            enabled=use_srf_auxiliary_loss,
            quality_margin=float(reference_quality_margin),
            quality_temperature=float(reference_quality_temperature),
            anchor=prediction_loss,
        )
        total_loss = (
            prediction_loss
            + float(reference_consistency_weight) * reference_loss
            + float(gate_regularization_weight) * gate_loss
        )
        return prediction_loss, total_loss, reference_loss, gate_loss, valid_count

    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    initial_val_loss: float | None = None
    if include_initial_checkpoint:
        # Identity-initialized adapters must be allowed to keep the exact
        # pre-adaptation solution.  The former loop began at epoch one, which
        # forced a degraded update even when every fine-tuned checkpoint was
        # worse than the fitted Native/SRF model.
        model.eval()
        val_loss_sum = 0.0
        val_total_loss_sum = 0.0
        val_reference_loss_sum = 0.0
        val_gate_loss_sum = 0.0
        val_weight = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _move_batch(batch, resolved_device)
                with _autocast_context(resolved_device, use_amp):
                    (
                        batch_loss,
                        batch_total_loss,
                        batch_reference_loss,
                        batch_gate_loss,
                        valid_count,
                    ) = evaluate_batch(batch)
                val_loss_sum += float(batch_loss.detach().cpu()) * valid_count
                val_total_loss_sum += (
                    float(batch_total_loss.detach().cpu()) * valid_count
                )
                val_reference_loss_sum += (
                    float(batch_reference_loss.detach().cpu()) * valid_count
                )
                val_gate_loss_sum += (
                    float(batch_gate_loss.detach().cpu()) * valid_count
                )
                val_weight += valid_count
        initial_val_loss = val_loss_sum / max(val_weight, 1)
        if not np.isfinite(initial_val_loss):
            raise RuntimeError("initial validation loss was not finite")
        best_loss = float(initial_val_loss)
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "val_loss": float(initial_val_loss),
                "train_total_loss": None,
                "val_total_loss": val_total_loss_sum / max(val_weight, 1),
                "train_reference_loss": None,
                "val_reference_loss": val_reference_loss_sum / max(val_weight, 1),
                "train_gate_loss": None,
                "val_gate_loss": val_gate_loss_sum / max(val_weight, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "group_lrs": {
                    str(group.get("group_name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "epoch_seconds": 0.0,
                "checkpoint_role": "pre_adaptation",
            }
        )

    for epoch in range(int(epochs)):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_total_loss_sum = 0.0
        train_reference_loss_sum = 0.0
        train_gate_loss_sum = 0.0
        train_weight = 0

        for batch in train_loader:
            batch = _move_batch(batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(resolved_device, use_amp):
                (
                    batch_loss,
                    batch_total_loss,
                    batch_reference_loss,
                    batch_gate_loss,
                    valid_count,
                ) = evaluate_batch(batch)

            scaler.scale(batch_total_loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(batch_loss.detach().cpu()) * valid_count
            train_total_loss_sum += (
                float(batch_total_loss.detach().cpu()) * valid_count
            )
            train_reference_loss_sum += (
                float(batch_reference_loss.detach().cpu()) * valid_count
            )
            train_gate_loss_sum += (
                float(batch_gate_loss.detach().cpu()) * valid_count
            )
            train_weight += valid_count

        model.eval()
        val_loss_sum = 0.0
        val_total_loss_sum = 0.0
        val_reference_loss_sum = 0.0
        val_gate_loss_sum = 0.0
        val_weight = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _move_batch(batch, resolved_device)
                with _autocast_context(resolved_device, use_amp):
                    (
                        batch_loss,
                        batch_total_loss,
                        batch_reference_loss,
                        batch_gate_loss,
                        valid_count,
                    ) = evaluate_batch(batch)
                val_loss_sum += float(batch_loss.detach().cpu()) * valid_count
                val_total_loss_sum += (
                    float(batch_total_loss.detach().cpu()) * valid_count
                )
                val_reference_loss_sum += (
                    float(batch_reference_loss.detach().cpu()) * valid_count
                )
                val_gate_loss_sum += (
                    float(batch_gate_loss.detach().cpu()) * valid_count
                )
                val_weight += valid_count

        train_loss = train_loss_sum / max(train_weight, 1)
        train_total_loss = train_total_loss_sum / max(train_weight, 1)
        train_reference_loss = train_reference_loss_sum / max(train_weight, 1)
        train_gate_loss = train_gate_loss_sum / max(train_weight, 1)
        val_loss = val_loss_sum / max(val_weight, 1)
        val_total_loss = val_total_loss_sum / max(val_weight, 1)
        val_reference_loss = val_reference_loss_sum / max(val_weight, 1)
        val_gate_loss = val_gate_loss_sum / max(val_weight, 1)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_total_loss": train_total_loss,
                "val_total_loss": val_total_loss,
                "train_reference_loss": train_reference_loss,
                "val_reference_loss": val_reference_loss,
                "train_gate_loss": train_gate_loss,
                "val_gate_loss": val_gate_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "group_lrs": {
                    str(group.get("group_name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )

        if np.isfinite(val_loss) and val_loss < best_loss - float(min_delta):
            best_loss = float(val_loss)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("validation loss was never finite; no checkpoint can be restored")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    model.zero_grad(set_to_none=True)

    elapsed = time.perf_counter() - training_start
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "initial_val_loss": initial_val_loss,
        "selected_initial_checkpoint": bool(
            include_initial_checkpoint and best_epoch == 0
        ),
        "epochs_ran": len(history) - int(include_initial_checkpoint),
        "stopped_early": (len(history) - int(include_initial_checkpoint)) < int(epochs),
        "elapsed_seconds": elapsed,
        "device": str(resolved_device),
        "amp_enabled": use_amp,
        "loss": loss_name,
        "checkpoint_objective": "normal_validation_prediction_loss",
        "loss_weights": {
            "prediction": 1.0,
            "reference_consistency": float(reference_consistency_weight),
            "gate_regularization": float(gate_regularization_weight),
            "reference_quality_margin": float(reference_quality_margin),
            "reference_quality_temperature": float(reference_quality_temperature),
        },
        "optimizer_groups": [
            {
                "name": str(group.get("group_name", f"group_{index}")),
                "final_lr": float(group["lr"]),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "history": history,
        "parameters": count_parameters(model),
    }


def predict_windows(
    model: nn.Module,
    arrays: NumpyWindowDataset | Mapping[str, ArrayLike] | Sequence[ArrayLike] | ArrayLike,
    futures: ArrayLike | None = None,
    *,
    analog: ArrayLike | None = None,
    distance: ArrayLike | None = None,
    divergence: ArrayLike | None = None,
    end_indices: ArrayLike | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
    amp: bool = True,
    num_workers: int = 0,
    return_aux: bool = True,
    auxiliary_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run window inference and return forecasts, targets, auxiliaries, and timing.

    For convenience, ``arrays`` can use the same container forms accepted by
    :func:`train_model`.  It may also be a raw contexts array when ``futures``
    is supplied separately.  Tensor-valued model auxiliaries (for example
    ``confidence`` or ``gate``) are concatenated over batches when possible.
    The model's train/eval mode is restored after prediction.
    """

    if futures is not None:
        dataset = NumpyWindowDataset(
            arrays,  # type: ignore[arg-type]
            futures,
            analog=analog,
            distance=distance,
            divergence=divergence,
            end_indices=end_indices,
        )
    else:
        dataset = _coerce_dataset(arrays)  # type: ignore[arg-type]
    if len(dataset) == 0:
        raise ValueError("prediction dataset must not be empty")

    resolved_device = _resolve_device(device)
    use_amp = bool(amp and resolved_device.type == "cuda")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=resolved_device.type == "cuda",
    )

    model.to(resolved_device)
    was_training = model.training
    model.eval()
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    input_aux_chunks: dict[str, list[np.ndarray]] = {
        "distance": [],
        "divergence": [],
    }
    model_aux_chunks: dict[str, list[np.ndarray]] = {}

    _synchronize_cuda(resolved_device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, resolved_device)
            with _autocast_context(resolved_device, use_amp):
                prediction, aux = _forward_model(
                    model, batch, return_aux=bool(return_aux)
                )
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("model produced a non-finite forecast during inference")
            for auxiliary_name, auxiliary_value in aux.items():
                if isinstance(auxiliary_value, Tensor) and not bool(
                    torch.isfinite(auxiliary_value).all()
                ):
                    raise RuntimeError(
                        f"model produced non-finite auxiliary output {auxiliary_name!r}"
                    )
            prediction_chunks.append(_tensor_to_numpy(prediction))
            target_chunks.append(_tensor_to_numpy(batch["y"]))
            if "end_indices" in batch:
                index_chunks.append(_tensor_to_numpy(batch["end_indices"]))
            for name in input_aux_chunks:
                if name in batch:
                    input_aux_chunks[name].append(_tensor_to_numpy(batch[name]))
            if return_aux:
                for name, value in aux.items():
                    if auxiliary_names is not None and name not in auxiliary_names:
                        continue
                    array = _value_to_batch_numpy(value, prediction.shape[0])
                    if array is not None:
                        model_aux_chunks.setdefault(name, []).append(array)

    _synchronize_cuda(resolved_device)
    elapsed = time.perf_counter() - start
    if was_training:
        model.train()

    result: dict[str, Any] = {
        "predictions": np.concatenate(prediction_chunks, axis=0),
        "targets": np.concatenate(target_chunks, axis=0),
        "elapsed_seconds": elapsed,
        "windows_per_second": len(dataset) / max(elapsed, np.finfo(float).eps),
        "latency_ms_per_window": 1000.0 * elapsed / len(dataset),
        "device": str(resolved_device),
        "amp_enabled": use_amp,
    }
    if index_chunks:
        result["end_indices"] = np.concatenate(index_chunks, axis=0).astype(
            np.int64, copy=False
        )
    for name, chunks in input_aux_chunks.items():
        if chunks:
            result[name] = np.concatenate(chunks, axis=0)
    for name, chunks in model_aux_chunks.items():
        if chunks and sum(len(chunk) for chunk in chunks) == len(dataset):
            result[name] = np.concatenate(chunks, axis=0)
    return result


def fit_robust_channel_stats(
    residuals: ArrayLike,
    *,
    residual_kind: str = "absolute",
    residuals_are_errors: bool = True,
    eps: float = 1e-6,
) -> dict[str, np.ndarray | str | float]:
    """Fit per-channel median/MAD statistics on *normal validation* residuals.

    ``residuals`` must end in a channel dimension; all leading dimensions are
    pooled.  If ``residuals_are_errors`` is true, absolute/squared magnitudes
    are first computed from signed errors.  MAD is multiplied by 1.4826 to be
    comparable to a standard deviation under a Gaussian model.  Zero-MAD
    channels fall back to their mean absolute deviation and then ``eps``.
    """

    values = _as_numpy(residuals, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("residuals must have at least a sample and channel axis")
    if residuals_are_errors:
        values = _residual_magnitude(values, residual_kind)
    elif residual_kind not in {"absolute", "abs", "squared", "square", "mse"}:
        raise ValueError("residual_kind must be 'absolute' or 'squared'")

    axes = tuple(range(values.ndim - 1))
    center = np.nanmedian(values, axis=axes)
    reshape = (1,) * (values.ndim - 1) + (values.shape[-1],)
    absolute_deviation = np.abs(values - center.reshape(reshape))
    mad = 1.4826 * np.nanmedian(absolute_deviation, axis=axes)
    mean_deviation = np.nanmean(absolute_deviation, axis=axes)
    scale = np.where(np.isfinite(mad) & (mad > eps), mad, mean_deviation)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, float(eps))
    return {
        "center": center.astype(np.float64, copy=False),
        "scale": scale.astype(np.float64, copy=False),
        "residual_kind": _canonical_residual_kind(residual_kind),
        "eps": float(eps),
    }


def overlap_add_scores(
    predictions: ArrayLike,
    targets: ArrayLike,
    end_indices: ArrayLike,
    *,
    validation_residuals: ArrayLike | None = None,
    channel_center: ArrayLike | None = None,
    channel_scale: ArrayLike | None = None,
    residual_kind: str = "absolute",
    top_q: float | int = 0.2,
    memory_distance: ArrayLike | None = None,
    memory_weight: float = 0.0,
    divergence: ArrayLike | None = None,
    divergence_weight: float = 0.0,
    agreement: ArrayLike | None = None,
    agreement_weight: float = 0.0,
    confidence: ArrayLike | None = None,
    confidence_mode: str = "none",
    confidence_floor: float = 0.1,
    forecast_offset: int = 0,
    n_points: int | None = None,
    clip_negative_z: bool = True,
) -> dict[str, np.ndarray | int | float | str]:
    """Convert overlapping window errors into raw point-level anomaly scores.

    Parameters
    ----------
    predictions, targets:
        Arrays shaped ``[N, H, C]``.
    end_indices:
        The absolute first forecast index for each window.  Thus the forecast
        at horizon ``h`` maps to ``end_indices + forecast_offset + h``.  The
        project data pipeline uses the default ``forecast_offset=0``.
    validation_residuals:
        Signed errors from a normal-only validation split.  When supplied,
        per-channel median/MAD statistics are fit without looking at the test
        split.  Alternatively pass precomputed ``channel_center`` and
        ``channel_scale``.  If neither is supplied, raw residual magnitudes are
        used (center 0, scale 1); test statistics are never fitted implicitly.
    top_q:
        Fraction of channels to average (``0 < q <= 1``) or an integer number
        of channels.  The largest channel scores at each timestamp are used.
    memory_distance, divergence:
        Optional already-calibrated components broadcastable from ``[N]``,
        ``[N, C]``, ``[N, H]``, or ``[N, H, C]`` and added with their weights.
    agreement:
        Optional value in ``[0,1]`` derived from reference divergence.  It
        implements the paper's ``(1-s_div) * z_pred`` term by multiplying the
        residual score by ``1 + agreement_weight * agreement``.
    confidence:
        Optional values in ``[0, 1]``.  ``confidence_mode='uncertainty'``
        multiplies residuals by ``2-confidence``; ``'inverse'`` divides by
        clipped confidence; ``'gate'`` multiplies by confidence; ``'none'``
        leaves them unchanged.

    Returns
    -------
    dict
        ``point_scores`` has length ``n_points`` and contains the top-q channel
        mean. ``channel_scores`` and ``overlap_counts`` have shape ``[T, C]``.
        Prefix/suffix locations not covered by any forecast are NaN.  This is a
        plain overlap-add average and performs no point adjustment.
    """

    prediction = _as_numpy(predictions, dtype=np.float64)
    target = _as_numpy(targets, dtype=np.float64)
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("predictions and targets must have shape [N, H, C]")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    n_windows, horizon, channels = prediction.shape
    indices = _as_numpy(end_indices, dtype=np.int64).reshape(-1)
    if len(indices) != n_windows:
        raise ValueError("end_indices must contain one index per window")
    if np.any(indices + int(forecast_offset) < 0):
        raise ValueError("forecast timestamps must be non-negative")

    residual = _residual_magnitude(prediction - target, residual_kind)
    if validation_residuals is not None:
        if channel_center is not None or channel_scale is not None:
            raise ValueError(
                "pass validation_residuals or channel_center/channel_scale, not both"
            )
        stats = fit_robust_channel_stats(
            validation_residuals,
            residual_kind=residual_kind,
            residuals_are_errors=True,
        )
        center = np.asarray(stats["center"], dtype=np.float64)
        scale = np.asarray(stats["scale"], dtype=np.float64)
    elif channel_center is None and channel_scale is None:
        center = np.zeros(channels, dtype=np.float64)
        scale = np.ones(channels, dtype=np.float64)
    elif channel_center is None or channel_scale is None:
        raise ValueError("channel_center and channel_scale must be passed together")
    else:
        center = _channel_vector(channel_center, channels, "channel_center")
        scale = _channel_vector(channel_scale, channels, "channel_scale")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("channel_scale must be finite and strictly positive")

    channel_window_scores = (residual - center.reshape(1, 1, channels)) / (
        scale.reshape(1, 1, channels) + np.finfo(np.float64).eps
    )
    if clip_negative_z:
        channel_window_scores = np.maximum(channel_window_scores, 0.0)

    if confidence is not None:
        confidence_array = np.clip(
            _broadcast_window_component(
                confidence, n_windows, horizon, channels, "confidence"
            ),
            0.0,
            1.0,
        )
        mode = confidence_mode.strip().lower()
        if mode == "uncertainty":
            channel_window_scores *= 2.0 - confidence_array
        elif mode == "inverse":
            channel_window_scores /= np.clip(
                confidence_array, float(confidence_floor), 1.0
            )
        elif mode == "gate":
            channel_window_scores *= confidence_array
        elif mode != "none":
            raise ValueError(
                "confidence_mode must be none, uncertainty, inverse, or gate"
            )

    if agreement is not None and agreement_weight != 0:
        agreement_array = np.clip(
            _broadcast_window_component(
                agreement, n_windows, horizon, channels, "agreement"
            ),
            0.0,
            1.0,
        )
        channel_window_scores *= 1.0 + float(agreement_weight) * agreement_array

    if memory_distance is not None and memory_weight != 0:
        channel_window_scores += float(memory_weight) * _broadcast_window_component(
            memory_distance, n_windows, horizon, channels, "memory_distance"
        )
    if divergence is not None and divergence_weight != 0:
        channel_window_scores += float(divergence_weight) * _broadcast_window_component(
            divergence, n_windows, horizon, channels, "divergence"
        )

    inferred_points = (
        int(np.max(indices)) + int(forecast_offset) + horizon if n_windows else 0
    )
    if n_points is None:
        n_points = inferred_points
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    sums = np.zeros((n_points, channels), dtype=np.float64)
    counts = np.zeros((n_points, channels), dtype=np.int64)
    for horizon_index in range(horizon):
        timestamps = indices + int(forecast_offset) + horizon_index
        valid_time = (timestamps >= 0) & (timestamps < n_points)
        if not np.any(valid_time):
            continue
        values = channel_window_scores[valid_time, horizon_index, :]
        finite = np.isfinite(values)
        safe_values = np.where(finite, values, 0.0)
        np.add.at(sums, timestamps[valid_time], safe_values)
        np.add.at(counts, timestamps[valid_time], finite.astype(np.int64))

    channel_scores = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=channel_scores, where=counts > 0)
    point_scores = _top_q_channel_mean(channel_scores, top_q)
    covered = np.any(counts > 0, axis=1)

    return {
        "point_scores": point_scores,
        "channel_scores": channel_scores,
        "overlap_counts": counts,
        "covered_mask": covered,
        "top_q": float(top_q),
        "residual_kind": _canonical_residual_kind(residual_kind),
        "forecast_offset": int(forecast_offset),
    }


def evaluate_scores(labels: ArrayLike, scores: ArrayLike) -> dict[str, Any]:
    """Compute point-wise AUPRC/AUROC without point adjustment.

    Non-finite score/label pairs are omitted, which permits direct evaluation
    of overlap-add output containing an uncovered context prefix.  If the
    retained labels contain only one class, both metrics are returned as NaN
    with ``single_class=True`` instead of raising or reporting a misleading
    value.
    """

    label_array = _as_numpy(labels).reshape(-1)
    score_array = _as_numpy(scores, dtype=np.float64).reshape(-1)
    if len(label_array) != len(score_array):
        raise ValueError(
            f"labels and scores must have equal length, got {len(label_array)} and "
            f"{len(score_array)}"
        )
    finite = np.isfinite(label_array) & np.isfinite(score_array)
    y_true = (label_array[finite] > 0).astype(np.int8, copy=False)
    y_score = score_array[finite]
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    result: dict[str, Any] = {
        "n_total": int(len(label_array)),
        "n_evaluated": int(len(y_true)),
        "n_dropped_nonfinite": int(len(label_array) - len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / len(y_true) if len(y_true) else float("nan"),
        "point_adjustment": False,
        "single_class": positives == 0 or negatives == 0,
    }
    if len(y_true) == 0 or result["single_class"]:
        result.update({"auprc": float("nan"), "auroc": float("nan")})
        return result

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - sklearn is in the experiment env
        raise ImportError("evaluate_scores requires scikit-learn") from exc

    result["auprc"] = float(average_precision_score(y_true, y_score))
    result["auroc"] = float(roc_auc_score(y_true, y_score))
    return result


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return total/trainable/frozen parameter counts and estimated MiB."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
        "parameter_mib": float(parameter_bytes / (1024**2)),
        "parameter_and_buffer_mib": float(
            (parameter_bytes + buffer_bytes) / (1024**2)
        ),
    }


def save_json(data: Any, path: str | os.PathLike[str], *, indent: int = 2) -> Path:
    """Atomically save nested experiment metadata as UTF-8 JSON.

    NumPy scalars/arrays, tensors, devices, and :class:`~pathlib.Path` objects
    are converted to JSON-compatible values.  NaN/Inf values become ``null`` so
    files remain standards-compliant and can be consumed outside Python.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _json_safe(data)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            normalized,
            handle,
            ensure_ascii=False,
            indent=int(indent),
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def _coerce_dataset(
    arrays: NumpyWindowDataset | Mapping[str, ArrayLike] | Sequence[ArrayLike],
) -> NumpyWindowDataset:
    if isinstance(arrays, NumpyWindowDataset):
        return arrays
    if isinstance(arrays, Mapping):
        contexts = _mapping_value(arrays, "contexts", "context", "x", "inputs")
        futures = _mapping_value(
            arrays, "futures", "future", "y", "targets", "target"
        )
        if contexts is None or futures is None:
            raise KeyError("array mapping requires contexts/x and futures/y")
        return NumpyWindowDataset(
            contexts,
            futures,
            analog=_mapping_value(
                arrays,
                "analog",
                "analogs",
                "analog_future",
                "retrieval",
                "retrieved_future",
            ),
            distance=_mapping_value(arrays, "distance", "distances"),
            divergence=_mapping_value(arrays, "divergence", "disagreement"),
            end_indices=_mapping_value(
                arrays, "end_indices", "forecast_indices", "indices"
            ),
        )
    if isinstance(arrays, Sequence) and not isinstance(arrays, (str, bytes)):
        values = list(arrays)
        if len(values) < 2 or len(values) > 6:
            raise ValueError("array tuple must contain between 2 and 6 elements")
        values.extend([None] * (6 - len(values)))
        return NumpyWindowDataset(
            values[0],
            values[1],
            analog=values[2],
            distance=values[3],
            divergence=values[4],
            end_indices=values[5],
        )
    raise TypeError("arrays must be a NumpyWindowDataset, mapping, or tuple/list")


def _mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_numpy(value: ArrayLike, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward_model(
    model: nn.Module, batch: Mapping[str, Tensor], *, return_aux: bool
) -> tuple[Tensor, dict[str, Any]]:
    # Backbones accept only ``x``.  The condition wrapper in this clean-room
    # implementation uses ``analog_future``/``return_gate`` while the generic
    # contract uses ``analog``/``return_aux``.  Resolve names from the declared
    # signature instead of catching TypeError, because TypeError raised *inside*
    # a model should never be mistaken for an interface mismatch.
    parameters = inspect.signature(model.forward).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "analog" in parameters or accepts_kwargs:
        kwargs["analog"] = batch.get("analog")
    elif "analog_future" in parameters:
        kwargs["analog_future"] = batch.get("analog")
    for name in ("distance", "divergence"):
        if name in parameters or accepts_kwargs:
            kwargs[name] = batch.get(name)

    gate_contract = False
    if "return_aux" in parameters or accepts_kwargs:
        kwargs["return_aux"] = return_aux
    elif "return_gate" in parameters:
        kwargs["return_gate"] = return_aux
        gate_contract = True

    output = model(batch["x"], **kwargs)
    prediction, aux = _unpack_model_output(output)
    if gate_contract and return_aux and "aux" in aux and "gate" not in aux:
        aux["gate"] = aux.pop("aux")
    if not isinstance(prediction, Tensor):
        raise TypeError("model forecast must be a torch.Tensor")
    return prediction, aux


def _unpack_model_output(output: Any) -> tuple[Tensor, dict[str, Any]]:
    if isinstance(output, Tensor):
        return output, {}
    if isinstance(output, tuple):
        if not output:
            raise ValueError("model returned an empty tuple")
        aux = output[1] if len(output) > 1 else {}
        if aux is None:
            aux = {}
        elif not isinstance(aux, Mapping):
            aux = {"aux": aux}
        return output[0], dict(aux)
    if isinstance(output, Mapping):
        for key in ("prediction", "predictions", "forecast", "y_hat", "pred"):
            if key in output:
                aux = {name: value for name, value in output.items() if name != key}
                if "aux" in aux and isinstance(aux["aux"], Mapping):
                    nested = dict(aux.pop("aux"))
                    nested.update(aux)
                    aux = nested
                return output[key], aux
    raise TypeError("model must return a Tensor, (Tensor, aux), or prediction mapping")


def _masked_forecast_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> tuple[Tensor, int]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"model forecast shape {tuple(prediction.shape)} does not match target "
            f"shape {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("model produced a non-finite forecast")
    finite = torch.isfinite(target)
    valid_count = int(finite.sum().item())
    if valid_count == 0:
        raise RuntimeError("batch contains no finite prediction/target pairs")
    prediction_valid = prediction[finite]
    target_valid = target[finite]
    if loss_name == "mse":
        value = F.mse_loss(prediction_valid, target_valid)
    else:
        value = F.huber_loss(
            prediction_valid, target_valid, delta=float(huber_delta), reduction="mean"
        )
    return value, valid_count


def _elementwise_forecast_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> Tensor:
    """Return an FP32, unreduced loss used by the SRF teacher signals."""
    prediction = prediction.float()
    target = target.float()
    if loss_name == "mse":
        return torch.square(prediction - target)
    return F.huber_loss(
        prediction,
        target,
        delta=float(huber_delta),
        reduction="none",
    )


def _srf_auxiliary_losses(
    auxiliary: Mapping[str, Any],
    batch: Mapping[str, Tensor],
    *,
    loss_name: str,
    huber_delta: float,
    enabled: bool,
    quality_margin: float,
    quality_temperature: float,
    anchor: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute reference-consistency and quality-aware gate losses.

    The detached soft teachers are derived only from normal adaptation data.
    They answer two separate questions: whether at least one retrieved future is
    materially better than the bypass prediction, and whether the fully open
    SRF proposal is better than that bypass.  Held-out labels never enter this
    calculation.
    """
    zero = anchor.float() * 0.0
    if not enabled:
        return zero, zero

    required = {
        "bypass_forecast",
        "analogue",
        "ungated_forecast",
        "gamma",
        "valid_reference",
    }
    missing = sorted(required.difference(auxiliary))
    if missing:
        raise RuntimeError(
            "SRF auxiliary loss requested but model did not return: "
            + ", ".join(missing)
        )
    if "analog" not in batch or "distance" not in batch:
        raise RuntimeError("SRF auxiliary loss requires analog and distance inputs")

    target = batch["y"].float()
    bypass = torch.as_tensor(
        auxiliary["bypass_forecast"], device=target.device
    ).float()
    analogue = torch.as_tensor(auxiliary["analogue"], device=target.device).float()
    proposal = torch.as_tensor(
        auxiliary["ungated_forecast"], device=target.device
    ).float()
    gamma = torch.as_tensor(auxiliary["gamma"], device=target.device).float()
    valid_reference = torch.as_tensor(
        auxiliary["valid_reference"], device=target.device
    ).bool()
    if valid_reference.ndim == 3 and valid_reference.shape[1] == 1:
        valid_reference = valid_reference[:, 0, :]
    if valid_reference.ndim != 2:
        raise ValueError("valid_reference must have shape [B,C] or [B,1,C]")
    if gamma.ndim == 3:
        gamma = gamma.mean(dim=1)
    if gamma.ndim != 2:
        raise ValueError("gamma must have shape [B,C] or [B,H,C]")

    candidates = batch["analog"].float()
    if candidates.ndim == 3:
        candidates = candidates[:, None, :, :]
    if candidates.ndim != 4:
        raise ValueError("analog must have shape [B,K,H,C] or [B,H,C]")
    distance = batch["distance"].float()
    if distance.ndim == 2:
        distance = distance[:, None, :]
    if distance.ndim != 3:
        raise ValueError("distance must have shape [B,K,C] or [B,C]")

    finite_target = torch.isfinite(target).all(dim=1)
    sentinel_limit = 0.5 * torch.finfo(torch.float32).max
    valid_candidate = (
        torch.isfinite(distance)
        & (distance < sentinel_limit)
        & torch.isfinite(candidates).all(dim=2)
        & finite_target[:, None, :]
    )
    valid_reference = valid_reference & valid_candidate.any(dim=1) & finite_target

    safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    safe_candidates = torch.nan_to_num(
        candidates, nan=0.0, posinf=0.0, neginf=0.0
    )
    candidate_error = _elementwise_forecast_loss(
        safe_candidates,
        safe_target[:, None, :, :].expand_as(safe_candidates),
        loss_name=loss_name,
        huber_delta=huber_delta,
    ).mean(dim=2)
    candidate_error = candidate_error.masked_fill(~valid_candidate, torch.inf)
    best_reference_error = candidate_error.amin(dim=1)

    bypass_error = _elementwise_forecast_loss(
        bypass,
        safe_target,
        loss_name=loss_name,
        huber_delta=huber_delta,
    ).mean(dim=1)
    denominator = (bypass_error + best_reference_error).clamp_min(1.0e-6)
    reference_margin = torch.nan_to_num(
        (bypass_error - best_reference_error) / denominator,
        nan=-1.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    reference_quality = torch.sigmoid(
        (reference_margin - float(quality_margin)) / float(quality_temperature)
    )
    reference_quality = (reference_quality * valid_reference).detach()
    analogue_error = _elementwise_forecast_loss(
        analogue,
        safe_target,
        loss_name=loss_name,
        huber_delta=huber_delta,
    ).mean(dim=1)
    # Dividing by all valid references (rather than sum(q)) keeps a rare good
    # neighbour from receiving an accidentally enormous gradient.
    reference_denominator = valid_reference.float().sum().clamp_min(1.0)
    reference_loss = (reference_quality * analogue_error).sum() / reference_denominator

    proposal_error = _elementwise_forecast_loss(
        proposal,
        safe_target,
        loss_name=loss_name,
        huber_delta=huber_delta,
    ).mean(dim=1)
    proposal_denominator = (bypass_error + proposal_error).clamp_min(1.0e-6)
    proposal_margin = torch.nan_to_num(
        (bypass_error - proposal_error) / proposal_denominator,
        nan=-1.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    gate_target = torch.sigmoid(
        (proposal_margin - float(quality_margin)) / float(quality_temperature)
    )
    gate_target = (gate_target * valid_reference).detach()
    gate_mask = finite_target.float()
    gate_loss = (
        torch.square(gamma.clamp(0.0, 1.0) - gate_target) * gate_mask
    ).sum() / gate_mask.sum().clamp_min(1.0)

    if not bool(torch.isfinite(reference_loss)) or not bool(torch.isfinite(gate_loss)):
        raise RuntimeError("SRF auxiliary loss became non-finite")
    return reference_loss, gate_loss


def _tensor_to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _value_to_batch_numpy(value: Any, batch_size: int) -> np.ndarray | None:
    if isinstance(value, Tensor):
        array = _tensor_to_numpy(value)
    elif isinstance(value, np.ndarray):
        array = value
    else:
        return None
    if array.ndim == 0 or len(array) != batch_size:
        return None
    return array


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _canonical_residual_kind(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized in {"absolute", "abs", "l1"}:
        return "absolute"
    if normalized in {"squared", "square", "mse", "l2"}:
        return "squared"
    raise ValueError("residual_kind must be 'absolute' or 'squared'")


def _residual_magnitude(errors: np.ndarray, kind: str) -> np.ndarray:
    canonical = _canonical_residual_kind(kind)
    return np.abs(errors) if canonical == "absolute" else np.square(errors)


def _channel_vector(value: ArrayLike, channels: int, name: str) -> np.ndarray:
    vector = _as_numpy(value, dtype=np.float64).reshape(-1)
    if len(vector) != channels:
        raise ValueError(f"{name} must have C={channels} elements")
    return vector


def _broadcast_window_component(
    value: ArrayLike,
    n_windows: int,
    horizon: int,
    channels: int,
    name: str,
) -> np.ndarray:
    array = _as_numpy(value, dtype=np.float64)
    if array.ndim == 0:
        return np.broadcast_to(array, (n_windows, horizon, channels))
    if array.shape[0] != n_windows:
        raise ValueError(f"{name} first dimension must be N={n_windows}")
    if array.ndim == 1:
        array = array[:, None, None]
    elif array.ndim == 2:
        if array.shape[1] == channels:
            array = array[:, None, :]
        elif array.shape[1] == horizon:
            array = array[:, :, None]
        elif array.shape[1] == 1:
            array = array[:, :, None]
        else:
            raise ValueError(
                f"cannot interpret {name} shape {array.shape}; expected [N,C] or [N,H]"
            )
    elif array.ndim != 3:
        raise ValueError(f"{name} must have between 1 and 3 dimensions")
    try:
        return np.broadcast_to(array, (n_windows, horizon, channels))
    except ValueError as exc:
        raise ValueError(
            f"{name} shape {array.shape} is not broadcastable to "
            f"{(n_windows, horizon, channels)}"
        ) from exc


def _top_q_channel_mean(channel_scores: np.ndarray, top_q: float | int) -> np.ndarray:
    channels = channel_scores.shape[1]
    if isinstance(top_q, (int, np.integer)):
        k = int(top_q)
        if k < 1 or k > channels:
            raise ValueError(f"integer top_q must be between 1 and C={channels}")
    else:
        q = float(top_q)
        if not 0 < q <= 1:
            raise ValueError("fractional top_q must satisfy 0 < top_q <= 1")
        k = max(1, int(math.ceil(q * channels)))

    output = np.full(channel_scores.shape[0], np.nan, dtype=np.float64)
    for row_index, row in enumerate(channel_scores):
        finite = row[np.isfinite(row)]
        if finite.size:
            use_k = min(k, finite.size)
            partitioned = np.partition(finite, finite.size - use_k)
            output[row_index] = float(np.mean(partitioned[-use_k:]))
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "NumpyWindowDataset",
    "count_parameters",
    "evaluate_scores",
    "fit_robust_channel_stats",
    "overlap_add_scores",
    "predict_windows",
    "save_json",
    "seed_all",
    "train_model",
]
