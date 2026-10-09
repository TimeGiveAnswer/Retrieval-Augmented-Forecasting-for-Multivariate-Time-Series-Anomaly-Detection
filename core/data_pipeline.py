"""Clean-room data utilities for small time-series anomaly experiments.

This module deliberately contains no project-specific model code.  It provides
full-data readers, leakage-safe scaling/splitting, window construction, and
a bounded NumPy implementation of the per-channel analogue memory used by the
retrieval experiments.

Index convention
----------------
``make_windows`` returns ``end_indices`` equal to the *first forecast index*.
For a window starting at ``s``, the context is ``[s, s + L)`` and the future is
``[s + L, s + L + H)``.  Thus overlap-add code should place forecast step ``h``
at ``end_indices + h``.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from full_data import allocate_array, concatenate_arrays, finite_rows, load_nasa_entity, load_smd_text


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Dataset CSV loading requires pandas. Install it with `pip install pandas`."
        ) from exc
    return pd


def _normalise_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_csv(path: Path):
    """Read a CSV while tolerating common SWaT encodings and separators."""

    pd = _import_pandas()
    errors: List[Exception] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            if frame.shape[1] == 1:
                # Some redistributed SWaT files use semicolons.
                alternate = pd.read_csv(
                    path, encoding=encoding, sep=None, engine="python"
                )
                if alternate.shape[1] > 1:
                    frame = alternate
            return frame
        except Exception as exc:  # pragma: no cover - only malformed input
            errors.append(exc)
    detail = "; ".join(str(item) for item in errors[-2:])
    raise ValueError(f"Could not parse CSV {path}: {detail}")


def _candidate_prefixes(root: Path, dataset_names: Iterable[str]) -> List[Path]:
    prefixes: List[Path] = [root]
    container_names = ("data", "dataset", "datasets", "processed")
    for dataset_name in dataset_names:
        prefixes.append(root / dataset_name)
        for container in container_names:
            prefixes.append(root / container / dataset_name)
    # Keep order but remove duplicates without requiring paths to exist.
    result: List[Path] = []
    seen = set()
    for path in prefixes:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _find_csv(
    root: Path,
    dataset_names: Sequence[str],
    filenames: Sequence[str],
    role_tokens: Sequence[str],
    forbidden_tokens: Sequence[str] = (),
) -> Path:
    """Resolve a dataset CSV deterministically, preferring conventional layouts."""

    prefixes = _candidate_prefixes(root, dataset_names)
    for prefix in prefixes:
        for filename in filenames:
            candidate = prefix / filename
            if candidate.is_file():
                return candidate.resolve()

    candidates: List[Tuple[int, int, str, Path]] = []
    dataset_tokens = tuple(item.lower() for item in dataset_names)
    role_tokens = tuple(item.lower() for item in role_tokens)
    forbidden_tokens = tuple(item.lower() for item in forbidden_tokens)
    filename_set = {item.lower() for item in filenames}
    for pattern in {f"*{token}*.csv" for token in role_tokens} | set(filenames):
        for path in root.rglob(pattern):
            lower = str(path).lower().replace("\\", "/")
            if any(token in lower for token in forbidden_tokens):
                continue
            dataset_score = sum(token in lower for token in dataset_tokens)
            role_score = sum(token in path.name.lower() for token in role_tokens)
            exact_score = int(path.name.lower() in filename_set)
            # Negative values make higher scores sort first.
            candidates.append(
                (-exact_score - dataset_score - role_score, len(path.parts), lower, path)
            )
    if not candidates:
        expected = ", ".join(str(prefix / filenames[0]) for prefix in prefixes[:4])
        raise FileNotFoundError(f"No matching CSV found below {root}. Tried e.g. {expected}")
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3].resolve()


def _find_smd_file(root: Path, machine: str, role: str) -> Path:
    if role not in {"train", "test", "label"}:
        raise ValueError(f"Unknown SMD role: {role}")

    dirs = ("SMD", "smd", "ServerMachineDataset", "server_machine_dataset")
    prefixes = _candidate_prefixes(root, dirs)
    if role == "train":
        relative = (
            Path("train") / f"{machine}.npy",
            Path("processed") / f"{machine}_train.npy",
            Path(f"{machine}_train.npy"),
            Path(f"train_{machine}.npy"),
        )
    elif role == "test":
        relative = (
            Path("test") / f"{machine}.npy",
            Path("processed") / f"{machine}_test.npy",
            Path(f"{machine}_test.npy"),
            Path(f"test_{machine}.npy"),
        )
    else:
        relative = (
            Path("test_label") / f"{machine}.npy",
            Path("labels") / f"{machine}.npy",
            Path("processed") / f"{machine}_test_label.npy",
            Path(f"{machine}_test_label.npy"),
            Path(f"{machine}_labels.npy"),
            Path(f"{machine}_label.npy"),
        )
    for prefix in prefixes:
        for suffix in relative:
            candidate = prefix / suffix
            if candidate.is_file():
                return candidate.resolve()

    ranked: List[Tuple[int, int, str, Path]] = []
    for path in root.rglob(f"*{machine}*.npy"):
        lower = str(path).lower().replace("\\", "/")
        name = path.stem.lower()
        has_label = "label" in lower
        has_train = "train" in lower
        has_test = "test" in lower
        if role == "label":
            if not has_label:
                continue
            score = 4 * int("test_label" in lower) + 2 * int(name.endswith("label"))
        elif role == "train":
            if has_label or has_test:
                continue
            if not has_train:
                continue
            score = 3 * int(name.endswith("train")) + 2 * int("/train/" in lower)
        else:
            if has_label or has_train:
                continue
            if not has_test:
                continue
            score = 3 * int(name.endswith("test")) + 2 * int("/test/" in lower)
        score += int("smd" in lower or "servermachine" in lower)
        ranked.append((-score, len(path.parts), lower, path))
    if not ranked:
        raise FileNotFoundError(
            f"Could not find SMD {role} .npy for {machine} below {root}"
        )
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3].resolve()


def _numeric_frame(frame, columns: Sequence[Any]) -> np.ndarray:
    pd = _import_pandas()
    converted = []
    for column in columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            series = series.astype(str).str.strip().str.replace(",", "", regex=False)
        converted.append(pd.to_numeric(series, errors="coerce").to_numpy())
    if not converted:
        raise ValueError("No sensor columns remain after removing timestamp/label columns")
    return np.column_stack(converted).astype(np.float64, copy=False)


def _timestamp_columns(frame) -> List[Any]:
    result: List[Any] = []
    explicit = {
        "time",
        "timestamp",
        "timestampmin",
        "datetime",
        "date",
        "index",
        "unnamed0",
    }
    for column in frame.columns:
        normalised = _normalise_column_name(column)
        if normalised in explicit or normalised.startswith("unnamed"):
            result.append(column)
    return result


def _normalise_labels(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map common string/numeric normal/attack encodings to {0, 1}."""

    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError("Label vector is empty")

    pd = _import_pandas()
    text = pd.Series(array).astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)
    mapping: Dict[str, int] = {}
    if np.isfinite(numeric).all():
        unique = set(np.unique(numeric).tolist())
        if unique.issubset({0.0, 1.0}):
            labels = (numeric == 1.0).astype(np.int64)
            mapping = {"0": 0, "1": 1}
        elif unique.issubset({-1.0, 1.0}):
            # The common ICS convention is +1 normal, -1 attack.
            labels = (numeric == -1.0).astype(np.int64)
            mapping = {"1": 0, "-1": 1}
        elif unique.issubset({-1.0, 0.0}):
            labels = (numeric == -1.0).astype(np.int64)
            mapping = {"0": 0, "-1": 1}
        else:
            labels = (numeric != 0.0).astype(np.int64)
            mapping = {"0": 0, "nonzero": 1}
        return labels, mapping

    normal_tokens = {
        "normal",
        "benign",
        "false",
        "no",
        "n",
        "normaloperation",
        "noattack",
    }
    attack_tokens = {
        "attack",
        "attacked",
        "anomaly",
        "anomalous",
        "abnormal",
        "malicious",
        "true",
        "yes",
        "y",
    }
    labels = np.empty(array.size, dtype=np.int64)
    unknown = set()
    for index, raw in enumerate(text.tolist()):
        token = re.sub(r"[^a-z0-9]+", "", raw.lower())
        if token in normal_tokens:
            labels[index] = 0
            mapping[raw] = 0
        elif token in attack_tokens:
            labels[index] = 1
            mapping[raw] = 1
        else:
            unknown.add(raw)
    if unknown:
        preview = ", ".join(repr(item) for item in sorted(unknown)[:8])
        raise ValueError(f"Unrecognised label values: {preview}")
    return labels, mapping


def _coerce_label_frame(frame, force_drop_first_if_needed: bool = False):
    columns = list(frame.columns)
    drop = set(_timestamp_columns(frame))
    if force_drop_first_if_needed and columns:
        # Standard PSM label CSVs carry the timestamp as their first field.
        first = columns[0]
        if first in drop or len(columns) > 1:
            drop.add(first)
    label_columns = [item for item in columns if item not in drop]
    if not label_columns:
        raise ValueError("No label column found")
    if len(label_columns) == 1:
        return _normalise_labels(frame[label_columns[0]].to_numpy())

    combined = np.zeros(len(frame), dtype=np.int64)
    combined_mapping: Dict[str, int] = {}
    for column in label_columns:
        current, mapping = _normalise_labels(frame[column].to_numpy())
        combined |= current
        combined_mapping.update({f"{column}:{key}": value for key, value in mapping.items()})
    return combined, combined_mapping


def _load_psm(root: Path):
    names = ("PSM", "psm")
    train_path = _find_csv(
        root, names, ("train.csv", "PSM_train.csv"), ("train",), ("label", "test")
    )
    test_path = _find_csv(
        root, names, ("test.csv", "PSM_test.csv"), ("test",), ("label", "train")
    )
    label_path = _find_csv(
        root,
        names,
        ("test_label.csv", "labels.csv", "PSM_test_label.csv"),
        ("label",),
        ("train",),
    )
    train_frame = _read_csv(train_path)
    test_frame = _read_csv(test_path)
    label_frame = _read_csv(label_path)

    def feature_columns(frame):
        columns = list(frame.columns)
        drop = set(_timestamp_columns(frame))
        # PSM's first field is the timestamp even in distributions where its
        # header was renamed.
        if columns:
            drop.add(columns[0])
        return [column for column in columns if column not in drop]

    train_columns = feature_columns(train_frame)
    test_columns = feature_columns(test_frame)
    if len(train_columns) != len(test_columns):
        raise ValueError(
            f"PSM channel mismatch: train has {len(train_columns)}, test has {len(test_columns)}"
        )
    train = _numeric_frame(train_frame, train_columns)
    test = _numeric_frame(test_frame, test_columns)
    labels, mapping = _coerce_label_frame(label_frame, force_drop_first_if_needed=True)
    metadata = {
        "dataset": "PSM",
        "feature_names": [str(item) for item in train_columns],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_path": str(label_path),
        "label_mapping": mapping,
    }
    return train, test, labels, metadata


def _orient_time_major(array: np.ndarray, role: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{role} must be 1-D or 2-D, got shape {array.shape}")
    # SMD has thousands of time steps and tens of channels.  This catches
    # channel-major exports without changing short toy arrays.
    if array.shape[0] <= 256 and array.shape[1] >= 4 * array.shape[0]:
        array = array.T
    return array.astype(np.float64, copy=False)


def _load_smd(root: Path, machine: str):
    train_path = _find_smd_file(root, machine, "train")
    test_path = _find_smd_file(root, machine, "test")
    label_path = _find_smd_file(root, machine, "label")
    train = _orient_time_major(np.load(train_path, allow_pickle=False), "SMD train")
    test = _orient_time_major(np.load(test_path, allow_pickle=False), "SMD test")
    raw_labels = np.asarray(np.load(label_path, allow_pickle=False))
    if raw_labels.ndim > 1:
        if 1 in raw_labels.shape:
            raw_labels = raw_labels.reshape(-1)
        elif raw_labels.shape[0] == test.shape[0]:
            raw_labels = np.any(raw_labels != 0, axis=1).astype(np.int64)
        elif raw_labels.shape[1] == test.shape[0]:
            raw_labels = np.any(raw_labels != 0, axis=0).astype(np.int64)
        else:
            raise ValueError(f"Cannot align SMD labels of shape {raw_labels.shape}")
    labels, mapping = _normalise_labels(raw_labels)
    metadata = {
        "dataset": "SMD",
        "machine": machine,
        "feature_names": [f"channel_{index}" for index in range(train.shape[1])],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_path": str(label_path),
        "label_mapping": mapping,
    }
    return train, test, labels, metadata


def _find_nasa_file(root: Path, dataset: str, role: str) -> Path:
    """Resolve conventional MSL/SMAP NumPy files without external code."""

    if role not in {"train", "test", "label"}:
        raise ValueError(f"Unknown {dataset} role: {role}")
    canonical = dataset.upper()
    suffix = "test_label" if role == "label" else role
    filenames = (
        f"{canonical}_{suffix}.npy",
        f"{canonical.lower()}_{suffix}.npy",
        f"{suffix}.npy",
    )
    prefixes = _candidate_prefixes(root, (canonical, canonical.lower()))
    for prefix in prefixes:
        for filename in filenames:
            candidate = prefix / filename
            if candidate.is_file():
                return candidate.resolve()

    ranked: List[Tuple[int, int, str, Path]] = []
    for path in root.rglob("*.npy"):
        lower = str(path).lower().replace("\\", "/")
        if canonical.lower() not in lower:
            continue
        name = path.stem.lower()
        has_label = "label" in name
        has_train = "train" in name
        has_test = "test" in name
        if role == "label" and not (has_test and has_label):
            continue
        if role == "train" and (not has_train or has_label):
            continue
        if role == "test" and (not has_test or has_label):
            continue
        exact = int(path.name.lower() in {item.lower() for item in filenames})
        ranked.append((-exact, len(path.parts), lower, path))
    if not ranked:
        raise FileNotFoundError(f"Could not find {canonical} {role} .npy below {root}")
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3].resolve()


def _load_nasa(root: Path, dataset: str):
    canonical = dataset.upper()
    train_path = _find_nasa_file(root, canonical, "train")
    test_path = _find_nasa_file(root, canonical, "test")
    label_path = _find_nasa_file(root, canonical, "label")
    train = _orient_time_major(
        np.load(train_path, allow_pickle=False), f"{canonical} train"
    )
    test = _orient_time_major(
        np.load(test_path, allow_pickle=False), f"{canonical} test"
    )
    raw_labels = np.asarray(np.load(label_path, allow_pickle=False))
    if raw_labels.ndim > 1:
        if 1 in raw_labels.shape:
            raw_labels = raw_labels.reshape(-1)
        elif raw_labels.shape[0] == test.shape[0]:
            raw_labels = np.any(raw_labels != 0, axis=1).astype(np.int64)
        elif raw_labels.shape[1] == test.shape[0]:
            raw_labels = np.any(raw_labels != 0, axis=0).astype(np.int64)
        else:
            raise ValueError(
                f"Cannot align {canonical} labels of shape {raw_labels.shape}"
            )
    labels, mapping = _normalise_labels(raw_labels)
    metadata = {
        "dataset": canonical,
        "feature_names": [f"channel_{index}" for index in range(train.shape[1])],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_path": str(label_path),
        "label_mapping": mapping,
    }
    return train, test, labels, metadata


def _find_swat_label_column(frame):
    priority = (
        "normalattack",
        "attacklabel",
        "attack",
        "anomalylabel",
        "anomaly",
        "label",
        "class",
        "status",
    )
    normalised = {_normalise_column_name(column): column for column in frame.columns}
    for candidate in priority:
        if candidate in normalised:
            return normalised[candidate]
    for key, column in normalised.items():
        if "normalattack" in key or key.endswith("label"):
            return column
    raise ValueError(
        "SWaT test CSV must contain its own Normal/Attack (or equivalent label) column; "
        "a separate test_label.csv is intentionally not used"
    )


def _load_swat(root: Path):
    names = ("SWaT", "swat", "SWAT")
    train_path = _find_csv(
        root,
        names,
        (
            "train.csv",
            "SWaT_train.csv",
            "SWaT_Dataset_Normal_v1.csv",
            "SWaT_Dataset_Normal_v0.csv",
        ),
        ("train", "normal"),
        ("test_label",),
    )
    test_path = _find_csv(
        root,
        names,
        (
            "test.csv",
            "SWaT_test.csv",
            "SWaT_Dataset_Attack_v0.csv",
            "SWaT_Dataset_Attack_v1.csv",
        ),
        ("test", "attack"),
        ("test_label", "labels"),
    )
    train_frame = _read_csv(train_path)
    test_frame = _read_csv(test_path)
    label_column = _find_swat_label_column(test_frame)
    labels, mapping = _normalise_labels(test_frame[label_column].to_numpy())

    train_label_columns = set()
    try:
        train_label_columns.add(_find_swat_label_column(train_frame))
    except ValueError:
        pass
    train_drop = set(_timestamp_columns(train_frame)) | train_label_columns
    test_drop = set(_timestamp_columns(test_frame)) | {label_column}
    train_columns = [column for column in train_frame.columns if column not in train_drop]
    test_columns = [column for column in test_frame.columns if column not in test_drop]

    test_by_name = {_normalise_column_name(column): column for column in test_columns}
    aligned_train: List[Any] = []
    aligned_test: List[Any] = []
    for train_column in train_columns:
        key = _normalise_column_name(train_column)
        if key in test_by_name:
            aligned_train.append(train_column)
            aligned_test.append(test_by_name[key])
    if len(aligned_train) != len(train_columns) or len(aligned_test) != len(test_columns):
        if len(train_columns) != len(test_columns):
            raise ValueError(
                f"SWaT channel mismatch after removing time/label columns: "
                f"train={len(train_columns)}, test={len(test_columns)}"
            )
        # Some mirrors rename all channels positionally.  Positional fallback
        # is safe only when the channel counts agree.
        aligned_train, aligned_test = train_columns, test_columns

    train = _numeric_frame(train_frame, aligned_train)
    test = _numeric_frame(test_frame, aligned_test)
    metadata = {
        "dataset": "SWaT",
        "feature_names": [str(item) for item in aligned_train],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_path": None,
        "label_source": f"embedded column {label_column!s} in test CSV",
        "label_mapping": mapping,
    }
    return train, test, labels, metadata


def _contiguous_crop(
    array: np.ndarray,
    maximum: Optional[int],
    seed: int,
    salt: str,
    mode: str = "random",
) -> Tuple[np.ndarray, int]:
    if maximum is None or int(maximum) == 0 or maximum >= len(array):
        return array, 0
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("max_train_points/max_test_points must be positive or None")
    mode = str(mode).strip().lower()
    if mode == "head":
        start = 0
    elif mode == "tail":
        start = len(array) - maximum
    elif mode == "random":
        stable_salt = zlib.crc32(salt.encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.default_rng((int(seed) ^ stable_salt) & 0xFFFFFFFF)
        start = int(rng.integers(0, len(array) - maximum + 1))
    else:
        raise ValueError("crop mode must be random, head, or tail")
    return array[start : start + maximum], start


def _causal_fill(array: np.ndarray, train_median: np.ndarray) -> np.ndarray:
    """Forward-fill each channel, then fill only leading gaps from train medians."""

    result = np.asarray(array, dtype=np.float64).copy()
    result[~np.isfinite(result)] = np.nan
    for channel in range(result.shape[1]):
        column = result[:, channel]
        valid = np.flatnonzero(np.isfinite(column))
        if valid.size:
            # maximum.accumulate supplies the most recent prior valid index;
            # positions before the first valid observation remain -1.
            previous = np.where(np.isfinite(column), np.arange(len(column)), -1)
            previous = np.maximum.accumulate(previous)
            can_fill = previous >= 0
            column[can_fill] = column[previous[can_fill]]
        column[~np.isfinite(column)] = train_median[channel]
        result[:, channel] = column
    return result


def load_series(
    name: str,
    root: Union[str, Path],
    max_train_points: Optional[int] = None,
    max_test_points: Optional[int] = None,
    seed: int = 0,
    crop_mode: str = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a clean-room PSM, MSL, SMAP, SMD, or SWaT time series.

    Parameters
    ----------
    name:
        ``PSM``, ``SWaT``, complete processed ``MSL``/``SMAP`` arrays, or an
        entity-qualified name such as ``SMD-machine-2-1`` / ``MSL@P-1``.
        Use full_data.discover_datasets to expand whole multi-entity families.
    root:
        Dataset root or an ancestor containing conventional ``PSM``, ``SMD``
        and ``SWaT`` directories.
    max_train_points, max_test_points:
        Optional pilot limits.  A single deterministic contiguous segment is
        selected, preserving temporal adjacency.  The offsets are recorded in
        metadata.  No label-guided test selection is performed.
    seed:
        Controls only the deterministic contiguous pilot crop.

    Returns
    -------
    train, test, labels, metadata:
        ``train`` and ``test`` are finite ``float32 [T, C]`` arrays; ``labels``
        is an ``int64 [T_test]`` vector where 1 means anomalous.  Missing values
        are causally forward-filled and any leading gaps use medians computed
        from the selected training series only.  In particular, SWaT labels
        always come from the test CSV's own Normal/Attack field; a separate
        ``test_label.csv`` is never trusted.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    key = re.sub(r"[_\s]+", "-", str(name).strip().lower())
    if key == "smd":
        raise ValueError("Expand SMD with full_data.discover_datasets and load each machine explicitly")
    if "@" in str(name) and str(name).split("@", 1)[0].upper() in {"MSL", "SMAP"}:
        train, test, labels, metadata = load_nasa_entity(root_path, str(name))
    elif key == "psm" or key.startswith("psm-"):
        train, test, labels, metadata = _load_psm(root_path)
    elif key == "msl" or key.startswith("msl-"):
        train, test, labels, metadata = _load_nasa(root_path, "MSL")
    elif key == "smap" or key.startswith("smap-"):
        train, test, labels, metadata = _load_nasa(root_path, "SMAP")
    elif key == "swat" or key.startswith("swat-"):
        train, test, labels, metadata = _load_swat(root_path)
    elif key == "smd" or key.startswith("smd-") or key.startswith("machine-"):
        match = re.search(r"machine-\d+-\d+", key)
        machine = match.group(0) if match else "machine-1-1"
        try:
            train, test, labels, metadata = _load_smd(root_path, machine)
        except FileNotFoundError:
            train, test, labels, metadata = load_smd_text(root_path, machine)
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}; expected PSM, MSL, SMAP, SWaT, "
            "SMD, or SMD-machine-X-Y"
        )

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"Expected [T,C] arrays, got train={train.shape}, test={test.shape}")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Channel mismatch: train has {train.shape[1]}, test has {test.shape[1]}"
        )
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) != len(test):
        raise ValueError(f"Test/label length mismatch: {len(test)} versus {len(labels)}")

    original_train_points = len(train)
    original_test_points = len(test)
    train, train_offset = _contiguous_crop(
        train,
        max_train_points,
        seed,
        f"{metadata['dataset']}-train-{metadata.get('machine', '')}",
        crop_mode,
    )
    test, test_offset = _contiguous_crop(
        test,
        max_test_points,
        seed,
        f"{metadata['dataset']}-test-{metadata.get('machine', '')}",
        crop_mode,
    )
    labels = labels[test_offset : test_offset + len(test)]

    raw_train = np.asarray(train, dtype=np.float64)
    raw_test = np.asarray(test, dtype=np.float64)
    raw_train[~np.isfinite(raw_train)] = np.nan
    raw_test[~np.isfinite(raw_test)] = np.nan
    missing_train = int(np.isnan(raw_train).sum())
    missing_test = int(np.isnan(raw_test).sum())
    train_median = np.nanmedian(raw_train, axis=0)
    bad_channels = np.flatnonzero(~np.isfinite(train_median))
    if bad_channels.size:
        raise ValueError(
            "Training data has channels with no finite values: "
            + ", ".join(map(str, bad_channels.tolist()))
        )
    train = _causal_fill(raw_train, train_median).astype(np.float32)
    test = _causal_fill(raw_test, train_median).astype(np.float32)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("Non-finite values remain after causal imputation")

    metadata = dict(metadata)
    metadata.update(
        {
            "original_train_points": int(original_train_points),
            "original_test_points": int(original_test_points),
            "train_crop_offset": int(train_offset),
            "test_crop_offset": int(test_offset),
            "train_points": int(len(train)),
            "test_points": int(len(test)),
            "channels": int(train.shape[1]),
            "missing_train_values": missing_train,
            "missing_test_values": missing_test,
            "imputation": "causal forward fill, then selected-training median for leading gaps",
            "anomaly_points": int(labels.sum()),
            "anomaly_rate": float(labels.mean()) if labels.size else 0.0,
            "seed": int(seed),
            "crop_mode": str(crop_mode),
        }
    )
    return train, test, labels, metadata


class RobustScaler:
    """Median/IQR scaler that can only be fitted explicitly on training data.

    The class mirrors the small subset of scikit-learn's scaler API needed by
    the experiments while keeping the dependency optional.  Constant channels
    receive scale 1.0.  Inputs must be finite to prevent silent propagation of
    bad values into the memory bank.
    """

    def __init__(self, eps: float = 1e-6):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, train: ArrayLike) -> "RobustScaler":
        values = np.asarray(train, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"train must have shape [T,C], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("RobustScaler.fit received non-finite training values")
        q25, q50, q75 = np.percentile(values, (25.0, 50.0, 75.0), axis=0)
        scale = q75 - q25
        scale[scale < self.eps] = 1.0
        self.center_ = q50.astype(np.float64)
        self.scale_ = scale.astype(np.float64)
        self.n_features_in_ = int(values.shape[1])
        return self

    def _check(self, values: ArrayLike) -> np.ndarray:
        if self.center_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("RobustScaler must be fitted before transform")
        array = np.asarray(values)
        if array.ndim < 2 or array.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected last dimension {self.n_features_in_}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("RobustScaler received non-finite values")
        return array.astype(np.float64, copy=False)

    def transform(self, values: ArrayLike) -> np.ndarray:
        array = self._check(values)
        return ((array - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, values: ArrayLike) -> np.ndarray:
        array = self._check(values)
        return (array * self.scale_ + self.center_).astype(np.float32)

    def fit_transform(self, train: ArrayLike) -> np.ndarray:
        return self.fit(train).transform(train)


def chronological_split(
    arr: np.ndarray,
    base_fraction: float = 0.60,
    adapt_fraction: float = 0.20,
    val_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an array into contiguous ``base``, ``adapt`` and ``val`` blocks.

    ``base`` is intended for native-model fitting, ``adapt`` for retrieval/LoRA
    adaptation, and ``val`` for calibration.  Fractions never shuffle data.  If
    ``val_fraction`` is omitted it receives the remainder.  Supplying all three
    fractions allows a sum below 1; any unassigned tail is intentionally not
    returned, which can be useful as an internal holdout.
    """

    values = np.asarray(arr)
    if values.ndim == 0:
        raise ValueError("arr must have a time axis")
    n = len(values)
    if n < 3:
        raise ValueError("At least three time points are needed for a 3-way split")
    if val_fraction is None:
        val_fraction = 1.0 - float(base_fraction) - float(adapt_fraction)
    fractions = np.asarray(
        [base_fraction, adapt_fraction, val_fraction], dtype=np.float64
    )
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError("All split fractions must be finite and positive")
    if fractions.sum() > 1.0 + 1e-12:
        raise ValueError("Split fractions cannot sum above 1")
    counts = np.floor(n * fractions).astype(int)
    if fractions.sum() >= 1.0 - 1e-12:
        counts[2] = n - counts[0] - counts[1]
    if np.any(counts <= 0):
        raise ValueError(f"Split produces an empty block: counts={counts.tolist()}")
    first = int(counts[0])
    second = first + int(counts[1])
    third = second + int(counts[2])
    return values[:first], values[first:second], values[second:third]


def _uniform_subsample_indices(count: int, wanted: int, seed: int) -> np.ndarray:
    """One deterministic sample per equal-width temporal bin, returned sorted."""

    if wanted >= count:
        return np.arange(count, dtype=np.int64)
    edges = np.linspace(0, count, wanted + 1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = np.empty(wanted, dtype=np.int64)
    for index in range(wanted):
        low, high = int(edges[index]), int(edges[index + 1])
        if high <= low:
            chosen[index] = low
        else:
            chosen[index] = int(rng.integers(low, high))
    chosen.sort()
    return chosen


def make_windows(arr, L, H, stride=1, max_windows=None, seed=0):
    """Every legal chronological window, using views/temporary arrays.

    Zero or None means unlimited. Only an explicitly positive cap subsamples.
    end_indices is the first forecast timestamp.
    """
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("arr must have shape [T,C]")
    L, H, stride = int(L), int(H), int(stride)
    if min(L, H, stride) <= 0 or len(values) < L + H:
        raise ValueError("Invalid L/H/stride or series shorter than L+H")
    last_start = len(values) - L - H
    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
    count, channels = len(starts), values.shape[1]
    common = (stride * values.strides[0], values.strides[0], values.strides[1])
    contexts = np.lib.stride_tricks.as_strided(values, (count, L, channels), common, writeable=False)
    futures = np.lib.stride_tricks.as_strided(values[L:], (count, H, channels), common, writeable=False)
    if starts[-1] != last_start:
        contexts = concatenate_arrays([contexts, values[last_start:last_start + L][None]])
        futures = concatenate_arrays([futures, values[last_start + L:last_start + L + H][None]])
        starts = np.append(starts, last_start)
    if max_windows is not None and int(max_windows) > 0 and int(max_windows) < len(starts):
        selection = _uniform_subsample_indices(len(starts), int(max_windows), seed)
        contexts, futures, starts = contexts[selection], futures[selection], starts[selection]
    return contexts, futures, starts + L


def _context_features(contexts, waveform_points, eps):
    """All features, chunked without copying the full N x L x C window array."""
    result = allocate_array((len(contexts), contexts.shape[2], 5 + waveform_points), np.float64)
    for start in range(0, len(contexts), 256):
        chunk = np.asarray(contexts[start:start + 256], dtype=np.float64)
        result[start:start + len(chunk)] = _context_features_chunk(chunk, waveform_points, eps)
    return result


def _context_features_chunk(contexts: np.ndarray, waveform_points: int, eps: float) -> np.ndarray:
    """Return [N,C,5+waveform_points] joint statistics/waveform features."""

    n, length, channels = contexts.shape
    mean = np.mean(contexts, axis=1)
    std = np.std(contexts, axis=1)
    last = contexts[:, -1, :]
    time = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    denominator = float(np.sum(time * time))
    if denominator <= eps:
        slope = np.zeros((n, channels), dtype=np.float64)
    else:
        slope = np.einsum("nlc,l->nc", contexts, time) / denominator
    if length > 1:
        differences = np.diff(contexts, axis=1)
        diff_energy = np.mean(differences * differences, axis=1)
    else:
        diff_energy = np.zeros((n, channels), dtype=np.float64)

    positions = np.linspace(0.0, max(length - 1, 0), waveform_points)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = (positions - lower)[None, :, None]
    sampled = contexts[:, lower, :] * (1.0 - weight) + contexts[:, upper, :] * weight
    z_wave = (sampled - mean[:, None, :]) / np.maximum(std[:, None, :], eps)
    statistics = np.stack((mean, std, last, slope, diff_energy), axis=-1)
    return np.concatenate((statistics, z_wave.transpose(0, 2, 1)), axis=-1)


def _robust_feature_scale(features: np.ndarray, eps: float):
    q25, center, q75 = np.percentile(features, (25.0, 50.0, 75.0), axis=0)
    scale = q75 - q25
    scale[scale < eps] = 1.0
    return center, scale


def _uniform_positions(count: int, wanted: int) -> np.ndarray:
    if wanted >= count:
        return np.arange(count, dtype=np.int64)
    # Bin centres avoid duplicates and cover the full time range.
    return np.floor((np.arange(wanted) + 0.5) * count / wanted).astype(np.int64)


def _farthest_positions(
    features: np.ndarray, wanted: int, pool_factor: int
) -> np.ndarray:
    count = len(features)
    if wanted >= count:
        return np.arange(count, dtype=np.int64)
    pool_size = min(count, max(wanted, wanted * pool_factor))
    pool_index = _uniform_positions(count, pool_size)
    pool = features[pool_index].astype(np.float64, copy=False)
    chosen = np.empty(wanted, dtype=np.int64)
    # Start at the point nearest the robust feature centre (zero after scaling).
    chosen[0] = int(np.argmin(np.einsum("nd,nd->n", pool, pool)))
    delta = pool - pool[chosen[0]]
    min_distance = np.einsum("nd,nd->n", delta, delta)
    min_distance[chosen[0]] = -1.0
    for position in range(1, wanted):
        next_index = int(np.argmax(min_distance))
        chosen[position] = next_index
        delta = pool - pool[next_index]
        current = np.einsum("nd,nd->n", delta, delta)
        min_distance = np.minimum(min_distance, current)
        min_distance[chosen[: position + 1]] = -1.0
    result = pool_index[chosen]
    result.sort()
    return result


class ChannelMemoryBank:
    """Bounded per-channel analogue memory with chunked exact top-k queries.

    Each context/channel is represented by mean, standard deviation, last
    value, linear slope, difference energy, and a 16-point (configurable)
    z-normalised waveform.  Features are robustly scaled per channel using only
    eligible normal training windows.  Stored futures are converted to changes
    relative to the candidate context's last value, then re-anchored to each
    query's last value (future level alignment).

    Parameters
    ----------
    memory_per_channel:
        Maximum normal exemplars retained independently for each variable.
    top_k, temperature:
        Number of neighbours and softmax temperature used by ``query``.
    selection:
        ``"uniform"`` for temporal coverage or ``"farthest"`` for a bounded
        greedy farthest-point coreset.  The latter first uses a uniform pool of
        at most ``pool_factor * memory_per_channel`` to avoid quadratic memory.
    query_chunk_size, memory_chunk_size:
        Bounds for distance matrix allocation; the full ``N x memory`` matrix
        is never materialised.
    """

    def __init__(
        self,
        memory_per_channel: int = 256,
        top_k: int = 5,
        temperature: float = 0.20,
        selection: str = "uniform",
        waveform_points: int = 16,
        query_chunk_size: int = 256,
        memory_chunk_size: int = 1024,
        pool_factor: int = 4,
        eps: float = 1e-6,
    ):
        integers = {
            "memory_per_channel": memory_per_channel,
            "top_k": top_k,
            "waveform_points": waveform_points,
            "query_chunk_size": query_chunk_size,
            "memory_chunk_size": memory_chunk_size,
            "pool_factor": pool_factor,
        }
        if any(int(value) <= 0 for value in integers.values()):
            raise ValueError("All memory/count/chunk parameters must be positive")
        if temperature <= 0 or eps <= 0:
            raise ValueError("temperature and eps must be positive")
        if selection not in {"uniform", "farthest", "boundary"}:
            raise ValueError("selection must be 'uniform', 'farthest', or 'boundary'")
        self.memory_per_channel = int(memory_per_channel)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.selection = selection
        self.waveform_points = int(waveform_points)
        self.query_chunk_size = int(query_chunk_size)
        self.memory_chunk_size = int(memory_chunk_size)
        self.pool_factor = int(pool_factor)
        self.eps = float(eps)

        self.features_: List[np.ndarray] = []
        self.future_deltas_: List[np.ndarray] = []
        self.source_end_indices_: List[np.ndarray] = []
        self.feature_center_: List[np.ndarray] = []
        self.feature_scale_: List[np.ndarray] = []
        self.size_per_channel_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None
        self.context_length_: Optional[int] = None
        self.horizon_: Optional[int] = None

    def fit(
        self,
        contexts: np.ndarray,
        futures: np.ndarray,
        normal_mask: Optional[np.ndarray] = None,
        end_indices: Optional[np.ndarray] = None,
    ) -> "ChannelMemoryBank":
        """Build memories from normal windows only.

        ``normal_mask`` may be ``[N]`` (whole-window normality) or ``[N,C]``
        (channel-specific normality).  Non-finite context/future pairs are
        excluded independently per channel.  ``end_indices`` is optional but,
        when supplied, enables temporal-neighbour exclusion in ``query``.
        """

        x = np.asarray(contexts)
        y = np.asarray(futures)
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"contexts/futures must be 3-D, got {x.shape}, {y.shape}")
        if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
            raise ValueError(f"Incompatible contexts/futures: {x.shape} versus {y.shape}")
        n, length, channels = x.shape
        horizon = y.shape[1]
        if n == 0:
            raise ValueError("Cannot fit an empty memory bank")
        if normal_mask is None:
            mask = np.ones((n, channels), dtype=bool)
        else:
            mask = np.asarray(normal_mask, dtype=bool)
            if mask.shape == (n,):
                mask = np.broadcast_to(mask[:, None], (n, channels))
            elif mask.shape != (n, channels):
                raise ValueError(
                    f"normal_mask must have shape {(n,)} or {(n, channels)}, got {mask.shape}"
                )
        if end_indices is None:
            source_indices = np.arange(n, dtype=np.int64)
        else:
            source_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)
            if source_indices.shape != (n,):
                raise ValueError(f"end_indices must have shape {(n,)}, got {source_indices.shape}")

        raw_features = _context_features(x, self.waveform_points, self.eps)
        self.features_ = []
        self.future_deltas_ = []
        self.source_end_indices_ = []
        self.feature_center_ = []
        self.feature_scale_ = []
        sizes = np.zeros(channels, dtype=np.int64)
        empty_channels: List[int] = []
        for channel in range(channels):
            finite = (
                finite_rows(x[:, :, channel])
                & finite_rows(y[:, :, channel])
                & np.isfinite(raw_features[:, channel, :]).all(axis=1)
            )
            eligible = np.flatnonzero(mask[:, channel] & finite)
            if eligible.size == 0:
                empty_channels.append(channel)
                self.features_.append(np.empty((0, raw_features.shape[-1]), dtype=np.float32))
                self.future_deltas_.append(np.empty((0, horizon), dtype=np.float32))
                self.source_end_indices_.append(np.empty(0, dtype=np.int64))
                self.feature_center_.append(np.zeros(raw_features.shape[-1], dtype=np.float32))
                self.feature_scale_.append(np.ones(raw_features.shape[-1], dtype=np.float32))
                continue
            channel_features = raw_features[eligible, channel, :]
            center, scale = _robust_feature_scale(channel_features, self.eps)
            scaled = (channel_features - center) / scale
            wanted = min(self.memory_per_channel, len(eligible))
            if self.selection == "uniform":
                selected_local = _uniform_positions(len(eligible), wanted)
            elif self.selection == "farthest":
                selected_local = _farthest_positions(scaled, wanted, self.pool_factor)
            else:
                # Boundary-aware memory: retain a diverse 80% coreset and use
                # the remaining 20% for normal windows at the edge of the
                # robust feature cloud.  The split is deterministic and uses
                # only eligible normal-training windows.
                boundary_wanted = min(
                    max(1, int(round(0.20 * wanted))), max(0, len(eligible) - 1)
                )
                core_wanted = wanted - boundary_wanted
                core = _farthest_positions(
                    scaled, max(core_wanted, 1), self.pool_factor
                )
                available = np.ones(len(eligible), dtype=bool)
                available[core] = False
                boundary_pool = np.flatnonzero(available)
                if boundary_wanted and boundary_pool.size:
                    # Robustly scaled radial distance is a simple, auditable
                    # definition of the normal support boundary.
                    radius = np.einsum("nd,nd->n", scaled, scaled)
                    order = np.argsort(radius[boundary_pool], kind="stable")
                    boundary = boundary_pool[order[-boundary_wanted:]]
                    selected_local = np.concatenate((core, boundary))
                else:
                    selected_local = core
                # Very small memories can make rounding leave a slot. Fill it
                # with the next deterministic temporal position.
                if len(selected_local) < wanted:
                    remaining = np.flatnonzero(
                        ~np.isin(np.arange(len(eligible)), selected_local)
                    )
                    selected_local = np.concatenate(
                        (selected_local, remaining[: wanted - len(selected_local)])
                    )
                selected_local = np.sort(selected_local[:wanted])
            selected = eligible[selected_local]
            self.features_.append(
                np.ascontiguousarray(scaled[selected_local], dtype=np.float32)
            )
            deltas = y[selected, :, channel].astype(np.float64) - x[selected, -1, channel, None].astype(np.float64)
            self.future_deltas_.append(np.ascontiguousarray(deltas, dtype=np.float32))
            self.source_end_indices_.append(source_indices[selected].copy())
            self.feature_center_.append(center.astype(np.float32))
            self.feature_scale_.append(scale.astype(np.float32))
            sizes[channel] = wanted
        if empty_channels:
            raise ValueError(
                "No finite normal windows for channel(s): "
                + ", ".join(map(str, empty_channels))
            )
        self.size_per_channel_ = sizes
        self.n_channels_ = channels
        self.context_length_ = length
        self.horizon_ = horizon
        return self

    def _ensure_fitted(self) -> None:
        if (
            self.n_channels_ is None
            or self.context_length_ is None
            or self.horizon_ is None
            or self.size_per_channel_ is None
        ):
            raise RuntimeError("ChannelMemoryBank must be fitted before query")

    def _topk(
        self,
        queries: np.ndarray,
        channel: int,
        k: int,
        query_indices: Optional[np.ndarray],
        exclusion_radius: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        memory = self.features_[channel]
        source = self.source_end_indices_[channel]
        batch = len(queries)
        best_distance = np.full((batch, k), np.inf, dtype=np.float64)
        best_index = np.full((batch, k), -1, dtype=np.int64)
        query_norm = np.einsum("nd,nd->n", queries, queries)[:, None]
        for start in range(0, len(memory), self.memory_chunk_size):
            stop = min(start + self.memory_chunk_size, len(memory))
            current_memory = memory[start:stop].astype(np.float64, copy=False)
            memory_norm = np.einsum("nd,nd->n", current_memory, current_memory)[None, :]
            distance2 = query_norm + memory_norm - 2.0 * queries @ current_memory.T
            np.maximum(distance2, 0.0, out=distance2)
            distance = np.sqrt(distance2 / max(queries.shape[1], 1))
            if query_indices is not None:
                too_close = (
                    np.abs(query_indices[:, None] - source[None, start:stop])
                    <= exclusion_radius
                )
                distance[too_close] = np.inf
            current_index = np.broadcast_to(
                np.arange(start, stop, dtype=np.int64)[None, :], distance.shape
            )
            merged_distance = np.concatenate((best_distance, distance), axis=1)
            merged_index = np.concatenate((best_index, current_index), axis=1)
            positions = np.argpartition(merged_distance, kth=k - 1, axis=1)[:, :k]
            best_distance = np.take_along_axis(merged_distance, positions, axis=1)
            best_index = np.take_along_axis(merged_index, positions, axis=1)
        order = np.argsort(best_distance, axis=1)
        return (
            np.take_along_axis(best_distance, order, axis=1),
            np.take_along_axis(best_index, order, axis=1),
        )

    def query(
        self,
        contexts: np.ndarray,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        query_end_indices: Optional[np.ndarray] = None,
        exclusion_radius: int = 0,
        return_candidates: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve and level-align analogue futures for a context batch.

        Parameters
        ----------
        contexts:
            Finite ``[N,L,C]`` query contexts.
        top_k, temperature:
            Optional per-call overrides.  Neighbours receive weights
            ``softmax(-(distance - nearest_distance) / temperature)``.
        query_end_indices, exclusion_radius:
            If query indices are supplied, memory entries whose stored index is
            within the inclusive radius are excluded.  This prevents exact or
            temporally-overlapping self retrieval when training queries and the
            memory originate from the same series.

        Returns
        -------
        analog_future, distance, divergence:
            By default arrays have shapes ``[N,H,C]``, ``[N,C]`` and
            ``[N,H,C]``.  With ``return_candidates=True`` the first two arrays
            retain the sorted neighbours and instead have shapes
            ``[N,K,H,C]`` and ``[N,K,C]``.  Keeping candidates separate lets
            SRF learn a selector rather than averaging references before the
            model sees them.
            Distance is the soft-weighted feature-space Euclidean distance;
            divergence is the weighted per-step standard deviation among the
            aligned neighbour futures.  Queries containing a non-finite value
            in a channel are not searched: their analogue is a flat continuation,
            distance is the largest finite float32 value, and divergence is 0.
        """

        self._ensure_fitted()
        x = np.asarray(contexts)
        if x.ndim != 3:
            raise ValueError(f"contexts must have shape [N,L,C], got {x.shape}")
        if x.shape[1] != self.context_length_ or x.shape[2] != self.n_channels_:
            raise ValueError(
                f"Expected [N,{self.context_length_},{self.n_channels_}], got {x.shape}"
            )
        k_requested = self.top_k if top_k is None else int(top_k)
        temp = self.temperature if temperature is None else float(temperature)
        if k_requested <= 0 or temp <= 0:
            raise ValueError("top_k and temperature must be positive")
        exclusion_radius = int(exclusion_radius)
        if exclusion_radius < 0:
            raise ValueError("exclusion_radius cannot be negative")
        if query_end_indices is None:
            query_indices = None
        else:
            query_indices = np.asarray(query_end_indices, dtype=np.int64).reshape(-1)
            if query_indices.shape != (len(x),):
                raise ValueError(
                    f"query_end_indices must have shape {(len(x),)}, got {query_indices.shape}"
                )

        n = len(x)
        horizon = int(self.horizon_)
        channels = int(self.n_channels_)
        analog = np.empty((n, horizon, channels), dtype=np.float32)
        divergence = np.zeros((n, horizon, channels), dtype=np.float32)
        distance_output = np.full(
            (n, channels), np.finfo(np.float32).max, dtype=np.float32
        )
        candidate_analog = None
        candidate_distance = None
        if return_candidates:
            candidate_analog = np.empty(
                (n, k_requested, horizon, channels), dtype=np.float32
            )
            candidate_distance = np.full(
                (n, k_requested, channels),
                np.finfo(np.float32).max,
                dtype=np.float32,
            )
        raw_features = _context_features(x, self.waveform_points, self.eps)
        for channel in range(channels):
            query_last = x[:, -1, channel]
            fallback_last = np.where(np.isfinite(query_last), query_last, 0.0)
            analog[:, :, channel] = fallback_last[:, None].astype(np.float32)
            if candidate_analog is not None:
                candidate_analog[:, :, :, channel] = fallback_last[:, None, None].astype(
                    np.float32
                )
            valid = (
                finite_rows(x[:, :, channel])
                & np.isfinite(raw_features[:, channel, :]).all(axis=1)
            )
            valid_positions = np.flatnonzero(valid)
            if not valid_positions.size:
                continue
            center = self.feature_center_[channel].astype(np.float64, copy=False)
            scale = self.feature_scale_[channel].astype(np.float64, copy=False)
            channel_features = (raw_features[valid_positions, channel, :] - center) / scale
            k = min(k_requested, len(self.features_[channel]))
            for offset in range(0, len(valid_positions), self.query_chunk_size):
                selected_positions = valid_positions[offset : offset + self.query_chunk_size]
                feature_batch = channel_features[offset : offset + self.query_chunk_size]
                index_batch = (
                    None if query_indices is None else query_indices[selected_positions]
                )
                distances, indices = self._topk(
                    feature_batch, channel, k, index_batch, exclusion_radius
                )
                has_neighbour = np.isfinite(distances).any(axis=1)
                safe_indices = np.maximum(indices, 0)
                deltas = self.future_deltas_[channel][safe_indices]
                weights = np.zeros_like(distances)
                if np.any(has_neighbour):
                    rows = np.flatnonzero(has_neighbour)
                    row_distances = distances[rows]
                    shifted = row_distances - np.min(
                        row_distances, axis=1, keepdims=True
                    )
                    row_logits = -shifted / temp
                    row_logits[~np.isfinite(row_distances)] = -np.inf
                    row_logits -= np.max(row_logits, axis=1, keepdims=True)
                    row_weights = np.exp(row_logits)
                    row_weights /= np.sum(row_weights, axis=1, keepdims=True)
                    weights[rows] = row_weights
                    aligned_candidates = (
                        deltas[rows]
                        + query_last[selected_positions[rows], None, None]
                    ).astype(np.float32)
                    if candidate_analog is not None and candidate_distance is not None:
                        candidate_analog[
                            selected_positions[rows], :k, :, channel
                        ] = aligned_candidates
                        candidate_distance[
                            selected_positions[rows], :k, channel
                        ] = row_distances.astype(np.float32)
                    weighted_delta = np.einsum("bk,bkh->bh", row_weights, deltas[rows])
                    # Direct two-dimensional assignments avoid NumPy's mixed
                    # advanced-index axis reordering.
                    analog[selected_positions[rows], :, channel] = (
                        weighted_delta + query_last[selected_positions[rows], None]
                    ).astype(np.float32)
                    residual = deltas[rows] - weighted_delta[:, None, :]
                    variance = np.einsum(
                        "bk,bkh->bh", row_weights, residual * residual
                    )
                    divergence[selected_positions[rows], :, channel] = np.sqrt(
                        np.maximum(variance, 0.0)
                    ).astype(np.float32)
                    distance_output[selected_positions[rows], channel] = np.einsum(
                        "bk,bk->b", row_weights, distances[rows]
                    ).astype(np.float32)
        if candidate_analog is not None and candidate_distance is not None:
            return candidate_analog, candidate_distance, divergence
        return analog, distance_output, divergence


class GlobalMemoryBank:
    """Shared-window retrieval baseline used to ablate channel-wise lookup.

    Unlike :class:`ChannelMemoryBank`, this baseline concatenates the robustly
    scaled descriptors from every channel and retrieves one common set of
    historical windows.  Each retained window stores all channel futures, so a
    budget of ``memory_per_channel`` windows contains the same number of scalar
    context/future values as ``memory_per_channel`` independent entries for
    every channel.  Its output contract intentionally matches
    ``ChannelMemoryBank.query`` so the forecasting and scoring paths stay
    identical during the ablation.
    """

    def __init__(
        self,
        memory_per_channel: int = 256,
        top_k: int = 5,
        temperature: float = 0.20,
        selection: str = "uniform",
        waveform_points: int = 16,
        query_chunk_size: int = 256,
        memory_chunk_size: int = 1024,
        pool_factor: int = 4,
        eps: float = 1e-6,
    ):
        if min(
            int(memory_per_channel),
            int(top_k),
            int(waveform_points),
            int(query_chunk_size),
            int(memory_chunk_size),
            int(pool_factor),
        ) <= 0:
            raise ValueError("All memory/count/chunk parameters must be positive")
        if temperature <= 0 or eps <= 0:
            raise ValueError("temperature and eps must be positive")
        if selection not in {"uniform", "farthest", "boundary"}:
            raise ValueError("selection must be 'uniform', 'farthest', or 'boundary'")
        self.memory_per_channel = int(memory_per_channel)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.selection = selection
        self.waveform_points = int(waveform_points)
        self.query_chunk_size = int(query_chunk_size)
        self.memory_chunk_size = int(memory_chunk_size)
        self.pool_factor = int(pool_factor)
        self.eps = float(eps)

        self.features_: Optional[np.ndarray] = None
        self.future_deltas_: Optional[np.ndarray] = None
        self.source_end_indices_: Optional[np.ndarray] = None
        self.feature_center_: Optional[np.ndarray] = None
        self.feature_scale_: Optional[np.ndarray] = None
        self.size_per_channel_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None
        self.context_length_: Optional[int] = None
        self.horizon_: Optional[int] = None

    def fit(
        self,
        contexts: np.ndarray,
        futures: np.ndarray,
        normal_mask: Optional[np.ndarray] = None,
        end_indices: Optional[np.ndarray] = None,
    ) -> "GlobalMemoryBank":
        x = np.asarray(contexts)
        y = np.asarray(futures)
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"contexts/futures must be 3-D, got {x.shape}, {y.shape}")
        if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
            raise ValueError(f"Incompatible contexts/futures: {x.shape} versus {y.shape}")
        n, length, channels = x.shape
        horizon = y.shape[1]
        if n == 0:
            raise ValueError("Cannot fit an empty memory bank")
        if normal_mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            provided = np.asarray(normal_mask, dtype=bool)
            if provided.shape == (n,):
                mask = provided
            elif provided.shape == (n, channels):
                mask = provided.all(axis=1)
            else:
                raise ValueError(
                    f"normal_mask must have shape {(n,)} or {(n, channels)}, got {provided.shape}"
                )
        source = (
            np.arange(n, dtype=np.int64)
            if end_indices is None
            else np.asarray(end_indices, dtype=np.int64).reshape(-1)
        )
        if source.shape != (n,):
            raise ValueError(f"end_indices must have shape {(n,)}, got {source.shape}")

        raw = _context_features(x, self.waveform_points, self.eps)
        finite = (
            finite_rows(x)
            & finite_rows(y)
            & finite_rows(raw)
        )
        eligible = np.flatnonzero(mask & finite)
        if not eligible.size:
            raise ValueError("No finite all-channel normal windows for global memory")

        centres = np.empty((channels, raw.shape[-1]), dtype=np.float64)
        scales = np.empty_like(centres)
        scaled = allocate_array((len(eligible), channels * raw.shape[-1]), np.float64)
        for channel in range(channels):
            centre, scale = _robust_feature_scale(raw[eligible, channel, :], self.eps)
            centres[channel] = centre
            scales[channel] = scale
            width = raw.shape[-1]
            scaled[:, channel * width:(channel + 1) * width] = (raw[eligible, channel, :] - centre) / scale
        wanted = min(self.memory_per_channel, len(eligible))
        if self.selection == "uniform":
            selected_local = _uniform_positions(len(eligible), wanted)
        elif self.selection == "farthest":
            selected_local = _farthest_positions(scaled, wanted, self.pool_factor)
        else:
            boundary_wanted = min(
                max(1, int(round(0.20 * wanted))), max(0, len(eligible) - 1)
            )
            core_wanted = wanted - boundary_wanted
            core = _farthest_positions(scaled, max(core_wanted, 1), self.pool_factor)
            available = np.ones(len(eligible), dtype=bool)
            available[core] = False
            boundary_pool = np.flatnonzero(available)
            if boundary_wanted and boundary_pool.size:
                radius = np.einsum("nd,nd->n", scaled, scaled)
                order = np.argsort(radius[boundary_pool], kind="stable")
                selected_local = np.concatenate(
                    (core, boundary_pool[order[-boundary_wanted:]])
                )
            else:
                selected_local = core
            if len(selected_local) < wanted:
                remaining = np.flatnonzero(
                    ~np.isin(np.arange(len(eligible)), selected_local)
                )
                selected_local = np.concatenate(
                    (selected_local, remaining[: wanted - len(selected_local)])
                )
            selected_local = np.sort(selected_local[:wanted])

        selected = eligible[selected_local]
        self.features_ = np.ascontiguousarray(scaled[selected_local], dtype=np.float32)
        self.future_deltas_ = np.ascontiguousarray(
            y[selected].astype(np.float64) - x[selected, -1:, :].astype(np.float64), dtype=np.float32
        )
        self.source_end_indices_ = source[selected].copy()
        self.feature_center_ = centres.astype(np.float32)
        self.feature_scale_ = scales.astype(np.float32)
        self.size_per_channel_ = np.full(channels, wanted, dtype=np.int64)
        self.n_channels_ = channels
        self.context_length_ = length
        self.horizon_ = horizon
        return self

    def _ensure_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.features_,
                self.future_deltas_,
                self.source_end_indices_,
                self.feature_center_,
                self.feature_scale_,
                self.size_per_channel_,
                self.n_channels_,
                self.context_length_,
                self.horizon_,
            )
        ):
            raise RuntimeError("GlobalMemoryBank must be fitted before query")

    def _topk(
        self,
        queries: np.ndarray,
        k: int,
        query_indices: Optional[np.ndarray],
        exclusion_radius: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.features_ is not None
        assert self.source_end_indices_ is not None
        memory = self.features_
        batch = len(queries)
        best_distance = np.full((batch, k), np.inf, dtype=np.float64)
        best_index = np.full((batch, k), -1, dtype=np.int64)
        query_norm = np.einsum("nd,nd->n", queries, queries)[:, None]
        for start in range(0, len(memory), self.memory_chunk_size):
            stop = min(start + self.memory_chunk_size, len(memory))
            current = memory[start:stop].astype(np.float64, copy=False)
            memory_norm = np.einsum("nd,nd->n", current, current)[None, :]
            distance2 = query_norm + memory_norm - 2.0 * queries @ current.T
            np.maximum(distance2, 0.0, out=distance2)
            distance = np.sqrt(distance2 / max(queries.shape[1], 1))
            if query_indices is not None:
                too_close = (
                    np.abs(
                        query_indices[:, None]
                        - self.source_end_indices_[None, start:stop]
                    )
                    <= exclusion_radius
                )
                distance[too_close] = np.inf
            current_index = np.broadcast_to(
                np.arange(start, stop, dtype=np.int64)[None, :], distance.shape
            )
            merged_distance = np.concatenate((best_distance, distance), axis=1)
            merged_index = np.concatenate((best_index, current_index), axis=1)
            positions = np.argpartition(merged_distance, kth=k - 1, axis=1)[:, :k]
            best_distance = np.take_along_axis(merged_distance, positions, axis=1)
            best_index = np.take_along_axis(merged_index, positions, axis=1)
        order = np.argsort(best_distance, axis=1)
        return (
            np.take_along_axis(best_distance, order, axis=1),
            np.take_along_axis(best_index, order, axis=1),
        )

    def query(
        self,
        contexts: np.ndarray,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        query_end_indices: Optional[np.ndarray] = None,
        exclusion_radius: int = 0,
        return_candidates: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_fitted()
        assert self.features_ is not None
        assert self.future_deltas_ is not None
        assert self.feature_center_ is not None
        assert self.feature_scale_ is not None
        x = np.asarray(contexts)
        if x.ndim != 3:
            raise ValueError(f"contexts must have shape [N,L,C], got {x.shape}")
        if x.shape[1] != self.context_length_ or x.shape[2] != self.n_channels_:
            raise ValueError(
                f"Expected [N,{self.context_length_},{self.n_channels_}], got {x.shape}"
            )
        k_requested = self.top_k if top_k is None else int(top_k)
        temp = self.temperature if temperature is None else float(temperature)
        if k_requested <= 0 or temp <= 0:
            raise ValueError("top_k and temperature must be positive")
        exclusion_radius = int(exclusion_radius)
        if exclusion_radius < 0:
            raise ValueError("exclusion_radius cannot be negative")
        query_indices = None
        if query_end_indices is not None:
            query_indices = np.asarray(query_end_indices, dtype=np.int64).reshape(-1)
            if query_indices.shape != (len(x),):
                raise ValueError(
                    f"query_end_indices must have shape {(len(x),)}, got {query_indices.shape}"
                )

        n = len(x)
        horizon = int(self.horizon_)
        channels = int(self.n_channels_)
        raw = _context_features(x, self.waveform_points, self.eps)
        valid = finite_rows(x) & finite_rows(raw)
        scaled = (raw - self.feature_center_[None, :, :]) / self.feature_scale_[
            None, :, :
        ]
        flat = scaled.reshape(n, -1)
        k = min(k_requested, len(self.features_))
        candidate_analog = np.repeat(
            np.where(np.isfinite(x[:, -1, :]), x[:, -1, :], 0.0)[
                :, None, None, :
            ],
            k_requested,
            axis=1,
        )
        candidate_analog = np.repeat(candidate_analog, horizon, axis=2).astype(np.float32)
        candidate_distance = np.full(
            (n, k_requested, channels), np.finfo(np.float32).max, dtype=np.float32
        )
        analog = candidate_analog[:, 0].copy()
        distance_output = np.full(
            (n, channels), np.finfo(np.float32).max, dtype=np.float32
        )
        divergence = np.zeros((n, horizon, channels), dtype=np.float32)

        valid_positions = np.flatnonzero(valid)
        for offset in range(0, len(valid_positions), self.query_chunk_size):
            positions = valid_positions[offset : offset + self.query_chunk_size]
            index_batch = None if query_indices is None else query_indices[positions]
            distances, indices = self._topk(
                flat[positions], k, index_batch, exclusion_radius
            )
            has_neighbour = np.isfinite(distances).any(axis=1)
            if not np.any(has_neighbour):
                continue
            rows = np.flatnonzero(has_neighbour)
            row_positions = positions[rows]
            row_distances = distances[rows]
            safe_indices = np.maximum(indices[rows], 0)
            deltas = self.future_deltas_[safe_indices]
            shifted = row_distances - np.min(row_distances, axis=1, keepdims=True)
            logits = -shifted / temp
            logits[~np.isfinite(row_distances)] = -np.inf
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= np.sum(weights, axis=1, keepdims=True)
            aligned = deltas + x[row_positions, -1, :][:, None, None, :]
            candidate_analog[row_positions, :k] = aligned.astype(np.float32)
            candidate_distance[row_positions, :k] = np.repeat(
                row_distances[:, :, None], channels, axis=2
            ).astype(np.float32)
            weighted_delta = np.einsum("bk,bkhc->bhc", weights, deltas)
            analog[row_positions] = (
                weighted_delta + x[row_positions, -1, :][:, None, :]
            ).astype(np.float32)
            residual = deltas - weighted_delta[:, None, :, :]
            variance = np.einsum("bk,bkhc->bhc", weights, residual * residual)
            divergence[row_positions] = np.sqrt(np.maximum(variance, 0.0)).astype(
                np.float32
            )
            weighted_distance = np.einsum("bk,bk->b", weights, row_distances)
            distance_output[row_positions] = np.repeat(
                weighted_distance[:, None], channels, axis=1
            ).astype(np.float32)
        if return_candidates:
            return candidate_analog, candidate_distance, divergence
        return analog, distance_output, divergence


__all__ = [
    "ChannelMemoryBank",
    "GlobalMemoryBank",
    "RobustScaler",
    "chronological_split",
    "load_series",
    "make_windows",
]
