"""Full-data discovery/readers and bounded temporary arrays; no bundled data."""
from __future__ import annotations

import ast
import re
import tempfile
import weakref
from pathlib import Path

import numpy as np

_SCRATCH = tempfile.TemporaryDirectory(prefix="srf-windows-")
_NEXT = 0


def allocate_array(shape, dtype=np.float64):
    """Temporary disk storage for large arrays, never dataset subsampling."""
    global _NEXT
    dtype = np.dtype(dtype)
    if int(np.prod(shape)) * dtype.itemsize <= 32 * 1024**2:
        return np.empty(shape, dtype=dtype)
    _NEXT += 1
    path = Path(_SCRATCH.name) / f"{_NEXT}.bin"
    result = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    weakref.finalize(result, _release_array, result._mmap, path)
    return result


def _release_array(mapping, path):
    """Release only this helper's own temporary mapping/file."""
    mapping.close()
    path.unlink(missing_ok=True)


def concatenate_arrays(parts):
    """Concatenate complete blocks in bounded chunks, preserving every window."""
    if len(parts) == 1:
        return parts[0]
    result = allocate_array((sum(len(p) for p in parts),) + parts[0].shape[1:], parts[0].dtype)
    offset = 0
    for part in parts:
        for start in range(0, len(part), 256):
            chunk = part[start:start + 256]
            result[offset + start:offset + start + len(chunk)] = chunk
        offset += len(part)
    return result


def finite_rows(values):
    result = np.empty(len(values), dtype=bool)
    axes = tuple(range(1, values.ndim))
    for start in range(0, len(values), 256):
        result[start:start + 256] = np.isfinite(values[start:start + 256]).all(axis=axes)
    return result


def family_of(name):
    return "SMD" if name.startswith("SMD-") else name.split("@", 1)[0]


def discover_datasets(root, requested):
    """Expand SMD/MSL/SMAP to every supplied entity, not a representative subset."""
    root = Path(root).expanduser().resolve()
    result = []
    for name in requested:
        if name == "SMD":
            machines = sorted({m.group(0) for p in root.rglob("*") if p.is_file()
                               and p.suffix.lower() in {".npy", ".txt", ".csv"}
                               for m in [re.search(r"machine-\d+-\d+", p.name)] if m})
            if not machines:
                raise FileNotFoundError("No SMD machine files found under --data-root")
            result.extend("SMD-" + machine for machine in machines)
        elif name in {"MSL", "SMAP"}:
            entities = []
            for path in root.rglob("labeled_anomalies.csv"):
                import pandas as pd
                frame = pd.read_csv(path)
                if {"spacecraft", "chan_id"}.issubset(frame.columns):
                    entities.extend(frame.loc[frame.spacecraft.astype(str).str.upper() == name,
                                              "chan_id"].astype(str).tolist())
            available = [entity for entity in sorted(set(entities))
                         if any(p.parent.name.lower() == "train" for p in root.rglob(entity + ".npy"))]
            if available:
                if len(available) != len(set(entities)):
                    raise FileNotFoundError(f"{name}: raw entity files are incomplete; supply all entities")
                result.extend(name + "@" + entity for entity in available)
            else:
                result.append(name)  # Conventional complete processed arrays.
        else:
            result.append(name)
    return list(dict.fromkeys(result))


def load_nasa_entity(root, name):
    """NASA train/test/{chan_id}.npy plus inclusive intervals from its label CSV."""
    import pandas as pd
    family, entity = name.split("@", 1)
    def locate(role):
        paths = sorted(p for p in Path(root).rglob(entity + ".npy") if p.parent.name.lower() == role)
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected one {role}/{entity}.npy, found {len(paths)}")
        return paths[0]
    train_path, test_path = locate("train"), locate("test")
    rows = []
    for manifest in Path(root).rglob("labeled_anomalies.csv"):
        frame = pd.read_csv(manifest)
        rows.extend(frame[(frame.chan_id.astype(str) == entity) &
                          (frame.spacecraft.astype(str).str.upper() == family)].to_dict("records"))
    if len(rows) != 1:
        raise ValueError(f"Expected one anomaly-interval record for {name}")
    train = np.load(train_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    labels = np.zeros(len(test), dtype=np.int64)
    for start, end in ast.literal_eval(str(rows[0]["anomaly_sequences"])):
        if not 0 <= int(start) <= int(end) < len(test):
            raise ValueError(f"Invalid inclusive anomaly interval for {name}")
        labels[int(start):int(end) + 1] = 1
    return train, test, labels, {
        "dataset": family, "entity": entity,
        "train_path": str(train_path), "test_path": str(test_path),
        "feature_names": [f"channel_{i}" for i in range(train.shape[1])],
    }


def load_smd_text(root, machine):
    def locate(role):
        directories = {"train": {"train"}, "test": {"test"}, "label": {"test_label", "labels"}}[role]
        paths = sorted(p for p in Path(root).rglob(machine + ".*")
                       if p.suffix.lower() in {".txt", ".csv"} and p.parent.name.lower() in directories)
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected one SMD {role}/{machine} text file")
        return paths[0]
    train_path, test_path, label_path = (locate(role) for role in ("train", "test", "label"))
    train = np.loadtxt(train_path, delimiter=",", ndmin=2)
    test = np.loadtxt(test_path, delimiter=",", ndmin=2)
    labels = np.loadtxt(label_path, delimiter=",")
    if labels.ndim > 1:
        labels = np.any(labels != 0, axis=1)
    return train, test, labels.astype(np.int64), {
        "dataset": "SMD", "machine": machine,
        "train_path": str(train_path), "test_path": str(test_path), "label_path": str(label_path),
        "feature_names": [f"channel_{i}" for i in range(train.shape[1])],
    }
