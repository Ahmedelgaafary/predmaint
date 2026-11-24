"""Loader for the Case Western Reserve University (CWRU) bearing data, 12 kHz drive-end files.

Data are NOT bundled. Download the .mat files from the CWRU Bearing Data Center (or any mirror) and point
`load_cwru` at the folder. Any folder layout works: files are matched by the leading file number in their
name (`105.mat`, `105_0.mat`, `130@6_0.mat`, ...). Labels map to the same four classes as the simulator.

Excluded on purpose: the 0.028 inch faults (files 3001-3008), which the dataset documents as recorded on a
different bearing, so the bearing geometry used for the physics features would not apply.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from .simulate import CLASSES, Dataset

TARGET_FS = 12000
LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}  # nominal speed at 0..3 hp
NORMAL_FS = 48000  # baseline files 97-100; verified against real data by examples/cwru_physics_check.py


@dataclass(frozen=True)
class CwruFile:
    number: int
    label: int  # index into CLASSES
    diameter_in: float  # fault diameter in inches (0.0 for normal)
    load_hp: int
    position: str | None  # outer-race load-zone position (@3, @6, @12)
    fs: int  # native sampling rate


def _build_catalog() -> dict[int, CwruFile]:
    cat: dict[int, CwruFile] = {}
    for load, num in enumerate((97, 98, 99, 100)):
        cat[num] = CwruFile(num, 0, 0.0, load, None, NORMAL_FS)
    first = {  # first file number (0 hp) per (fault, diameter); files are consecutive for 0..3 hp
        (2, 0.007): 105, (2, 0.014): 169, (2, 0.021): 209,  # inner race
        (3, 0.007): 118, (3, 0.014): 185, (3, 0.021): 222,  # ball
    }
    for (label, dia), start in first.items():
        for load in range(4):
            cat[start + load] = CwruFile(start + load, label, dia, load, None, TARGET_FS)
    outer = {  # outer race, by load-zone position
        ("@6", 0.007): [130, 131, 132, 133], ("@6", 0.014): [197, 198, 199, 200], ("@6", 0.021): [234, 235, 236, 237],
        ("@3", 0.007): [144, 145, 146, 147], ("@3", 0.021): [246, 247, 248, 249],
        ("@12", 0.007): [156, 158, 159, 160], ("@12", 0.021): [258, 259, 260, 261],
    }
    for (pos, dia), nums in outer.items():
        for load, num in enumerate(nums):
            cat[num] = CwruFile(num, 1, dia, load, pos, TARGET_FS)
    return cat


CATALOG = _build_catalog()


def read_mat(path: str | Path) -> dict:
    """Return {'DE': array, 'FE': array|None, 'rpm': float|None} from one CWRU .mat file."""
    raw = loadmat(str(path))
    out: dict = {"DE": None, "FE": None, "rpm": None}
    for key, val in raw.items():
        if key.endswith("_DE_time"):
            out["DE"] = np.asarray(val, dtype=float).ravel()
        elif key.endswith("_FE_time"):
            out["FE"] = np.asarray(val, dtype=float).ravel()
        elif key.endswith("RPM"):
            out["rpm"] = float(np.ravel(val)[0])
    if out["DE"] is None:
        raise ValueError(f"{path}: no drive-end signal (*_DE_time) found")
    return out


def find_files(root: str | Path) -> dict[int, Path]:
    """Map CWRU file number -> path for every .mat under root (layout-agnostic)."""
    found: dict[int, Path] = {}
    for p in sorted(Path(root).rglob("*.mat")):
        m = re.match(r"(\d+)", p.stem)
        if m and int(m.group(1)) in CATALOG:
            found.setdefault(int(m.group(1)), p)
    return found


def _to_target_rate(x: np.ndarray, fs: int) -> np.ndarray:
    if fs == TARGET_FS:
        return x
    g = math.gcd(TARGET_FS, fs)
    return resample_poly(x, TARGET_FS // g, fs // g)


def _windows(x: np.ndarray, size: int, hop: int, segment: tuple[float, float]) -> np.ndarray:
    n = len(x)
    lo, hi = int(segment[0] * n), int(segment[1] * n)
    starts = range(lo, hi - size + 1, hop)
    return np.array([x[s : s + size] for s in starts], dtype=np.float32).reshape(-1, size)


def load_cwru(
    root: str | Path,
    loads: tuple[int, ...] = (0, 1, 2, 3),
    diameters: tuple[float, ...] = (0.007, 0.014, 0.021),
    positions: tuple[str, ...] = ("@6",),
    segment: tuple[float, float] = (0.0, 1.0),
    window_s: float = 1.0,
    hop_s: float = 0.5,
) -> Dataset:
    """Load windows as a `Dataset`.

    `segment` selects a time fraction of each recording, so train/test can be split in time without any
    window overlapping across the boundary (e.g. (0, 0.7) and (0.7, 1.0)). Normal baselines ignore
    `diameters`. Raises if nothing matches, so a wrong path fails loudly instead of returning empty data.
    """
    files = find_files(root)
    size, hop = int(window_s * TARGET_FS), int(hop_s * TARGET_FS)
    sig, rpm, lab, dia, ld, fid = [], [], [], [], [], []
    for num, path in files.items():
        f = CATALOG[num]
        if f.load_hp not in loads:
            continue
        if f.label != 0 and f.diameter_in not in diameters:
            continue
        if f.label == 1 and f.position not in positions:
            continue
        rec = read_mat(path)
        x = _to_target_rate(rec["DE"], f.fs)
        w = _windows(x, size, hop, segment)
        if len(w) == 0:
            continue
        speed = rec["rpm"] if rec["rpm"] else LOAD_RPM[f.load_hp]
        sig.append(w)
        n = len(w)
        rpm += [speed] * n
        lab += [f.label] * n
        dia += [f.diameter_in] * n
        ld += [f.load_hp] * n
        fid += [num] * n
    if not sig:
        raise FileNotFoundError(f"no matching CWRU .mat files found under {root} (found {len(files)} catalogued files)")
    meta = {"file": np.array(fid), "load_hp": np.array(ld), "diameter_in": np.array(dia)}
    return Dataset(
        np.concatenate(sig), np.array(rpm, dtype=float), np.array(lab), np.array(dia, dtype=float), TARGET_FS, meta=meta
    )


def class_counts(ds: Dataset) -> dict[str, int]:
    return {c: int(np.sum(ds.labels == i)) for i, c in enumerate(CLASSES)}


PROTOCOLS = ("temporal", "cross_load", "cross_severity")


def protocol(name: str, root: str | Path) -> tuple[Dataset, Dataset]:
    """(train, test) pair for a named evaluation protocol. Windows never overlap across the split.

    temporal       first 70% of every recording for training, last 30% for testing (easiest; same loads
                   and fault sizes on both sides)
    cross_load     train on 0-1 hp, test on 2-3 hp (different load and slightly different shaft speed)
    cross_severity train on 0.007/0.014 inch faults, test on unseen 0.021 inch faults (last 30% of time)
    """
    if name == "temporal":
        return load_cwru(root, segment=(0.0, 0.7)), load_cwru(root, segment=(0.7, 1.0), hop_s=1.0)
    if name == "cross_load":
        return load_cwru(root, loads=(0, 1)), load_cwru(root, loads=(2, 3), hop_s=1.0)
    if name == "cross_severity":
        return (
            load_cwru(root, diameters=(0.007, 0.014), segment=(0.0, 0.7)),
            load_cwru(root, diameters=(0.021,), segment=(0.7, 1.0), hop_s=1.0),
        )
    raise ValueError(f"unknown protocol {name!r}; choose from {PROTOCOLS}")
