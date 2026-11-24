"""Data-quality gate: reject windows that should never reach feature extraction or the model."""
from __future__ import annotations

import numpy as np

from .simulate import Dataset


def validate_window(
    x: np.ndarray, fs: int, rpm: float, expected_fs: int = 12000, min_len: int = 6000,
    max_abs: float = 50.0, rpm_range: tuple[float, float] = (300, 6000),
) -> list[str]:
    """Return a list of issue codes (empty list = window is valid)."""
    issues: list[str] = []
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < min_len:
        return ["too_short"]
    if not np.all(np.isfinite(x)):
        return ["non_finite"]
    if fs != expected_fs:
        issues.append("wrong_sampling_rate")
    if not (rpm_range[0] <= rpm <= rpm_range[1]):
        issues.append("rpm_out_of_range")
    if np.std(x) < 1e-6:
        issues.append("flatline")
    if np.max(np.abs(x)) > max_abs:
        issues.append("amplitude_out_of_range")
    return issues


def gate_dataset(ds: Dataset, **kw) -> tuple[Dataset, dict[str, int]]:
    """Drop invalid windows; return the clean dataset and a count of issues by code."""
    keep, counts = [], {}
    for i in range(len(ds)):
        issues = validate_window(ds.signals[i], ds.fs, float(ds.rpm[i]), **kw)
        if issues:
            for code in issues:
                counts[code] = counts.get(code, 0) + 1
        else:
            keep.append(i)
    return ds.subset(np.array(keep, dtype=int)), counts
