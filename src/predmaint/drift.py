"""Feature drift detection: Population Stability Index + two-sample Kolmogorov-Smirnov test."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with quantile bins taken from the reference sample. Rule of thumb: >0.2 is a major shift."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class DriftReport:
    drift: bool
    share: float  # share of features flagged
    per_feature: dict[str, dict[str, float]] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    severe: list[str] = field(default_factory=list)  # single features shifted far beyond the PSI threshold

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(((n, v["psi"]) for n, v in self.per_feature.items()), key=lambda t: -t[1])[:k]


def detect_drift(
    ref: np.ndarray, cur: np.ndarray, names: list[str], psi_threshold: float = 0.2, ks_alpha: float = 0.01,
    share_threshold: float = 0.2, min_samples: int = 30, severe_psi: float = 1.0,
) -> DriftReport:
    """A feature is flagged when BOTH PSI and KS agree (fewer false alarms). Drift = enough features flagged,
    or any single feature shifted severely (e.g. a sensor gain error moves RMS by orders of magnitude)."""
    if len(cur) < min_samples:
        raise ValueError(f"need at least {min_samples} windows to assess drift, got {len(cur)}")
    per, flagged, severe = {}, [], []
    for j, name in enumerate(names):
        p = psi(ref[:, j], cur[:, j])
        ks_p = float(ks_2samp(ref[:, j], cur[:, j]).pvalue)
        per[name] = {"psi": p, "ks_p": ks_p}
        if p > psi_threshold and ks_p < ks_alpha:
            flagged.append(name)
            if p > severe_psi:
                severe.append(name)
    share = len(flagged) / len(names)
    return DriftReport(share >= share_threshold or bool(severe), share, per, flagged, severe)
