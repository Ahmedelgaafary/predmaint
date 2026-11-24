"""Physics-based vibration simulator for a faulty bearing (SYNTHETIC data).

Model: shaft harmonics + a structural resonance excited by (a) periodic fault impacts at the
characteristic frequency (with slip jitter and the physical amplitude modulation of each fault type)
and (b) random non-fault impacts, plus Gaussian noise. Replace with real data (e.g. CWRU, NASA IMS)
by producing a `Dataset` from your files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from .bearing import DEFAULT_GEOMETRY, BearingGeometry, fault_frequencies

CLASSES = ["normal", "outer_race", "inner_race", "ball"]
_FAULT_KEY = {1: "bpfo", 2: "bpfi", 3: "bsf"}


@dataclass
class SimConfig:
    fs: int = 12000
    duration: float = 1.0
    rpm_range: tuple[float, float] = (1500, 1800)
    severity_range: tuple[float, float] = (0.4, 2.0)
    noise_std_range: tuple[float, float] = (0.2, 0.6)
    f_res_range: tuple[float, float] = (2500, 4500)
    damping: float = 0.04  # fraction of the resonance frequency
    slip: float = 0.01  # relative jitter of the impact period
    gain: float = 1.0  # sensor gain (use for drift experiments)
    distractor_rate: tuple[int, int] = (3, 10)  # random non-fault impacts per window


@dataclass
class Dataset:
    signals: np.ndarray  # (n, L)
    rpm: np.ndarray  # (n,)
    labels: np.ndarray  # (n,)
    severity: np.ndarray  # (n,)
    fs: int
    geom: BearingGeometry = field(default_factory=lambda: DEFAULT_GEOMETRY)
    meta: dict[str, np.ndarray] | None = None  # optional per-window metadata (file id, load, ...)

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx) -> Dataset:
        meta = {k: v[idx] for k, v in self.meta.items()} if self.meta else None
        return Dataset(self.signals[idx], self.rpm[idx], self.labels[idx], self.severity[idx], self.fs, self.geom, meta)


def _resonator(f_res: float, fs: int, damping: float):
    theta = 2 * np.pi * f_res / fs
    r = np.exp(-damping * theta)
    return [1.0], [1.0, -2 * r * np.cos(theta), r * r]


def generate_window(
    label: int, rpm: float, rng: np.random.Generator, cfg: SimConfig, geom: BearingGeometry = DEFAULT_GEOMETRY,
    severity: float = 1.0, noise_std: float = 0.4,
) -> np.ndarray:
    n = int(cfg.fs * cfg.duration)
    t = np.arange(n) / cfg.fs
    freqs = fault_frequencies(rpm, geom)
    fr = freqs["fr"]
    x = np.zeros(n)
    for k, a in enumerate([1.0, 0.4, 0.2], start=1):  # shaft harmonics
        x += a * rng.uniform(0.7, 1.3) * np.sin(2 * np.pi * k * fr * t + rng.uniform(0, 2 * np.pi))

    f_res = rng.uniform(*cfg.f_res_range)
    b, a = _resonator(f_res, cfg.fs, cfg.damping)
    impulses = np.zeros(n)

    n_distract = rng.integers(cfg.distractor_rate[0], cfg.distractor_rate[1] + 1)
    np.add.at(impulses, rng.integers(0, n, n_distract), rng.uniform(0.3, 1.0, n_distract))

    if label != 0:
        f_fault = freqs[_FAULT_KEY[label]] * (2.0 if label == 3 else 1.0)  # ball defect hits both races per spin
        period = 1.0 / f_fault
        steps = period * (1 + cfg.slip * rng.standard_normal(int(f_fault * cfg.duration) + 3))
        times = rng.uniform(0, period) + np.cumsum(steps)
        times = times[times < cfg.duration]
        amp = severity * (1 + 0.1 * rng.standard_normal(len(times)))
        if label == 2:  # inner race: load-zone modulation at shaft frequency
            amp *= 1 + 0.8 * np.cos(2 * np.pi * fr * times + rng.uniform(0, 2 * np.pi))
        elif label == 3:  # ball defect (2 x BSF): modulation at cage frequency
            amp *= 1 + 0.8 * np.cos(2 * np.pi * freqs["ftf"] * times + rng.uniform(0, 2 * np.pi))
        np.add.at(impulses, np.minimum((times * cfg.fs).astype(int), n - 1), amp)

    x += lfilter(b, a, impulses)
    x += noise_std * rng.standard_normal(n)
    return cfg.gain * x


def make_dataset(
    n_per_class: int, cfg: SimConfig | None = None, seed: int = 0, geom: BearingGeometry = DEFAULT_GEOMETRY
) -> Dataset:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    sig, rpms, labels, sevs = [], [], [], []
    for label in range(len(CLASSES)):
        for _ in range(n_per_class):
            rpm = rng.uniform(*cfg.rpm_range)
            sev = 0.0 if label == 0 else rng.uniform(*cfg.severity_range)
            noise = rng.uniform(*cfg.noise_std_range)
            sig.append(generate_window(label, rpm, rng, cfg, geom, sev, noise).astype(np.float32))
            rpms.append(rpm)
            labels.append(label)
            sevs.append(sev)
    order = rng.permutation(len(labels))
    return Dataset(
        np.array(sig)[order], np.array(rpms)[order], np.array(labels)[order], np.array(sevs)[order], cfg.fs, geom
    )
