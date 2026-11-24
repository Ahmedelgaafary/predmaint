"""Feature extraction. Three feature sets are compared in the benchmark:

* generic_spectrum : time statistics + FFT band energies (no knowledge of the machine)
* generic_envelope : time statistics + envelope-spectrum bins in absolute Hz (no knowledge of speed)
* physics_orders   : time statistics + envelope spectrum read at the bearing's fault orders,
                     using the measured shaft speed (order tracking)
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from .bearing import DEFAULT_GEOMETRY, BearingGeometry, fault_frequencies
from .simulate import Dataset

EPS = 1e-12
ENV_BAND = (1500.0, 5500.0)
TIME_NAMES = ["rms", "peak_to_peak", "kurtosis", "skewness", "crest_factor", "shape_factor"]
PHYSICS_NAMES = [f"{n}_x{k}" for n in ("bpfo", "bpfi", "bdf") for k in (1, 2, 3)] + [
    "bpfi_sideband_fr", "bdf_sideband_ftf", "shaft_1x_envelope",
]
FEATURE_SETS = ("generic_spectrum", "generic_envelope", "physics_orders")


def time_features(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    return np.array([
        rms, np.ptp(x), kurtosis(x, fisher=False), skew(x), peak / (rms + EPS), rms / (np.mean(np.abs(x)) + EPS),
    ])


def band_features(x: np.ndarray, fs: int, n_bands: int = 24, fmax: float = 6000.0) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    edges = np.linspace(0, fmax, n_bands + 1)
    return np.array([np.log10(spec[(freqs >= lo) & (freqs < hi)].sum() + EPS) for lo, hi in pairwise(edges)])


def envelope_spectrum(x: np.ndarray, fs: int, band: tuple[float, float] = ENV_BAND) -> tuple[np.ndarray, np.ndarray]:
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env)) / len(env)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def _floor(freqs, spec) -> float:
    return float(np.median(spec[(freqs > 20) & (freqs < 1000)])) + EPS


def envelope_bins(x: np.ndarray, fs: int, fmax: float = 600.0, bin_hz: float = 10.0) -> np.ndarray:
    freqs, spec = envelope_spectrum(x, fs)
    edges = np.arange(0, fmax + bin_hz, bin_hz)
    floor = _floor(freqs, spec)
    return np.array([np.log10(spec[(freqs >= lo) & (freqs < hi)].max() / floor + EPS) for lo, hi in pairwise(edges)])


def _peak_snr(freqs, spec, f0: float, floor: float, tol: float = 0.025) -> float:
    mask = (freqs >= f0 * (1 - tol)) & (freqs <= f0 * (1 + tol))
    return float(np.log10(spec[mask].max() / floor + EPS)) if mask.any() else 0.0


def physics_features(x: np.ndarray, fs: int, rpm: float, geom: BearingGeometry = DEFAULT_GEOMETRY) -> np.ndarray:
    freqs, spec = envelope_spectrum(x, fs)
    f = fault_frequencies(rpm, geom)
    floor = _floor(freqs, spec)
    f["bdf"] = 2.0 * f["bsf"]  # ball defect frequency (ball strikes both races once per spin)
    out = [_peak_snr(freqs, spec, k * f[name], floor) for name in ("bpfo", "bpfi", "bdf") for k in (1, 2, 3)]
    out.append(max(_peak_snr(freqs, spec, f["bpfi"] - f["fr"], floor), _peak_snr(freqs, spec, f["bpfi"] + f["fr"], floor)))
    out.append(max(_peak_snr(freqs, spec, f["bdf"] - f["ftf"], floor), _peak_snr(freqs, spec, f["bdf"] + f["ftf"], floor)))
    out.append(_peak_snr(freqs, spec, f["fr"], floor))
    return np.array(out)


def feature_names(kind: str) -> list[str]:
    if kind == "generic_spectrum":
        return TIME_NAMES + [f"band_{i}" for i in range(24)]
    if kind == "generic_envelope":
        return TIME_NAMES + [f"env_{10 * i}Hz" for i in range(60)]
    if kind == "physics_orders":
        return TIME_NAMES + PHYSICS_NAMES
    raise ValueError(f"unknown feature set {kind!r}")


def extract(x: np.ndarray, fs: int, rpm: float, kind: str, geom: BearingGeometry = DEFAULT_GEOMETRY) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = time_features(x)
    if kind == "generic_spectrum":
        return np.concatenate([t, band_features(x, fs)])
    if kind == "generic_envelope":
        return np.concatenate([t, envelope_bins(x, fs)])
    if kind == "physics_orders":
        return np.concatenate([t, physics_features(x, fs, rpm, geom)])
    raise ValueError(f"unknown feature set {kind!r}")


def build_matrix(ds: Dataset, kind: str) -> np.ndarray:
    return np.vstack([extract(ds.signals[i], ds.fs, float(ds.rpm[i]), kind, ds.geom) for i in range(len(ds))])
