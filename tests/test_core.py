import numpy as np
import pytest

from predmaint.bearing import BearingGeometry, fault_frequencies
from predmaint.drift import detect_drift, psi
from predmaint.features import FEATURE_SETS, build_matrix, extract, feature_names
from predmaint.quality import gate_dataset, validate_window
from predmaint.simulate import CLASSES, SimConfig, make_dataset


def test_cwru_6205_fault_orders():
    o = BearingGeometry().fault_orders()
    assert o["bpfo"] == pytest.approx(3.5848, abs=1e-3)
    assert o["bpfi"] == pytest.approx(5.4152, abs=1e-3)
    assert o["ftf"] == pytest.approx(0.39828, abs=1e-3)
    assert 2 * o["bsf"] == pytest.approx(4.7135, abs=1e-3)  # CWRU "ball defect" = 2 x BSF


def test_fault_frequencies_scale_with_speed():
    f = fault_frequencies(1800)
    assert f["fr"] == pytest.approx(30.0)
    assert f["bpfo"] == pytest.approx(3.5848 * 30, abs=0.05)


def test_simulation_is_deterministic_and_balanced():
    a, b = make_dataset(5, seed=3), make_dataset(5, seed=3)
    assert np.array_equal(a.signals, b.signals)
    assert a.signals.shape == (20, 12000)
    assert np.bincount(a.labels).tolist() == [5] * len(CLASSES)
    assert np.all(a.severity[a.labels == 0] == 0)


@pytest.mark.parametrize("kind", FEATURE_SETS)
def test_feature_dims_match_names(kind):
    ds = make_dataset(2, seed=1)
    X = build_matrix(ds, kind)
    assert X.shape == (8, len(feature_names(kind))) and np.isfinite(X).all()


def test_physics_feature_responds_to_the_right_fault():
    ds = make_dataset(20, SimConfig(severity_range=(1.5, 2.0), noise_std_range=(0.2, 0.3)), seed=7)
    X = build_matrix(ds, "physics_orders")
    names = feature_names("physics_orders")
    i_bpfo, i_bpfi = names.index("bpfo_x1"), names.index("bpfi_x1")
    assert X[ds.labels == 1, i_bpfo].mean() > X[ds.labels == 0, i_bpfo].mean() + 0.3  # outer race
    assert X[ds.labels == 2, i_bpfi].mean() > X[ds.labels == 0, i_bpfi].mean() + 0.3  # inner race


def test_extract_rejects_unknown_feature_set():
    with pytest.raises(ValueError):
        extract(np.zeros(100), 12000, 1500, "nope")


def test_quality_gate_codes():
    ok = np.random.default_rng(0).standard_normal(12000)
    assert validate_window(ok, 12000, 1700) == []
    assert validate_window(ok[:100], 12000, 1700) == ["too_short"]
    bad = ok.copy()
    bad[5] = np.nan
    assert validate_window(bad, 12000, 1700) == ["non_finite"]
    assert "flatline" in validate_window(np.zeros(12000), 12000, 1700)
    assert "wrong_sampling_rate" in validate_window(ok, 8000, 1700)
    assert "rpm_out_of_range" in validate_window(ok, 12000, 20)
    assert "amplitude_out_of_range" in validate_window(ok * 100, 12000, 1700)


def test_gate_dataset_drops_bad_windows():
    ds = make_dataset(5, seed=2)
    ds.signals[0, :] = 0.0
    clean, counts = gate_dataset(ds)
    assert len(clean) == len(ds) - 1 and counts == {"flatline": 1}


def test_psi_identical_is_near_zero_and_shift_is_large():
    r = np.random.default_rng(0)
    a, b = r.normal(size=2000), r.normal(size=2000)
    assert psi(a, b) < 0.05
    assert psi(a, b + 2) > 0.5


def test_drift_detection_scenarios():
    names = feature_names("physics_orders")
    ref = build_matrix(make_dataset(40, seed=1), "physics_orders")
    same = build_matrix(make_dataset(20, seed=2), "physics_orders")
    assert not detect_drift(ref, same, names).drift
    gain = build_matrix(make_dataset(20, SimConfig(gain=2.5), seed=3), "physics_orders")
    rep = detect_drift(ref, gain, names)
    assert rep.drift and "rms" in rep.severe  # a sensor gain error is caught as severe drift


def test_drift_needs_enough_samples():
    with pytest.raises(ValueError):
        detect_drift(np.zeros((50, 2)), np.zeros((5, 2)), ["a", "b"])
