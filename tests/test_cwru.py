import os

import numpy as np
import pytest
from scipy.io import savemat

from predmaint.cwru import CATALOG, PROTOCOLS, find_files, load_cwru, protocol, read_mat
from predmaint.features import build_matrix, feature_names


def write_mat(path, number, signal, rpm=None, fan_end=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {f"X{number:03d}_DE_time": signal.reshape(-1, 1).astype(float)}
    if fan_end:
        d[f"X{number:03d}_FE_time"] = signal.reshape(-1, 1).astype(float)
    if rpm is not None:
        d[f"X{number:03d}RPM"] = np.array([[rpm]])
    savemat(str(path), d)


@pytest.fixture
def full_set(tmp_path):
    """All 40 standard files, tiny: 6 s per fault file at 12 kHz, 6 s normal files at 48 kHz."""
    rng = np.random.default_rng(0)
    for n, f in CATALOG.items():
        if f.label == 1 and f.position != "@6":
            continue
        fs = f.fs
        write_mat(tmp_path / f"d{f.load_hp}" / f"{n}_{f.load_hp}.mat", n, rng.standard_normal(6 * fs))
    return tmp_path


def test_catalog_matches_the_standard_40_file_set():
    std = [f for f in CATALOG.values() if f.position in (None, "@6")]
    assert len(std) == 40
    labels = [f.label for f in std]
    assert [labels.count(c) for c in range(4)] == [4, 12, 12, 12]
    assert not any(3001 <= n <= 3008 for n in CATALOG)  # 0.028 inch files: different bearing, excluded
    assert CATALOG[105].label == 2 and CATALOG[118].label == 3 and CATALOG[130].label == 1


def test_find_files_is_layout_agnostic(tmp_path):
    for rel, n in [("a/105_0.mat", 105), ("b/c/130@6_0.mat", 130), ("97_Normal_0.mat", 97), ("x/9999.mat", 9999)]:
        write_mat(tmp_path / rel, n, np.zeros(10))
    write_mat(tmp_path / "readme.mat", 1, np.zeros(10))
    assert sorted(find_files(tmp_path)) == [97, 105, 130]


def test_read_mat_and_missing_drive_end(tmp_path):
    write_mat(tmp_path / "105.mat", 105, np.arange(5.0), rpm=1797)
    r = read_mat(tmp_path / "105.mat")
    assert r["DE"].tolist() == [0, 1, 2, 3, 4] and r["rpm"] == 1797 and r["FE"] is not None
    savemat(str(tmp_path / "bad.mat"), {"X999_FE_time": np.zeros((5, 1))})
    with pytest.raises(ValueError, match="drive-end"):
        read_mat(tmp_path / "bad.mat")


def test_load_labels_rpm_resampling_and_meta(tmp_path):
    rng = np.random.default_rng(1)
    write_mat(tmp_path / "105.mat", 105, rng.standard_normal(24000), rpm=1790)  # inner race, load 0
    write_mat(tmp_path / "118.mat", 118, rng.standard_normal(24000))  # ball, no RPM -> nominal speed
    write_mat(tmp_path / "130@6.mat", 130, rng.standard_normal(24000), rpm=1796)  # outer race @6
    write_mat(tmp_path / "97.mat", 97, rng.standard_normal(96000), rpm=1796)  # normal, 48 kHz -> 12 kHz
    ds = load_cwru(tmp_path)
    assert ds.signals.shape[1] == 12000 and ds.fs == 12000
    assert len(ds) == 12  # 3 windows (1 s, hop 0.5 s) per 2 s... 24000 samples -> 3 windows, x4 files
    by_file = {int(f): ds.labels[ds.meta["file"] == f][0] for f in np.unique(ds.meta["file"])}
    assert by_file == {97: 0, 105: 2, 118: 3, 130: 1}
    assert set(ds.rpm[ds.meta["file"] == 105]) == {1790.0}
    assert set(ds.rpm[ds.meta["file"] == 118]) == {1797.0}


def test_temporal_segments_never_overlap(tmp_path):
    write_mat(tmp_path / "105.mat", 105, np.arange(60000.0))  # sample value == sample index
    tr = load_cwru(tmp_path, segment=(0.0, 0.7))
    te = load_cwru(tmp_path, segment=(0.7, 1.0))
    assert len(tr) > 0 and len(te) > 0
    assert tr.signals.max() < 0.7 * 60000 <= te.signals.min()


def test_filters_by_load_diameter_and_position(tmp_path):
    write_mat(tmp_path / "105.mat", 105, np.zeros(24000))  # IR 0.007, load 0
    write_mat(tmp_path / "106.mat", 106, np.zeros(24000))  # IR 0.007, load 1
    write_mat(tmp_path / "169.mat", 169, np.zeros(24000))  # IR 0.014, load 0
    write_mat(tmp_path / "144.mat", 144, np.zeros(24000))  # OR 0.007 @3
    assert set(load_cwru(tmp_path, loads=(1,)).meta["file"]) == {106}
    assert set(load_cwru(tmp_path, diameters=(0.014,)).meta["file"]) == {169}
    assert 144 not in set(load_cwru(tmp_path).meta["file"])
    assert 144 in set(load_cwru(tmp_path, positions=("@6", "@3")).meta["file"])


def test_missing_data_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cwru(tmp_path)


def test_protocols_split_as_documented(full_set):
    tr, te = protocol("cross_load", full_set)
    assert set(tr.meta["load_hp"]) == {0, 1} and set(te.meta["load_hp"]) == {2, 3}
    tr, te = protocol("cross_severity", full_set)
    assert set(tr.meta["diameter_in"][tr.labels != 0]) == {0.007, 0.014}
    assert set(te.meta["diameter_in"][te.labels != 0]) == {0.021}
    tr, te = protocol("temporal", full_set)
    assert set(np.unique(tr.labels)) == set(np.unique(te.labels)) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        protocol("nope", full_set)
    assert PROTOCOLS == ("temporal", "cross_load", "cross_severity")


def test_loaded_windows_work_with_the_feature_pipeline(full_set):
    ds = load_cwru(full_set, loads=(0,))
    X = build_matrix(ds, "physics_orders")
    assert X.shape == (len(ds), len(feature_names("physics_orders"))) and np.isfinite(X).all()


@pytest.mark.skipif(not os.getenv("CWRU_DIR"), reason="set CWRU_DIR to a folder with the real CWRU .mat files")
def test_real_data_physics_features_respond_to_the_right_fault():
    ds = load_cwru(os.environ["CWRU_DIR"])
    X, names = build_matrix(ds, "physics_orders"), feature_names("physics_orders")
    i_o, i_i = names.index("bpfo_x1"), names.index("bpfi_x1")
    normal = ds.labels == 0
    assert X[ds.labels == 1, i_o].mean() > X[normal, i_o].mean() + 0.5  # outer race lights up BPFO
    assert X[ds.labels == 2, i_i].mean() > X[normal, i_i].mean() + 0.5  # inner race lights up BPFI
