import numpy as np
import pytest

from predmaint.pipeline import monitor, run_training
from predmaint.registry import MODEL_NAME, Store, champion_version, load_bundle, promotion_decision, rollback
from predmaint.simulate import CLASSES, SimConfig, make_dataset

A, B = SimConfig(rpm_range=(1500, 1800)), SimConfig(rpm_range=(2200, 2600))
GOOD = {"macro_f1": 0.95, "per_class_recall": [0.9, 0.9, 0.9, 0.9]}


def test_promotion_decision_rules():
    assert promotion_decision(GOOD, None)[0]
    assert not promotion_decision({"macro_f1": 0.5, "per_class_recall": [0.5] * 4}, None)[0]  # below floor
    assert not promotion_decision({"macro_f1": 0.95, "per_class_recall": [1, 1, 1, 0.3]}, None)[0]  # weak class
    assert not promotion_decision(GOOD, {"macro_f1": 0.95})[0]  # no gain
    assert promotion_decision(GOOD, {"macro_f1": 0.90})[0]


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("state"))
    out = {"store": store}
    out["r1"] = run_training(store, make_dataset(60, A, 1), make_dataset(30, A, 2), "generic_envelope")
    out["mon"] = monitor(store, make_dataset(30, B, 3), with_labels=True)
    out["r2"] = run_training(store, make_dataset(60, B, 4), make_dataset(30, B, 5), "physics_orders")
    out["r3"] = run_training(store, make_dataset(60, B, 4), make_dataset(30, B, 5), "physics_orders")
    return out


def test_first_model_becomes_champion(lifecycle):
    assert lifecycle["r1"]["promoted"] and lifecycle["r1"]["reason"] == "no champion yet"


def test_monitoring_detects_the_regime_change(lifecycle):
    m = lifecycle["mon"]
    assert m["drift"] and m["live_macro_f1"] < 0.8 and m["action"] == "retrain" and m["recommend_retrain"]


def test_better_challenger_is_promoted_on_the_same_holdout(lifecycle):
    r2 = lifecycle["r2"]
    assert r2["promoted"]
    assert r2["metrics"]["macro_f1"] > r2["metrics"]["champion_macro_f1_on_holdout"] + 0.2


def test_equal_challenger_is_rejected(lifecycle):
    r3 = lifecycle["r3"]
    assert not r3["promoted"] and "does not beat" in r3["reason"]
    assert champion_version(lifecycle["store"].activate()) == lifecycle["r2"]["version"]


def test_rollback_and_load_bundle(lifecycle):
    store = lifecycle["store"]
    rollback(store, lifecycle["r1"]["version"])
    b = load_bundle(store)
    assert b.version == lifecycle["r1"]["version"] and b.feature_set == "generic_envelope"
    assert b.ref_X.shape[1] == len(b.names)
    rollback(store, lifecycle["r2"]["version"])
    assert load_bundle(store).feature_set == "physics_orders"


def test_api_predict_and_quality_rejection(lifecycle):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from predmaint.serve import create_app

    client = TestClient(create_app(lifecycle["store"]))
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["feature_set"] == "physics_orders"

    ds = make_dataset(3, B, seed=11)
    i = int(np.where(ds.labels == 1)[0][0])
    r = client.post("/predict", json={"samples": ds.signals[i].tolist(), "fs": 12000, "rpm": float(ds.rpm[i])})
    body = r.json()
    assert r.status_code == 200 and body["label"] in CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)

    flat = client.post("/predict", json={"samples": [0.0] * 12000, "fs": 12000, "rpm": 1700})
    assert flat.status_code == 422 and "flatline" in flat.json()["detail"]["quality_issues"]
    assert client.post("/predict", json={"samples": [0.1] * 10, "rpm": 1700}).status_code == 422
    assert MODEL_NAME in {"bearing-fault-classifier"}
