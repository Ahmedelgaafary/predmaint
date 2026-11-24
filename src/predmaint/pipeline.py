"""Training and monitoring pipelines: gate -> features -> train -> evaluate -> compare -> register/promote."""
from __future__ import annotations

import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .drift import detect_drift
from .features import build_matrix, feature_names
from .quality import gate_dataset
from .registry import Store, champion_version, load_bundle, log_candidate, promote, promotion_decision
from .simulate import CLASSES, Dataset


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_recall": [float(r) for r in recall_score(y, pred, labels=range(len(CLASSES)), average=None)],
    }


def run_training(
    store: Store, train_ds: Dataset, holdout_ds: Dataset, feature_set: str, seed: int = 0,
    min_f1: float = 0.8, min_gain: float = 0.01, auto_promote: bool = True,
) -> dict:
    """Train a challenger and promote it only if it beats the champion on the SAME holdout."""
    clean, issues = gate_dataset(train_ds)
    X = build_matrix(clean, feature_set)
    model = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(X, clean.labels)
    metrics = evaluate_model(model, build_matrix(holdout_ds, feature_set), holdout_ds.labels)

    client = store.activate()
    champ_metrics = None
    if champion_version(client) is not None:
        champ = load_bundle(store)
        champ_metrics = evaluate_model(champ.model, build_matrix(holdout_ds, champ.feature_set), holdout_ds.labels)
        metrics["champion_macro_f1_on_holdout"] = champ_metrics["macro_f1"]
    promote_it, reason = promotion_decision(metrics, champ_metrics, min_f1=min_f1, min_gain=min_gain)

    version = log_candidate(
        store, model, feature_set, feature_names(feature_set), X, metrics,
        {"n_train": len(clean), "dropped_windows": len(train_ds) - len(clean), "seed": seed, "promotion_reason": reason},
    )
    if promote_it and auto_promote:
        promote(store, version, reason)
    return {
        "version": version, "feature_set": feature_set, "metrics": metrics, "promoted": bool(promote_it and auto_promote),
        "reason": reason, "quality_issues": issues,
    }


def monitor(store: Store, current: Dataset, with_labels: bool = False, min_f1: float = 0.8) -> dict:
    """Compare production windows with the champion's training reference; optionally score if labels exist."""
    bundle = load_bundle(store)
    clean, issues = gate_dataset(current)
    X = build_matrix(clean, bundle.feature_set)
    report = detect_drift(bundle.ref_X, X, bundle.names)
    out = {
        "champion_version": bundle.version, "feature_set": bundle.feature_set, "n_windows": len(clean),
        "quality_issues": issues, "drift": report.drift, "drift_share": report.share, "top_psi": report.top(),
    }
    out["severe_drift_features"] = report.severe
    if with_labels:
        out["live_macro_f1"] = evaluate_model(bundle.model, X, clean.labels)["macro_f1"]
        if out["live_macro_f1"] < min_f1:
            action = "retrain"
        else:
            action = "watch: drift without performance loss" if report.drift else "ok"
    else:
        action = "retrain_candidate: confirm with labels" if report.drift else "ok"
    out["action"] = action
    out["recommend_retrain"] = action.startswith("retrain")
    store.activate()
    with mlflow.start_run(tags={"kind": "monitoring", "champion_version": bundle.version}):
        metrics = {"drift_share": report.share, "n_windows": len(clean)}
        if with_labels:
            metrics["live_macro_f1"] = out["live_macro_f1"]
        mlflow.log_metrics(metrics)
        mlflow.log_param("recommend_retrain", out["recommend_retrain"])
    return out
