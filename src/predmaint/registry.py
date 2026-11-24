"""MLflow experiment tracking + model registry with a champion/challenger promotion gate.

All state lives in one directory (`Store`): a SQLite tracking DB and a local artifact folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "bearing-fault-classifier"
EXPERIMENT = "predictive-maintenance"


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{(self.root / 'mlflow.db').resolve()}"
        self.artifact_root = (self.root / "artifacts").resolve().as_uri()

    def activate(self) -> MlflowClient:
        mlflow.set_tracking_uri(self.tracking_uri)
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(EXPERIMENT, artifact_location=self.artifact_root)
        mlflow.set_experiment(EXPERIMENT)
        return MlflowClient(self.tracking_uri)


@dataclass
class Bundle:
    model: object
    feature_set: str
    names: list[str]
    ref_X: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    version: str
    run_id: str
    holdout: dict


def promotion_decision(
    challenger: dict, champion: dict | None, min_f1: float = 0.8, min_gain: float = 0.01, min_recall: float = 0.6
) -> tuple[bool, str]:
    """Pure function: should the challenger replace the champion? Both are metric dicts on the SAME holdout."""
    if challenger["macro_f1"] < min_f1:
        return False, f"macro-F1 {challenger['macro_f1']:.3f} below floor {min_f1}"
    if min(challenger["per_class_recall"]) < min_recall:
        return False, f"a class recall {min(challenger['per_class_recall']):.3f} is below floor {min_recall}"
    if champion is None:
        return True, "no champion yet"
    if challenger["macro_f1"] >= champion["macro_f1"] + min_gain:
        return True, f"macro-F1 {challenger['macro_f1']:.3f} beats champion {champion['macro_f1']:.3f} by >= {min_gain}"
    return False, f"macro-F1 {challenger['macro_f1']:.3f} does not beat champion {champion['macro_f1']:.3f} by {min_gain}"


def champion_version(client: MlflowClient, alias: str = "champion") -> str | None:
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except MlflowException:  # alias not set yet
        return None


def log_candidate(
    store: Store, model, feature_set: str, names: list[str], X_ref: np.ndarray, metrics: dict, params: dict
) -> str:
    client = store.activate()
    lo, hi = np.quantile(X_ref, 0.005, axis=0), np.quantile(X_ref, 0.995, axis=0)
    flat = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    for i, r in enumerate(metrics["per_class_recall"]):
        flat[f"recall_class_{i}"] = r
    with mlflow.start_run(tags={"kind": "training"}) as run:
        mlflow.log_params({"feature_set": feature_set, **params})
        mlflow.log_metrics(flat)
        mlflow.log_text(
            json.dumps({
                "feature_set": feature_set, "names": names, "X": np.round(X_ref, 6).tolist(),
                "lo": lo.tolist(), "hi": hi.tolist(), "holdout": metrics,
            }),
            "reference.json",
        )
        try:
            mlflow.sklearn.log_model(
                model, name="model", registered_model_name=MODEL_NAME, serialization_format="cloudpickle"
            )
        except TypeError:  # older MLflow
            mlflow.sklearn.log_model(
                model, artifact_path="model", registered_model_name=MODEL_NAME, serialization_format="cloudpickle"
            )
        run_id = run.info.run_id
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    mine = [v for v in versions if v.run_id == run_id] or sorted(versions, key=lambda v: int(v.version))[-1:]
    version = str(mine[0].version)
    client.set_model_version_tag(MODEL_NAME, version, "feature_set", feature_set)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    return version


def promote(store: Store, version: str, reason: str = "") -> None:
    client = store.activate()
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    client.set_model_version_tag(MODEL_NAME, str(version), "promotion_reason", reason or "manual")


def rollback(store: Store, version: str) -> None:
    promote(store, version, reason="rollback")


def load_bundle(store: Store, alias: str = "champion") -> Bundle:
    client = store.activate()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    ref = json.loads(Path(client.download_artifacts(mv.run_id, "reference.json")).read_text())
    return Bundle(
        model, ref["feature_set"], ref["names"], np.array(ref["X"]), np.array(ref["lo"]), np.array(ref["hi"]),
        str(mv.version), mv.run_id, ref["holdout"],
    )
