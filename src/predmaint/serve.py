"""FastAPI inference service for the current champion. State dir: env PREDMAINT_STATE (default ./mlops_state)."""
from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import extract
from .quality import validate_window
from .registry import Bundle, Store, load_bundle
from .simulate import CLASSES


class PredictRequest(BaseModel):
    samples: list[float] = Field(min_length=100)
    fs: int = 12000
    rpm: float


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(title="predmaint", version="0.1.0")
    state: dict[str, Bundle | None] = {"bundle": None}
    the_store = store or Store(os.getenv("PREDMAINT_STATE", "mlops_state"))

    def bundle() -> Bundle:
        if state["bundle"] is None:
            try:
                state["bundle"] = load_bundle(the_store)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"no champion model available: {exc}") from exc
        return state["bundle"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        b = bundle()
        return {"version": b.version, "feature_set": b.feature_set, "holdout": b.holdout}

    @app.post("/reload")
    def reload() -> dict:
        state["bundle"] = None
        return {"version": bundle().version}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        b = bundle()
        x = np.asarray(req.samples, dtype=float)
        issues = validate_window(x, req.fs, req.rpm)
        if issues:
            raise HTTPException(status_code=422, detail={"quality_issues": issues})
        feats = extract(x, req.fs, req.rpm, b.feature_set)
        proba = b.model.predict_proba(feats.reshape(1, -1))[0]
        ood_share = float(np.mean((feats < b.lo) | (feats > b.hi)))
        return {
            "label": CLASSES[int(np.argmax(proba))],
            "probabilities": {c: float(p) for c, p in zip(CLASSES, proba, strict=True)},
            "model_version": b.version,
            "feature_set": b.feature_set,
            "out_of_range_feature_share": ood_share,
            "low_confidence_or_ood": bool(ood_share > 0.3 or proba.max() < 0.6),
        }

    return app


def get_app() -> FastAPI:  # `uvicorn predmaint.serve:get_app --factory`
    return create_app()
