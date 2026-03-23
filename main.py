import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from cloud_run_logging import bind_request_headers, clear_request_headers, logger

MODEL_PRE_PATH = os.getenv("MODEL_PRE_PATH", "enr_preprocess_ohe.joblib")
MODEL_BST_PATH = os.getenv("MODEL_BST_PATH", "enr_xgb_cuda_booster.json")
META_PATH = os.getenv("META_PATH", "enr_model_meta.json")

pre = None
bst = None
baseline_nacional: float = 0.0436

REQUIRED_FEATURES = [
    "estado_ocurrencia",
    "tamano_localidad_ocurrencia",
    "edad_madre_rango",
    "edad_madre_no_especificada",
    "escolaridad_madre",
    "condicion_actividad_madre",
    "estado_civil_madre_modelo",
    "edad_padre_rango",
    "edad_padre_no_especificada",
    "escolaridad_padre",
    "condicion_actividad_padre",
    "hijos_vivo_bucket",
]


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(...)


class PredictResponse(BaseModel):
    prob_parto_no_institucional: float
    riesgo_relativo_vs_nacional: Optional[float]
    latency_ms: float



def load_artifacts() -> None:
    global pre, bst, baseline_nacional

    logger.info(
        "artifacts_loading",
        model_pre_path=MODEL_PRE_PATH,
        model_bst_path=MODEL_BST_PATH,
        meta_path=META_PATH,
    )

    pre = joblib.load(MODEL_PRE_PATH)

    bst = xgb.Booster()
    bst.load_model(MODEL_BST_PATH)

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        baseline_nacional = float(meta.get("baseline_nacional", baseline_nacional))
    except Exception:
        logger.warning("metadata_load_failed", meta_path=META_PATH, exc_info=True)

    logger.info("artifacts_loaded", baseline_nacional=baseline_nacional)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    try:
        yield
    finally:
        logger.info("app_shutdown")


app = FastAPI(title="Inferencia ENR API", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    bind_request_headers(request.headers)

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "request_failed",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=latency_ms,
            client_host=request.client.host if request.client else None,
        )
        clear_request_headers()


@app.get("/")
def root():
    return {
        "service": "inferencia-enr-api",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {"ok": pre is not None and bst is not None, "baseline_nacional": baseline_nacional}


@app.get("/ping/{ch}")
def ping(ch: str):
    return {"input": ch, "message": "llamado exitoso"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pre is None or bst is None:
        logger.error("prediction_requested_before_model_ready")
        raise HTTPException(status_code=503, detail="Modelo no cargado aún")

    feats = req.features
    missing = [k for k in REQUIRED_FEATURES if k not in feats]
    if missing:
        logger.warning("prediction_missing_features", missing=missing)
        raise HTTPException(status_code=400, detail={"missing": missing})

    t0 = time.perf_counter()
    X = pd.DataFrame([feats])
    Xv = pre.transform(X)
    dX = xgb.DMatrix(Xv)
    p = float(bst.predict(dX)[0])

    rr = (p / baseline_nacional) if baseline_nacional > 0 else None
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "prediction_complete",
        prob_parto_no_institucional=p,
        riesgo_relativo_vs_nacional=rr,
        latency_ms=round(latency_ms, 3),
    )

    return PredictResponse(
        prob_parto_no_institucional=p,
        riesgo_relativo_vs_nacional=rr,
        latency_ms=latency_ms,
    )
