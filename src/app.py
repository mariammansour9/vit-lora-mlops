"""Step 3: FastAPI inference service for the quantized ViT."""
import io
import json
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor

from src import metrics as m
from src.monitoring import LiveMonitor, PredictionLogger, image_features
from src.quant_utils import load_quantized

MODEL_DIR = os.getenv("MODEL_DIR", "models/quantized")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
LOG_DIR = os.getenv("MONITOR_LOG_DIR", "logs")
BASELINE_PATH = os.getenv("BASELINE_PATH", os.path.join(MODEL_DIR, "reference_stats.json"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 1))))
    state["processor"] = AutoImageProcessor.from_pretrained(MODEL_DIR)
    state["model"] = load_quantized(MODEL_DIR)
    state["id2label"] = state["model"].config.id2label
    baseline = json.load(open(BASELINE_PATH)) if os.path.exists(BASELINE_PATH) else None
    state["logger"] = PredictionLogger(LOG_DIR)
    state["monitor"] = LiveMonitor(baseline)
    m.MODEL_LOADED.set(1)
    yield
    m.MODEL_LOADED.set(0)
    state.clear()


app = FastAPI(title="ViT-LoRA INT8 Image Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint (model, drift and process/infrastructure metrics)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class Feedback(BaseModel):
    request_id: str
    true_label: str


@app.post("/feedback")
def feedback(fb: Feedback):
    """Delayed ground truth for a previous prediction -> feeds performance monitoring."""
    logger = state["logger"]
    pred = next((p for p in logger.predictions() if p["id"] == fb.request_id), None)
    if pred is None:
        raise HTTPException(404, "Unknown request_id")
    if fb.true_label not in state["id2label"].values():
        raise HTTPException(422, "Unknown label")
    logger.log_label(fb.request_id, fb.true_label)
    ok = pred["prediction"] == fb.true_label
    m.LABELED.labels(correct=str(ok).lower()).inc()
    state["monitor"].add_outcome(ok)
    acc = state["monitor"].rolling_accuracy()
    if acc is not None:
        m.ROLLING_ACC.set(acc)
    return {"status": "recorded", "correct": ok}


@app.get("/monitoring/drift")
def drift():
    """Current drift report over the rolling window (needs a baseline + enough traffic)."""
    rep = state["monitor"].drift()
    return rep or {"status": "insufficient_data_or_no_baseline"}


@app.get("/labels")
def labels():
    return {"labels": list(state["id2label"].values())}


@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Uploaded file is not a valid image") from None

    m.INFLIGHT.inc()
    try:
        return _predict(image, top_k)
    finally:
        m.INFLIGHT.dec()


def _predict(image, top_k):
    t0 = time.perf_counter()
    inputs = state["processor"](image, return_tensors="pt")
    with torch.inference_mode():
        logits = state["model"](**inputs).logits
    probs = logits.softmax(-1)[0]
    top = probs.topk(min(top_k, probs.numel()))
    latency = (time.perf_counter() - t0) * 1000

    preds = [{"label": state["id2label"][i.item()], "confidence": round(p.item(), 4)}
             for p, i in zip(top.values, top.indices)]

    feats = image_features(image)
    top1, conf = preds[0]["label"], preds[0]["confidence"]
    m.LATENCY.observe(latency / 1000)
    m.CONFIDENCE.observe(conf)
    m.PRED_CLASS.labels(label=top1).inc()
    m.REQUESTS.labels(endpoint="predict", status="200").inc()
    rid = state["logger"].log_prediction(top1, conf, feats, round(latency, 1))
    mon = state["monitor"]
    mon.add({"prediction": top1, "confidence": conf, "features": feats})
    rep = mon.drift()
    if rep:
        for f, v in rep["features"].items():
            m.DRIFT_PSI.labels(feature=f).set(v["psi"])
        m.DRIFT_PSI.labels(feature="prediction_class").set(rep["prediction_class"]["psi"])
        m.DRIFT_MAX.set(rep["max_psi"])
    return {"request_id": rid, "prediction": top1, "confidence": conf,
            "top_k": preds, "latency_ms": round(latency, 1)}
