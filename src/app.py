"""Step 3: FastAPI inference service for the quantized ViT."""
import io
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor

from src.quant_utils import load_quantized

MODEL_DIR = os.getenv("MODEL_DIR", "models/quantized")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 1))))
    state["processor"] = AutoImageProcessor.from_pretrained(MODEL_DIR)
    state["model"] = load_quantized(MODEL_DIR)
    state["id2label"] = state["model"].config.id2label
    yield
    state.clear()


app = FastAPI(title="ViT-LoRA INT8 Image Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


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
        raise HTTPException(400, "Uploaded file is not a valid image")

    t0 = time.perf_counter()
    inputs = state["processor"](image, return_tensors="pt")
    with torch.inference_mode():
        logits = state["model"](**inputs).logits
    probs = logits.softmax(-1)[0]
    top = probs.topk(min(top_k, probs.numel()))
    latency = (time.perf_counter() - t0) * 1000

    preds = [{"label": state["id2label"][i.item()], "confidence": round(p.item(), 4)}
             for p, i in zip(top.values, top.indices)]
    return {"prediction": preds[0]["label"], "confidence": preds[0]["confidence"],
            "top_k": preds, "latency_ms": round(latency, 1)}
