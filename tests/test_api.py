"""End-to-end API test on a tiny random ViT (no downloads, runs in seconds)."""
import io
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.monitoring import build_baseline  # noqa: E402
from src.dev_model import LABELS, make_tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    make_tiny_model(d)
    # baseline built from "healthy" bright images so dark traffic registers as drift
    base = build_baseline([{"prediction": LABELS[i % 10], "confidence": 0.5 + (i % 50) / 100,
                            "features": {"brightness": 0.5 + (i % 10) / 100, "contrast": 0.25 + (i % 10) / 200,
                                         "mean_r": 0.5, "mean_g": 0.5 + (i % 7) / 100, "mean_b": 0.5}}
                           for i in range(200)])
    (d / "reference_stats.json").write_text(json.dumps(base))
    import os
    os.environ.update(MODEL_DIR=str(d), MONITOR_LOG_DIR=str(d / "logs"))
    import importlib
    import src.app as appmod
    importlib.reload(appmod)
    appmod.LiveMonitor.__init__.__defaults__ = (500, 20)   # min_samples=20 for the test
    with TestClient(appmod.app) as c:
        yield c


def png(color=(120, 120, 120)):
    b = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(b, "PNG")
    return b.getvalue()


def test_health_and_labels(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/labels").json()["labels"]) == 10


def test_predict_feedback_metrics(client):
    r = client.post("/predict?top_k=2", files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_k"]) == 2 and body["request_id"]
    fb = client.post("/feedback", json={"request_id": body["request_id"], "true_label": body["prediction"]})
    assert fb.json() == {"status": "recorded", "correct": True}
    assert client.post("/feedback", json={"request_id": "nope", "true_label": "cat"}).status_code == 404
    assert client.post("/feedback", json={"request_id": body["request_id"], "true_label": "dragon"}).status_code == 422
    text = client.get("/metrics").text
    for name in ("vit_requests_total", "vit_inference_latency_seconds_bucket",
                 "vit_rolling_accuracy 1.0", "vit_model_loaded 1.0", "process_resident_memory_bytes"):
        assert name in text


def test_bad_upload(client):
    assert client.post("/predict", files={"file": ("x.txt", b"not an image")}).status_code == 400


def test_drift_detected_on_dark_images(client):
    for _ in range(30):
        client.post("/predict", files={"file": ("d.png", png((10, 10, 10)), "image/png")})
    rep = client.get("/monitoring/drift").json()
    assert rep["status"] == "alert" and rep["features"]["brightness"]["level"] == "alert"
    assert "vit_drift_psi_max" in client.get("/metrics").text
