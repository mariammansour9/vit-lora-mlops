"""AI observability: data drift, model performance and Prometheus metrics.

Pure numpy (no torch) so it can be unit-tested and run offline on prediction logs.

Three layers:
  1. Logging      - every prediction is appended to a JSONL file (PredictionLogger);
                    ground-truth labels arrive later through /feedback (label log).
  2. Drift        - PSI + KS statistic of live input features / confidence / predicted
                    classes versus a reference baseline captured at training time.
  3. Performance  - accuracy, macro-F1, calibration gap (ECE) from joined predictions+labels.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import numpy as np

EPS = 1e-6
PSI_WARN, PSI_ALERT = 0.1, 0.25          # standard PSI rules of thumb
FEATURES = ("brightness", "contrast", "mean_r", "mean_g", "mean_b")


# --------------------------------------------------------------------------- features
def image_features(image) -> dict:
    """Cheap, model-agnostic input statistics of a PIL RGB image (data-drift signals)."""
    a = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    gray = a.mean(axis=2)
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()),
            "mean_r": float(a[..., 0].mean()), "mean_g": float(a[..., 1].mean()),
            "mean_b": float(a[..., 2].mean())}


# --------------------------------------------------------------------------- drift maths
def psi(reference, live, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the reference sample."""
    ref, cur = np.asarray(reference, float), np.asarray(live, float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # constant reference -> any change is drift
        return 0.0 if np.allclose(cur, ref[0]) else PSI_ALERT
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def ks_statistic(reference, live) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF distance, 0..1)."""
    a, b = np.sort(np.asarray(reference, float)), np.sort(np.asarray(live, float))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def categorical_psi(reference: dict, live: dict) -> float:
    keys = set(reference) | set(live)
    rt, lt = sum(reference.values()) or 1, sum(live.values()) or 1
    return float(sum(((live.get(k, 0) / lt + EPS) - (reference.get(k, 0) / rt + EPS))
                     * np.log((live.get(k, 0) / lt + EPS) / (reference.get(k, 0) / rt + EPS))
                     for k in keys))


def drift_level(score: float) -> str:
    return "alert" if score >= PSI_ALERT else "warn" if score >= PSI_WARN else "ok"


# --------------------------------------------------------------------------- baseline
def build_baseline(records: list[dict]) -> dict:
    """Reference statistics from a trusted dataset (validation set / first healthy week).

    records: [{"features": {...}, "confidence": float, "prediction": str, ["label": str]}]
    """
    base = {"n": len(records), "features": {}, "created": time.time()}
    for f in (*FEATURES, "confidence"):
        vals = [r["features"][f] if f in FEATURES else r[f] for r in records]
        base["features"][f] = [float(v) for v in vals]
    base["classes"] = dict(Counter(r["prediction"] for r in records))
    return base


def drift_report(baseline: dict, live_records: list[dict]) -> dict:
    """PSI/KS per feature + prediction-class PSI. Overall status = worst feature."""
    out = {"n_live": len(live_records), "features": {}}
    if not live_records:
        out["status"] = "no_data"
        return out
    for f in (*FEATURES, "confidence"):
        live = [r["features"][f] if f in FEATURES else r[f] for r in live_records]
        p = psi(baseline["features"][f], live)
        out["features"][f] = {"psi": round(p, 4),
                              "ks": round(ks_statistic(baseline["features"][f], live), 4),
                              "level": drift_level(p)}
    cp = categorical_psi(baseline["classes"], Counter(r["prediction"] for r in live_records))
    out["prediction_class"] = {"psi": round(cp, 4), "level": drift_level(cp)}
    worst = max([v["psi"] for v in out["features"].values()] + [cp])
    out["max_psi"], out["status"] = round(worst, 4), drift_level(worst)
    return out


# --------------------------------------------------------------------------- performance
def performance_report(preds: list[dict], labels: dict[str, str], n_bins: int = 10) -> dict:
    """Join predictions to ground truth by request id and score them."""
    joined = [(p, labels[p["id"]]) for p in preds if p["id"] in labels]
    out = {"n_predictions": len(preds), "n_labeled": len(joined),
           "label_coverage": round(len(joined) / len(preds), 4) if preds else 0.0}
    if not joined:
        out["status"] = "no_labels"
        return out
    y_pred = np.array([p["prediction"] for p, _ in joined])
    y_true = np.array([t for _, t in joined])
    conf = np.array([p["confidence"] for p, _ in joined])
    correct = (y_pred == y_true).astype(float)
    f1s = {}
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp, fn = np.sum((y_pred == c) & (y_true != c)), np.sum((y_pred != c) & (y_true == c))
        f1s[str(c)] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    ece, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    out.update(accuracy=round(float(correct.mean()), 4),
               macro_f1=round(float(np.mean(list(f1s.values()))), 4),
               mean_confidence=round(float(conf.mean()), 4),
               ece=round(float(ece), 4),
               per_class_f1={k: round(v, 4) for k, v in sorted(f1s.items())})
    return out


# --------------------------------------------------------------------------- persistence
def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class PredictionLogger:
    """Thread-safe JSONL logs: predictions.jsonl and labels.jsonl (delayed ground truth)."""

    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pred_path, self.label_path = self.dir / "predictions.jsonl", self.dir / "labels.jsonl"
        self._lock = threading.Lock()

    def _append(self, path: Path, rec: dict):
        with self._lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_prediction(self, prediction, confidence, features, latency_ms) -> str:
        rid = uuid.uuid4().hex
        self._append(self.pred_path, {"id": rid, "ts": time.time(), "prediction": prediction,
                                      "confidence": confidence, "features": features,
                                      "latency_ms": latency_ms})
        return rid

    def log_label(self, request_id: str, label: str):
        self._append(self.label_path, {"id": request_id, "ts": time.time(), "label": label})

    def predictions(self) -> list[dict]:
        return read_jsonl(self.pred_path)

    def labels(self) -> dict[str, str]:
        return {r["id"]: r["label"] for r in read_jsonl(self.label_path)}


class LiveMonitor:
    """Rolling window of recent predictions -> drift/performance gauges for Prometheus."""

    def __init__(self, baseline: dict | None, window: int = 500, min_samples: int = 50):
        self.baseline, self.min_samples = baseline, min_samples
        self.recent = deque(maxlen=window)
        self.outcomes = deque(maxlen=window)         # 1/0 per labelled prediction
        self._lock = threading.Lock()

    def add(self, rec: dict):
        with self._lock:
            self.recent.append(rec)

    def add_outcome(self, correct: bool):
        with self._lock:
            self.outcomes.append(1.0 if correct else 0.0)

    def drift(self) -> dict | None:
        with self._lock:
            live = list(self.recent)
        if self.baseline is None or len(live) < self.min_samples:
            return None
        return drift_report(self.baseline, live)

    def rolling_accuracy(self) -> float | None:
        with self._lock:
            return float(np.mean(self.outcomes)) if self.outcomes else None
