"""Offline monitoring CLI.

  baseline --from-log logs/predictions.jsonl -o models/quantized/reference_stats.json
  baseline --model-dir models/quantized --samples 1000 -o ...   (runs the real model on CIFAR-10 test)
  report   --baseline ... --logs logs [--window 500]            (drift + performance, exit 1 on alert)
  simulate --out logs --n 400 [--drift]                         (synthetic traffic for demos / CI)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.monitoring import (PredictionLogger, build_baseline, drift_report,
                            performance_report, read_jsonl)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def baseline_from_model(model_dir, samples):  # pragma: no cover - needs torch + weights
    import torch
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    from src.monitoring import image_features
    from src.quant_utils import load_quantized
    proc, model = AutoImageProcessor.from_pretrained(model_dir), load_quantized(model_dir)
    ds = load_dataset("cifar10", split="test").shuffle(seed=0).select(range(samples))
    recs = []
    for ex in ds:
        im = ex["img"].convert("RGB")
        with torch.inference_mode():
            p = model(**proc(im, return_tensors="pt")).logits.softmax(-1)[0]
        recs.append({"prediction": model.config.id2label[int(p.argmax())],
                     "confidence": float(p.max()), "features": image_features(im)})
    return build_baseline(recs)


def simulate(out, n, drift, seed=0):
    """Synthetic traffic: healthy ~90% accurate model, or a drifted/degraded one."""
    rng = np.random.default_rng(seed)
    log = PredictionLogger(out)
    for _ in range(n):
        truth = CLASSES[rng.integers(10)]
        acc = 0.55 if drift else 0.9
        pred = truth if rng.random() < acc else CLASSES[rng.integers(10)]
        if drift:   # darker, low-contrast images; model gets less confident, skews to 3 classes
            b, c = rng.normal(0.25, 0.05), rng.normal(0.08, 0.02)
            conf = float(np.clip(rng.normal(0.62, 0.15), 0.2, 1))
            pred = pred if rng.random() < 0.6 else rng.choice(CLASSES[:3])
        else:
            b, c = rng.normal(0.47, 0.06), rng.normal(0.25, 0.04)
            conf = float(np.clip(rng.normal(0.93 if pred == truth else 0.7, 0.05), 0.2, 1))
        f = {"brightness": b, "contrast": c, "mean_r": b + .02, "mean_g": b, "mean_b": b - .02}
        rid = log.log_prediction(pred, conf, f, float(rng.normal(80, 10)))
        if rng.random() < 0.6:                       # only 60% of predictions get a label
            log.log_label(rid, truth)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--from-log"); b.add_argument("--model-dir"); b.add_argument("--samples", type=int, default=1000)
    b.add_argument("-o", "--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--baseline", required=True); r.add_argument("--logs", default="logs")
    r.add_argument("--window", type=int, default=500)
    r.add_argument("--min-accuracy", type=float, default=0.85)
    s = sub.add_parser("simulate")
    s.add_argument("--out", default="logs"); s.add_argument("--n", type=int, default=400)
    s.add_argument("--drift", action="store_true"); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "simulate":
        simulate(a.out, a.n, a.drift, a.seed)
        print(f"wrote {a.n} synthetic predictions to {a.out}")
        return 0
    if a.cmd == "baseline":
        base = (build_baseline(read_jsonl(a.from_log)) if a.from_log
                else baseline_from_model(a.model_dir, a.samples))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(base))
        print(f"baseline from {base['n']} samples -> {a.out}")
        return 0

    log = PredictionLogger(a.logs)
    preds = log.predictions()[-a.window:]
    base = json.loads(Path(a.baseline).read_text())
    rep = {"drift": drift_report(base, preds), "performance": performance_report(preds, log.labels())}
    print(json.dumps(rep, indent=2))
    perf = rep["performance"]
    bad = rep["drift"]["status"] == "alert" or perf.get("accuracy", 1.0) < a.min_accuracy
    print("STATUS:", "ALERT" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
