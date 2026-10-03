import json

import numpy as np
from PIL import Image

from src import monitor_cli
from src.monitoring import (PredictionLogger, build_baseline, categorical_psi, drift_report,
                            image_features, ks_statistic, performance_report, psi)


def test_psi_same_distribution_is_small_and_shift_is_large():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(2, 1, 5000)) > 0.25


def test_ks_bounds():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    assert ks_statistic(a, a) == 0
    assert ks_statistic(a, a + 10) > 0.99


def test_categorical_psi():
    assert categorical_psi({"a": 50, "b": 50}, {"a": 50, "b": 50}) < 1e-6
    assert categorical_psi({"a": 50, "b": 50}, {"a": 95, "b": 5}) > 0.25


def test_image_features_ranges():
    f = image_features(Image.new("RGB", (32, 32), (255, 0, 0)))
    assert f["mean_r"] > 0.99 and f["mean_g"] < 0.01 and f["contrast"] < 0.01


def test_performance_report_accuracy_and_coverage():
    preds = [{"id": str(i), "prediction": "a" if i % 2 else "b", "confidence": 0.9} for i in range(10)]
    labels = {str(i): "a" for i in range(5)}        # only half labelled
    r = performance_report(preds, labels)
    assert r["label_coverage"] == 0.5 and r["n_labeled"] == 5
    assert r["accuracy"] == round(2 / 5, 4)         # ids 1 and 3 are "a"
    assert performance_report(preds, {})["status"] == "no_labels"


def test_logger_roundtrip(tmp_path):
    log = PredictionLogger(tmp_path)
    rid = log.log_prediction("cat", 0.9, {"brightness": 0.5}, 10.0)
    log.log_label(rid, "dog")
    assert log.predictions()[0]["id"] == rid and log.labels() == {rid: "dog"}


def test_end_to_end_healthy_vs_drifted(tmp_path):
    healthy, drifted = tmp_path / "h", tmp_path / "d"
    monitor_cli.simulate(healthy, 600, drift=False, seed=1)
    monitor_cli.simulate(drifted, 600, drift=True, seed=2)
    base = build_baseline(PredictionLogger(healthy).predictions())
    ok = drift_report(base, PredictionLogger(tmp_path / "h").predictions()[-300:])
    bad = drift_report(base, PredictionLogger(drifted).predictions())
    assert ok["status"] == "ok"
    assert bad["status"] == "alert" and bad["features"]["brightness"]["level"] == "alert"
    p = PredictionLogger(drifted)
    assert performance_report(p.predictions(), p.labels())["accuracy"] < 0.7


def test_cli_exit_codes(tmp_path, capsys):
    monitor_cli.main(["simulate", "--out", str(tmp_path / "h"), "--n", "500"])
    bl = tmp_path / "base.json"
    monitor_cli.main(["baseline", "--from-log", str(tmp_path / "h" / "predictions.jsonl"), "-o", str(bl)])
    monitor_cli.main(["simulate", "--out", str(tmp_path / "d"), "--n", "500", "--drift", "--seed", "5"])
    assert monitor_cli.main(["report", "--baseline", str(bl), "--logs", str(tmp_path / "h")]) == 0
    assert monitor_cli.main(["report", "--baseline", str(bl), "--logs", str(tmp_path / "d")]) == 1
    assert json.loads(bl.read_text())["n"] == 500
