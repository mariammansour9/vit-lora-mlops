"""Prometheus metric definitions for the inference service (model + infrastructure)."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("vit_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("vit_inference_latency_seconds", "Model inference latency",
                    buckets=(.01, .025, .05, .1, .2, .35, .5, 1, 2, 5))
CONFIDENCE = Histogram("vit_prediction_confidence", "Top-1 softmax confidence",
                       buckets=(.2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0))
PRED_CLASS = Counter("vit_predictions_total", "Predictions per class", ["label"])
LABELED = Counter("vit_labeled_predictions_total", "Predictions with ground truth", ["correct"])
ROLLING_ACC = Gauge("vit_rolling_accuracy", "Accuracy over the recent labelled window")
DRIFT_PSI = Gauge("vit_drift_psi", "PSI of live data vs training baseline", ["feature"])
DRIFT_MAX = Gauge("vit_drift_psi_max", "Worst PSI across all monitored signals")
INFLIGHT = Gauge("vit_inflight_requests", "Requests currently being processed")
MODEL_LOADED = Gauge("vit_model_loaded", "1 if the model is loaded")
