# ViT + LoRA + INT8 quantization + FastAPI + Docker

```
train.py  ->  models/lora_adapter   (best LoRA adapter + head)
quantize.py -> models/quantized     (merged, INT8 weights, config, processor)
Dockerfile + app.py                 (FastAPI serving the INT8 model on CPU)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
```

## 2. Fine-tune with LoRA (best checkpoint is kept automatically)
```bash
# full run (GPU recommended)
python -m src.train --epochs 3

# quick smoke test on CPU
python -m src.train --epochs 1 --train-samples 2000 --batch-size 16
```
Output: `models/lora_adapter/` (adapter weights, processor, `meta.json` with test accuracy).

## 3. Merge + quantize
```bash
python -m src.quantize --eval-samples 500
```
Prints FP32 vs INT8 size, accuracy and CPU latency and writes `models/quantized/`.

## 4. (Optional) run the API without Docker
```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn src.app:app --port 8000
```

## 5. Build and run the container
```bash
docker build -t vit-lora-int8 .
docker run --rm -p 8000:8000 vit-lora-int8
```
Docs: http://localhost:8000/docs

## 6. Run inference through the container
```bash
# grab a sample image
python -c "from datasets import load_dataset as l; l('cifar10',split='test')[0]['img'].resize((224,224)).save('sample.png')"

curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict?top_k=3" -F "file=@sample.png"
# or
pip install requests && python client.py sample.png
```
Example response:
```json
{"prediction": "cat", "confidence": 0.97,
 "top_k": [{"label": "cat", "confidence": 0.97}, ...], "latency_ms": 85.3}
```

## Notes
- Quantization is PyTorch **dynamic INT8** on all `nn.Linear` layers (~4x smaller, CPU only).
  Use the same `torch` version for quantizing and serving.
- Merging LoRA before quantizing means the container needs no `peft` dependency.
- Env vars: `MODEL_DIR` (default `models/quantized`), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (all cores), `MONITOR_LOG_DIR` (`logs`), `BASELINE_PATH` (`$MODEL_DIR/reference_stats.json`).

## 7. Tests
```bash
pip install -r requirements-serve.txt -r requirements-dev.txt
pytest -q          # monitoring maths, API (tiny random ViT), alert-rule/compose sanity
```

## 8. CI/CD (GitHub Actions - `.github/workflows/ci.yml`)
| Trigger | Jobs |
|---|---|
| push `feature/**`, `bugfix/**`, PR into `develop`/`main` | `test` |
| push `develop` | `test` -> `build` (image + smoke test, push to GHCR) -> `deploy-staging` |
| push `main` or tag `v*` | `test` -> `build` -> `deploy-production` (needs approval via the `production` environment) |
| nightly cron | `monitoring-check` (drift / accuracy gate) |

Branching model: `feature/*` -> PR -> `develop` -> release merge -> `main` (tagged `vX.Y.Z`).
See `docs/GIT_WORKFLOW.md` for the full git session (add/commit/pull/push/merge/conflict/diff/status).

## 9. Observability (AI + infrastructure)
```bash
docker compose up --build      # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
```
**Model / data monitoring** (`src/monitoring.py`, exposed at `/metrics`, `/monitoring/drift`, `/feedback`)
- Every prediction is logged (`logs/predictions.jsonl`) with the predicted class, confidence, latency and input statistics (brightness, contrast, mean R/G/B).
- **Data drift**: PSI (and KS) of live inputs, confidence and predicted-class mix vs. a baseline (`reference_stats.json`). PSI < 0.1 ok, 0.1-0.25 warn, > 0.25 alert.
- **Performance**: ground truth arrives later via `POST /feedback {"request_id", "true_label"}`; the service tracks rolling accuracy, and `monitor_cli report` computes accuracy, macro-F1, per-class F1, label coverage and calibration error (ECE).
- Without labels, drift + confidence + class distribution act as proxies for model health.

```bash
python -m src.monitor_cli baseline --model-dir models/quantized --samples 1000 -o models/quantized/reference_stats.json
python -m src.monitor_cli report   --baseline models/quantized/reference_stats.json --logs logs   # exit 1 on alert
python -m src.monitor_cli simulate --out logs --n 500 --drift                                     # demo traffic
```
**Infrastructure monitoring**: node-exporter (host CPU/memory/disk), cAdvisor (per-container), process metrics from the API, Prometheus alert rules (`monitoring/alerts.yml`: ApiDown, HighLatencyP95, HighCPU, HighMemory, DiskAlmostFull, DataDriftHigh, ModelAccuracyLow, LowConfidence) and a provisioned Grafana dashboard.
