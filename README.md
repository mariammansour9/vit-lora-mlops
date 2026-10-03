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
- Env vars: `MODEL_DIR` (default `models/quantized`), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (all cores).
