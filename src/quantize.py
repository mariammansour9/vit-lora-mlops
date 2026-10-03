"""Step 2: merge LoRA into the base ViT, apply INT8 dynamic quantization, compare fp32 vs int8."""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoImageProcessor, ViTForImageClassification

from src.quant_utils import quantize_model


def size_mb(path):
    return os.path.getsize(path) / 1e6


@torch.inference_mode()
def evaluate(model, processor, ds, img_col, label_col, bs=32):
    model.eval()
    correct, t0 = 0, time.perf_counter()
    for i in range(0, len(ds), bs):
        rows = ds[i:i + bs]
        inputs = processor([im.convert("RGB") for im in rows[img_col]], return_tensors="pt")
        preds = model(**inputs).logits.argmax(-1)
        correct += (preds == torch.tensor(rows[label_col])).sum().item()
    elapsed = time.perf_counter() - t0
    return correct / len(ds), elapsed / len(ds) * 1000  # accuracy, ms/img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="models/lora_adapter")
    p.add_argument("--merged-dir", default="models/merged_fp32")
    p.add_argument("--quant-dir", default="models/quantized")
    p.add_argument("--eval-samples", type=int, default=500, help="0 to skip evaluation")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.adapter_dir, "meta.json")))
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}

    # 1) Merge LoRA weights into the base model -> plain ViT, no PEFT dependency at serve time
    base = ViTForImageClassification.from_pretrained(
        meta["base_model"], num_labels=len(id2label), id2label=id2label, label2id=label2id)
    merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload().eval()
    merged.save_pretrained(args.merged_dir)
    fp32_bytes = size_mb(os.path.join(args.merged_dir, "model.safetensors"))

    # 2) Dynamic INT8 quantization of all Linear layers
    q_model = quantize_model(merged)
    os.makedirs(args.quant_dir, exist_ok=True)
    weights_path = os.path.join(args.quant_dir, "model_int8.pt")
    torch.save(q_model.state_dict(), weights_path)
    merged.config.save_pretrained(args.quant_dir)
    AutoImageProcessor.from_pretrained(args.adapter_dir).save_pretrained(args.quant_dir)
    q_bytes = size_mb(weights_path)

    print(f"\nFP32 merged : {fp32_bytes:8.1f} MB")
    print(f"INT8 dynamic: {q_bytes:8.1f} MB  ({fp32_bytes / q_bytes:.2f}x smaller)")

    report = {"fp32_mb": fp32_bytes, "int8_mb": q_bytes}

    # 3) Sanity check: accuracy + CPU latency, fp32 vs int8
    if args.eval_samples > 0:
        ds = load_dataset(meta["dataset"], split="test").select(range(args.eval_samples))
        img_col = "img" if "img" in ds.column_names else "image"
        label_col = "label" if "label" in ds.column_names else "labels"
        processor = AutoImageProcessor.from_pretrained(args.quant_dir)
        acc32, ms32 = evaluate(merged.cpu(), processor, ds, img_col, label_col)
        acc8, ms8 = evaluate(q_model, processor, ds, img_col, label_col)
        print(f"FP32: acc={acc32:.4f}  {ms32:.1f} ms/img (CPU)")
        print(f"INT8: acc={acc8:.4f}  {ms8:.1f} ms/img (CPU)")
        report.update(fp32_acc=acc32, int8_acc=acc8, fp32_ms=ms32, int8_ms=ms8)

    json.dump(report, open(os.path.join(args.quant_dir, "quantization_report.json"), "w"), indent=2)
    print(f"Quantized model saved to {args.quant_dir}")


if __name__ == "__main__":
    main()
