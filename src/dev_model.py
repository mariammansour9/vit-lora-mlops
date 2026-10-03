"""Create a tiny random INT8 ViT in the same on-disk format as models/quantized.

Used by tests and by CI (so the Docker image can be built and smoke-tested without GPU
training or large downloads).   python -m src.dev_model models/quantized
"""
import sys
from pathlib import Path

import torch
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from src.quant_utils import quantize_model

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def make_tiny_model(out_dir) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=64, num_labels=10,
                    id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    cfg.save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    torch.save(quantize_model(ViTForImageClassification(cfg)).state_dict(), d / "model_int8.pt")
    return d


if __name__ == "__main__":
    print("tiny model written to", make_tiny_model(sys.argv[1] if len(sys.argv) > 1 else "models/quantized"))
