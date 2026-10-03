"""Shared helpers for building / loading the INT8 dynamically-quantized ViT."""
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification


def set_quant_engine() -> None:
    """Pick a quantization backend that exists on this CPU (x86 -> fbgemm, ARM -> qnnpack)."""
    supported = torch.backends.quantized.supported_engines
    for engine in ("fbgemm", "x86", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return


def quantize_model(model: nn.Module) -> nn.Module:
    """Post-training dynamic INT8 quantization of every nn.Linear layer (CPU)."""
    set_quant_engine()
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def load_quantized(model_dir: str, weights_name: str = "model_int8.pt") -> nn.Module:
    """Rebuild the architecture from config, quantize it, then load the INT8 state dict."""
    config = ViTConfig.from_pretrained(model_dir)
    model = ViTForImageClassification(config).eval()
    model = quantize_model(model)
    state = torch.load(f"{model_dir}/{weights_name}", map_location="cpu")
    model.load_state_dict(state)
    return model.eval()
