"""Step 1: fine-tune ViT with LoRA and keep the best adapter (by validation accuracy)."""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoImageProcessor, Trainer, TrainingArguments,
                          ViTForImageClassification, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/vit-base-patch16-224-in21k")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--image-column", default="img")
    p.add_argument("--label-column", default="label")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--train-samples", type=int, default=None, help="subset for quick runs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def subset(ds, n, seed):
    return ds if n is None else ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def main():
    args = parse_args()
    set_seed(args.seed)

    ds = load_dataset(args.dataset)
    labels = ds["train"].features[args.label_column].names
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}

    split = ds["train"].train_test_split(test_size=0.1, seed=args.seed,
                                         stratify_by_column=args.label_column)
    train_ds = subset(split["train"], args.train_samples, args.seed)
    val_ds = subset(split["test"], None if args.train_samples is None else args.train_samples // 5, args.seed)
    test_ds = ds["test"]

    processor = AutoImageProcessor.from_pretrained(args.base_model)

    def transform(batch):
        images = [im.convert("RGB") for im in batch[args.image_column]]
        out = processor(images, return_tensors="pt")
        out["labels"] = batch[args.label_column]
        return out

    for d in (train_ds, val_ds, test_ds):
        d.set_transform(transform)

    def collate(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["labels"] for e in examples]),
        }

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": float((preds == eval_pred.label_ids).mean())}

    base = ViTForImageClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],   # attention projections in HF ViT
        bias="none",
        modules_to_save=["classifier"],      # the new head must be trained + saved too
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,         # <- keeps the best LoRA weights
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=25,
        remove_unused_columns=False,         # keep raw image column for the transform
        label_names=["labels"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate,
                      compute_metrics=compute_metrics)
    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)       # best adapter (+ classifier head)
    processor.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "meta.json"), "w") as f:
        json.dump({"base_model": args.base_model, "id2label": id2label,
                   "dataset": args.dataset, "test_metrics": test_metrics}, f, indent=2)
    print(f"Best LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
