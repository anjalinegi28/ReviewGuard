"""
Exports a fine-tuned checkpoint to TFLite so it can plausibly be pitched as
"runs client-side" (browser extension, mobile). Goes through ONNX as the
intermediate step since that's the path that actually works reliably for
HF transformer checkpoints - direct PyTorch -> TFLite isn't a thing.

Path: PyTorch checkpoint -> ONNX -> TensorFlow SavedModel -> TFLite
"""
import argparse
import os
import subprocess

import onnx
import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_to_onnx(checkpoint_dir: str, onnx_path: str, max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    dummy = tokenizer("placeholder review text for tracing the graph", return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    onnx.checker.check_model(onnx_path)
    print(f"exported ONNX graph to {onnx_path}")


def onnx_to_tflite(onnx_path: str, saved_model_dir: str, tflite_path: str):
    # tf2onnx doesn't do ONNX->TF directly; use onnx-tf if available, else
    # shell out to the tf2onnx convert utility in reverse mode isn't a thing,
    # so this step assumes `onnx-tf` is installed for the SavedModel hop.
    subprocess.run(
        ["onnx-tf", "convert", "-i", onnx_path, "-o", saved_model_dir],
        check=True,
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quantization
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote quantized TFLite model to {tflite_path}")
    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"final size: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a fine-tuned HF checkpoint dir")
    parser.add_argument("--out", default="artifacts/models/reviewguard.tflite")
    parser.add_argument("--work-dir", default="artifacts/models/_tflite_work")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    onnx_path = os.path.join(args.work_dir, "model.onnx")
    saved_model_dir = os.path.join(args.work_dir, "saved_model")

    export_to_onnx(args.checkpoint, onnx_path)
    onnx_to_tflite(onnx_path, saved_model_dir, args.out)


if __name__ == "__main__":
    main()
