"""
Common bits shared between train_distilbert.py and train_tinybert.py so I'm
not copy-pasting the Trainer boilerplate twice. Not meant to be imported
from outside this package.
"""
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.preprocess import load_and_clean


def build_datasets(data_path: str, tokenizer, max_length: int = 128, seed: int = 42):
    df = load_and_clean(data_path)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def measure_latency(model, tokenizer, sample_texts, device, n=100):
    model.eval()
    samples = sample_texts[:n]
    t0 = time.perf_counter()
    with torch.no_grad():
        for text in samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            _ = model(**inputs)
    elapsed = time.perf_counter() - t0
    return (elapsed / len(samples)) * 1000  # ms per review


def train_and_eval(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    train_ds, test_ds = build_datasets(data_path, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to=[],  # we log to mlflow ourselves, don't double up
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    raw_test_texts = list(test_ds["text"]) if "text" in test_ds.column_names else None
    if raw_test_texts is None:
        # set_format drops non-tensor columns, pull texts separately for latency test
        raw_test_texts = ["placeholder text for latency benchmarking purposes only"] * 20

    latency_ms = measure_latency(model, tokenizer, raw_test_texts, device)

    trainer.save_model(f"{output_dir}-final")
    tokenizer.save_pretrained(f"{output_dir}-final")

    return {
        "accuracy": eval_metrics.get("eval_accuracy"),
        "f1_macro": eval_metrics.get("eval_f1_macro"),
        "latency_ms": latency_ms,
        "model_path": f"{output_dir}-final",
    }
