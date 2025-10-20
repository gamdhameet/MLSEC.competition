#!/usr/bin/env python3
"""
train_dropper_aware.py

Train a dropper-aware BERT classifier (3 classes: benign, dropper, payload) using
synthetic PE-like samples. This is for demonstration / research only — it does NOT
execute any binaries or extracted payloads. All binary content is represented as
bounded textual features (short hex snippets, import lists, embedded strings).
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# ML imports (install with: pip install transformers datasets torch)
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset, ClassLabel, load_metric
except Exception as exc:
    raise RuntimeError(
        "Missing ML dependencies. Install with: pip install transformers datasets torch"
    ) from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# Config / defaults
DEFAULT_MODEL = "distilbert-base-uncased"
MAX_HEX_CHARS = 512        # how many hex chars to include from the binary (bounded)
RANDOM_SEED = 42
LABELS = {0: "benign", 1: "dropper", 2: "payload"}



# Synthetic data generator (We can add more synthetic samples later)
def create_synthetic_pe_data(kind: str = "benign") -> Dict:
    """
    Create a synthetic PE-like sample. 'kind' in {"benign","dropper","payload"}.
    Returns a dict with textual/meta features and bounded hex snippet.
    """
    import string

    benign_libs = ["kernel32.dll", "user32.dll", "ntdll.dll", "advapi32.dll", "msvcrt.dll"]
    benign_funcs = ["GetProcAddress", "LoadLibrary", "CreateFile", "ReadFile", "WriteFile"]

    dropper_libs = ["urlmon.dll", "wininet.dll", "ws2_32.dll"]
    dropper_funcs = ["URLDownloadToFileA", "InternetOpenA", "InternetReadFile", "CreateProcessA"]

    payload_libs = ["kernel32.dll", "advapi32.dll"]
    payload_funcs = ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread", "SetThreadContext"]

    # strings to embed (benign vs suspicious)
    benign_strings = ["Microsoft Corporation", "Program Files", "C:\\Windows\\System32", "Version"]
    dropper_strings = ["http://", "https://", "download", "payload.exe", "temp"]
    payload_strings = ["shellcode", "inject", "rwx", "privilege", "dll"]

    if kind == "benign":
        libraries = random.sample(benign_libs, k=2)
        functions = random.sample(benign_funcs, k=3)
        strings = benign_strings + [
            "".join(random.choices(string.ascii_letters, k=random.randint(4, 10)))
            for _ in range(2)
        ]
    elif kind == "dropper":
        libraries = random.sample(benign_libs + dropper_libs, k=3)
        functions = random.sample(benign_funcs + dropper_funcs, k=5)
        strings = dropper_strings + [
            "http://example.com/payload", "download", "save_as:payload.exe"
        ]
    elif kind == "payload":
        libraries = random.sample(benign_libs + payload_libs, k=3)
        functions = random.sample(benign_funcs + payload_funcs, k=5)
        strings = payload_strings + ["inject", "CreateRemoteThread", "shellcode"]
    else:
        raise ValueError("kind must be 'benign'|'dropper'|'payload'")

    # create pseudo-binary (but we won't execute it). We'll store only bounded hex snippet.
    blob = bytearray(b"MZ\x90\x00")
    for _ in range(random.randint(500, 2500)):
        blob.append(random.randint(0, 255))
    # embed some strings
    for s in random.sample(strings, k=min(3, len(strings))):
        sb = s.encode("utf-8") + b"\x00"
        pos = random.randint(32, max(64, len(blob) - len(sb)))
        blob[pos:pos] = sb

    # bounded hex snippet for tokenization (don't include whole binary)
    hex_snippet = blob.hex()[:MAX_HEX_CHARS]

    return {
        "libraries": libraries,
        "functions": functions,
        "strings": strings,
        "size": len(blob),
        "hex_snippet": hex_snippet,
    }


def create_synthetic_dataset(output_path: str, n_samples: int = 1000, seed: int = RANDOM_SEED) -> str:
    """
    Create a synthetic dataset with 3 classes and save as JSON.
    Returns path to saved dataset.
    """
    random.seed(seed)
    samples = []
    kinds = ["benign", "dropper", "payload"]
    # generate roughly balanced classes
    for i in range(n_samples):
        kind = kinds[i % 3]  # simple balanced cycling
        sample = {
            "label": 0 if kind == "benign" else 1 if kind == "dropper" else 2,
            "data": create_synthetic_pe_data(kind),
        }
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f)
    logger.info(f"Synthetic dataset written to {output_path} ({n_samples} samples)")
    return output_path



# Convert sample -> text representation
def sample_to_text(sample: Dict) -> str:
    """
    Convert the structured sample to a single text string to feed into BERT.
    Keep everything bounded and textual; do NOT attempt to reassemble/extract raw payloads.
    """
    d = sample["data"]
    # join features into a short string: libraries, functions, top strings, and short hex snippet
    libs = " ".join([f"L:{x}" for x in d.get("libraries", [])])
    funcs = " ".join([f"F:{x}" for x in d.get("functions", [])])
    strs = " ".join([f"S:{x}" for x in d.get("strings", [])][:10])
    hex_snip = d.get("hex_snippet", "")
    # only keep short groups of hex to keep tokenizer steps manageable
    hex_words = " ".join([hex_snip[i:i+32] for i in range(0, min(len(hex_snip), 256), 32)])
    txt = f"{libs} || {funcs} || {strs} || HEX: {hex_words}"
    return txt

# Training utilities
def load_json_dataset(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    texts = [sample_to_text(s) for s in data]
    labels = [s["label"] for s in data]
    return texts, labels


def prepare_hf_dataset(texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
    ds = Dataset.from_dict({"text": texts, "label": labels})
    # map to tokens
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds



# Main train function (Where our training happens)
def train_dropper_aware(
    dataset_path: str,
    model_name_or_path: str = DEFAULT_MODEL,
    out_dir: str = "dropper_model",
    epochs: int = 2,
    batch_size: int = 16,
    seed: int = RANDOM_SEED,
):
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=3
    )

    logger.info("Loading dataset...")
    texts, labels = load_json_dataset(dataset_path)
    # train/test split
    split = int(0.9 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    val_texts, val_labels = texts[split:], labels[split:]

    train_ds = prepare_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = prepare_hf_dataset(val_texts, val_labels, tokenizer)

    now = int(time.time())
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )

    # simple accuracy metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels_eval)

    logger.info("Creating Trainer and starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    logger.info("Training finished. Saving model...")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"Model saved to {out_dir}")
    return out_dir


# -------------------------
# Simple demonstration of model usage (safe)
# -------------------------
def demo_prediction(model_dir: str, text_tokenizer=None):
    """
    Load saved model and tokenizer and run a safe prediction on a synthetic sample.
    No execution of binary content occurs.
    """
    tokenizer = text_tokenizer or AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # create a sample dropper-like text and predict
    sample = {
        "data": create_synthetic_pe_data("dropper")
    }
    txt = sample_to_text({"data": sample["data"]})
    inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    logger.info(f"Demo sample predicted as: {pred} -> {LABELS[pred]}")
    return pred


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a dropper-aware BERT classifier (3 classes).")
    parser.add_argument("--out-dir", default="dropper_model", help="Directory to save trained model")
    parser.add_argument("--samples", type=int, default=1200, help="Number of synthetic samples to generate")
    parser.add_argument("--data-path", default="data/synthetic_dropper.json", help="Path to save synthetic dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    args = parser.parse_args()

    # create synthetic dataset
    ds_path = create_synthetic_dataset(args.data_path, n_samples=args.samples)
    model_dir = train_dropper_aware(ds_path, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size)
    demo_prediction(model_dir)


if __name__ == "__main__":
    main()
