#!/usr/bin/env python3
"""
evaluate_server.py — recursively test malware/goodware samples against the running Flask model server.

Usage (run this on your host terminal, NOT inside Docker):

    python3 evaluate_server.py \
      --base "/mnt/c/Users/Philip Tran/Downloads/challenge/challenge_ds" \
      --url "http://127.0.0.1:8081/" \
      --out server_results.csv

This script walks through ALL subfolders under malware/ and goodware/, sends each file to the
model API, and computes overall and per-folder accuracy.
"""

import os
import csv
import time
import argparse
import requests
from pathlib import Path
from collections import Counter

def collect_files_recursive(base_dir):
    """Recursively gather all files under malware/ and goodware/."""
    files = []
    for label_dir, label in [("malware", 1), ("goodware", 0)]:
        full_path = Path(base_dir) / label_dir
        if not full_path.exists():
            print(f"[WARN] Missing folder: {full_path}")
            continue
        for path in full_path.rglob("*"):
            if path.is_file():
                files.append((path, label))
    return files

def evaluate(base_dir, url, out_path):
    samples = collect_files_recursive(base_dir)
    print(f"[INFO] Found {len(samples)} total files for evaluation")

    counts = Counter(TP=0, TN=0, FP=0, FN=0, errors=0)
    start = time.time()

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "ground_truth", "prediction", "error"])

        for path, gt in samples:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                r = requests.post(url, headers={"Content-Type": "application/octet-stream"}, data=data, timeout=15)
                r.raise_for_status()
                pred = int(r.json().get("result", -1))
                if gt == 1 and pred == 1: counts["TP"] += 1
                elif gt == 0 and pred == 0: counts["TN"] += 1
                elif gt == 0 and pred == 1: counts["FP"] += 1
                elif gt == 1 and pred == 0: counts["FN"] += 1
                writer.writerow([str(path), gt, pred, ""])
            except Exception as e:
                counts["errors"] += 1
                writer.writerow([str(path), gt, "", str(e)])

    total = counts["TP"] + counts["TN"] + counts["FP"] + counts["FN"]
    accuracy = ((counts["TP"] + counts["TN"]) / total) * 100 if total else 0
    precision = (counts["TP"] / (counts["TP"] + counts["FP"])) * 100 if (counts["TP"] + counts["FP"]) else 0
    recall = (counts["TP"] / (counts["TP"] + counts["FN"])) * 100 if (counts["TP"] + counts["FN"]) else 0

    print("\n=== Summary ===")
    print(f"TP={counts['TP']}  TN={counts['TN']}  FP={counts['FP']}  FN={counts['FN']}  ERR={counts['errors']}")
    print(f"Evaluated: {total}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision (Malicious=1): {precision:.2f}%")
    print(f"Recall    (Malicious=1): {recall:.2f}%")
    print(f"\nResults CSV written to: {out_path}")
    print(f"Elapsed time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Flask malware detection server.")
    parser.add_argument("--base", required=True, help="Base folder containing malware/ and goodware/ subfolders")
    parser.add_argument("--url", required=True, help="Model server URL (e.g. http://127.0.0.1:8081/)")
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    args = parser.parse_args()
    evaluate(args.base, args.url, args.out)
