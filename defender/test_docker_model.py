#!/usr/bin/env python3

import os
import sys

# Test if model files exist
model_dir = "defender/models"
files_to_check = [
    "reliable_nn_model.pkl",
    "reliable_tfidf_vectorizer.pkl", 
    "reliable_scaler.pkl"
]

print("Checking for model files:")
for file in files_to_check:
    path = os.path.join(model_dir, file)
    exists = os.path.exists(path)
    print(f"  {path}: {'✅ EXISTS' if exists else '❌ MISSING'}")
    if exists:
        size = os.path.getsize(path)
        print(f"    Size: {size:,} bytes")

print("\nTesting model loading:")
try:
    from defender.models.reliable_nn_model import ReliableNNMalwareModel
    model_path = os.path.join(model_dir, "reliable_nn_model.pkl")
    model = ReliableNNMalwareModel(model_path=model_path, thresh=0.5)
    print(f"✅ Model created successfully")
    print(f"   Model loaded: {model.is_loaded}")
    print(f"   Model info: {model.model_info()}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

print(f"\nCurrent directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
if os.path.exists("defender"):
    print(f"defender/ contents: {os.listdir('defender')}")
    if os.path.exists("defender/models"):
        print(f"defender/models/ contents: {os.listdir('defender/models')}")
