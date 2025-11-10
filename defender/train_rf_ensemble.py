#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defender.models.enhanced_feature_extractor import EnhancedPEFeatureExtractor


# ---------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------
def extract_features_from_directory(directory: str, label: int, max_files: int = None):
    extractor = EnhancedPEFeatureExtractor()
    features_list, labels_list = [], []

    files = list(Path(directory).iterdir())
    if max_files:
        files = files[:max_files]

    logger.info(f"Processing {len(files)} files from {directory}...")

    for i, file_path in enumerate(files):
        try:
            if not file_path.is_file():
                continue
            with open(file_path, "rb") as f:
                bytez = f.read()

            features = extractor.extract(bytez)
            numerical_features = {
                k: v for k, v in features.items() if isinstance(v, (int, float))
            }
            features_list.append(numerical_features)
            labels_list.append(label)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files...")
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue

    return features_list, labels_list


# ---------------------------------------------------------------------
# 2. Main training function
# ---------------------------------------------------------------------
def train_rf_ensemble(ember_dir: str, ember2017_dir:str, dike_dir: str,
                      challenge_goodware: str, challenge_malware: str,
                      ember_samples: int = 30000, n_forests: int = 5):
    logger.info("=" * 80)
    logger.info("TRAINING RANDOM FOREST ENSEMBLE")
    logger.info("=" * 80)

    all_features, all_labels = [], []

    # --- Challenge Dataset ---
    logger.info("### Loading Challenge Dataset ###")
    challenge_good_feat, challenge_good_labels = extract_features_from_directory(
        challenge_goodware, label=0
    )
    challenge_mal_feat, challenge_mal_labels = extract_features_from_directory(
        challenge_malware, label=1
    )
    all_features.extend(challenge_good_feat + challenge_mal_feat)
    all_labels.extend(challenge_good_labels + challenge_mal_labels)

    # --- Dike Dataset ---
    logger.info("### Loading Dike Dataset ###")
    dike_path = Path(dike_dir)
    dike_good_feat, dike_good_labels = extract_features_from_directory(
        dike_path / "benign", label=0
    )
    dike_mal_feat, dike_mal_labels = extract_features_from_directory(
        dike_path / "malware", label=1
    )
    all_features.extend(dike_good_feat + dike_mal_feat)
    all_labels.extend(dike_good_labels + dike_mal_labels)

    # --- Ember (optional) ---
    if os.path.exists(ember_dir):
        logger.info(f"### Loading Ember samples (first {ember_samples}) ###")
        json_files = list(Path(ember_dir).glob("*.jsonl"))
        loaded = 0
        for jf in json_files:
            # if loaded >= ember_samples:
            #     break
            with open(jf, "r") as f:
                for line in f:
                    # if loaded >= ember_samples:
                    #     break
                    try:
                        sample = eval(line.strip())
                        label = sample.get("label", -1)
                        if label not in [0, 1]:
                            continue
                        features = {k: v for k, v in sample.items() if isinstance(v, (int, float))}
                        all_features.append(features)
                        all_labels.append(label)
                        loaded += 1
                    except Exception:
                        continue
        logger.info(f"Loaded {loaded} Ember samples.")

    if os.path.exists(ember2017_dir):
        logger.info(f"### Loading Ember 2017 samples (first {ember_samples}) ###")
        json_files = list(Path(ember2017_dir).glob("*.jsonl"))
        loaded = 0
        for jf in json_files:
            # if loaded >= ember_samples:
            #     break
            with open(jf, "r") as f:
                for line in f:
                    # if loaded >= ember_samples:
                    #     break
                    try:
                        sample = eval(line.strip())
                        label = sample.get("label", -1)
                        if label not in [0, 1]:
                            continue
                        features = {k: v for k, v in sample.items() if isinstance(v, (int, float))}
                        all_features.append(features)
                        all_labels.append(label)
                        loaded += 1
                    except Exception:
                        continue
        logger.info(f"Loaded {loaded} Ember samples.")

    # --- Combine all ---
    df = pd.DataFrame(all_features).fillna(0).replace([np.inf, -np.inf], 0)
    y = np.array(all_labels)
    logger.info(f"Total samples: {len(df)}, Features: {len(df.columns)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # -----------------------------------------------------------------
    # Train 7 Random Forests
    # -----------------------------------------------------------------
    forests = []
    preds_proba = np.zeros(len(y_test))
    for i in range(n_forests):
        logger.info(f"\nTraining forest {i+1}/{n_forests}...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42 + i,
        )
        rf.fit(X_train, y_train)
        forests.append(rf)

        p = rf.predict_proba(X_test)[:, 1]
        preds_proba += p

    preds_proba /= n_forests
    y_pred = (preds_proba > 0.5).astype(int)

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, preds_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    # -----------------------------------------------------------------
    # Threshold Optimization
    # -----------------------------------------------------------------
    best_thresh = 0.5
    best_score = 0
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_thresh = (preds_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_thresh).ravel()
        fpr_t = fp / (fp + tn)
        fnr_t = fn / (fn + tp)
        if fpr_t < 0.01 and fnr_t < 0.10:
            score = 1 - (fpr_t + fnr_t) / 2
            if score > best_score:
                best_score = score
                best_thresh = thresh

    logger.info(f"Best Threshold: {best_thresh:.2f}")

    y_best = (preds_proba > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_best).ravel()
    acc_best = accuracy_score(y_test, y_best)
    fpr_best = fp / (fp + tn)
    fnr_best = fn / (fn + tp)

    logger.info(f"With Best Threshold: Acc={acc_best:.4f}, FPR={fpr_best:.4f}, FNR={fnr_best:.4f}")

    # -----------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------
    model_dir = os.path.join(os.path.dirname(__file__), "defender/models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "rf_ensemble_models.pkl"), "wb") as f:
        pickle.dump(forests, f)

    with open(os.path.join(model_dir, "rf_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(model_dir, "rf_features.pkl"), "wb") as f:
        pickle.dump(list(df.columns), f)

    with open(os.path.join(model_dir, "rf_threshold.pkl"), "wb") as f:
        pickle.dump(best_thresh, f)

    logger.info(f"Models saved to {model_dir}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (Random Forest Ensemble)")
    print("=" * 80)
    print(f"Accuracy: {acc_best:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"FPR: {fpr_best:.4f} (target < 0.01)")
    print(f"FNR: {fnr_best:.4f} (target < 0.10)")
    print(f"Best Threshold: {best_thresh:.2f}")
    print("=" * 80)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 5 Random Forest ensemble")
    parser.add_argument('--ember-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2018_2/ember2018')
    parser.add_argument('--ember2017-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2017_2/ember_2017_2')
    parser.add_argument('--dike-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/DikeDataset/files')
    parser.add_argument('--challenge-goodware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/goodware')
    parser.add_argument('--challenge-malware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/malware')    
    parser.add_argument("--ember-samples", type=int, default=30000)
    parser.add_argument("--n-forests", type=int, default=7)
    args = parser.parse_args()

    train_rf_ensemble(
        args.ember_dir,
        args.ember2017_dir,
        args.dike_dir,
        args.challenge_goodware,
        args.challenge_malware,
        args.ember_samples,
        args.n_forests,
    )
