#!/usr/bin/env python3
"""
train_dt_aggressive.py  (Windows-friendly defaults + pathlib throughout)

Train an aggressive Decision Tree and a small Bagging ensemble of Decision Trees
for malware detection using features from EnhancedPEFeatureExtractor.

- Trains:
    1) single DecisionTreeClassifier (aggressive: high depth, small leaf size, class_weight favoring malware)
    2) BaggingClassifier of 5 DecisionTreeClassifier (same aggressive base estimator)
- Performs threshold tuning to find a threshold that favors detection while keeping FPR reasonable
- Saves models and artifacts to defender/models/
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# allow importing EnhancedPEFeatureExtractor from project (module lives under project root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defender.models.enhanced_feature_extractor import EnhancedPEFeatureExtractor


def extract_features_from_directory(directory: Path, label: int, max_files: int = None):
    """Extract numerical features using EnhancedPEFeatureExtractor from files in directory"""
    extractor = EnhancedPEFeatureExtractor()
    features_list = []
    labels_list = []
    p = Path(directory)
    if not p.exists():
        logger.warning(f"Directory does not exist: {p}")
        return features_list, labels_list

    files = sorted([f for f in p.iterdir() if f.is_file()])
    if max_files:
        files = files[:max_files]

    logger.info(f"Processing {len(files)} files from {p} (label={label})")
    for i, fpath in enumerate(files):
        try:
            with open(fpath, "rb") as fh:
                data = fh.read()
            feats = extractor.extract(data)
            num_feats = {k: v for k, v in feats.items() if isinstance(v, (int, float))}
            features_list.append(num_feats)
            labels_list.append(label)
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(files)}")
        except Exception as e:
            logger.warning(f"Skipping {fpath}: {e}")
            continue

    return features_list, labels_list


def load_all_data(ember_dir: Path, ember2017_dir: Path, dike_dir: Path, challenge_good: Path, challenge_malware: Path, ember_samples=30000):
    all_features = []
    all_labels = []

    # challenge dataset
    cg_feat, cg_lab = extract_features_from_directory(challenge_good, label=0)
    cm_feat, cm_lab = extract_features_from_directory(challenge_malware, label=1)
    all_features.extend(cg_feat)
    all_features.extend(cm_feat)
    all_labels.extend(cg_lab)
    all_labels.extend(cm_lab)
    logger.info(f"Challenge: {len(cg_lab)} good, {len(cm_lab)} mal")

    # dike dataset
    dike_p = Path(dike_dir)
    dg_feat, dg_lab = extract_features_from_directory(dike_p / "benign", label=0)
    dm_feat, dm_lab = extract_features_from_directory(dike_p / "malware", label=1)
    all_features.extend(dg_feat)
    all_features.extend(dm_feat)
    all_labels.extend(dg_lab)
    all_labels.extend(dm_lab)
    logger.info(f"Dike: {len(dg_lab)} good, {len(dm_lab)} mal")

    # Ember (if present): try to load jsonl features (numeric fields only)
    ember_path = Path(ember_dir)
    if ember_path.exists():
        logger.info("Loading Ember-style JSONL features if present (numeric fields only).")
        files = sorted(ember_path.glob("*.jsonl"))
        loaded = 0
        for jf in files:
            # if loaded >= ember_samples:
            #     break
            try:
                with open(jf, "r") as fh:
                    for line in fh:
                        # if loaded >= ember_samples:
                        #     break
                        try:
                            sample = __import__("json").loads(line.strip())
                            label = sample.get("label", -1)
                            if label not in [0, 1]:
                                continue
                            # collect numeric fields from common Ember structure (flattened numeric fields)
                            features = {}
                            # generic numeric fields at top-level
                            for k, v in sample.items():
                                if isinstance(v, (int, float)):
                                    features[k] = v
                            # some nested blocks (general/header/strings/section) - best effort numeric flatten
                            if "general" in sample:
                                for k, v in sample["general"].items():
                                    if isinstance(v, (int, float)):
                                        features[f"general_{k}"] = v
                            if "section" in sample and isinstance(sample["section"], dict):
                                secs = sample["section"].get("sections", [])
                                if secs:
                                    entropies = [s.get("entropy", 0) for s in secs if isinstance(s.get("entropy", 0), (int, float))]
                                    features["section_count"] = len(secs)
                                    if entropies:
                                        features["section_avg_entropy"] = float(np.mean(entropies))
                            all_features.append(features)
                            all_labels.append(label)
                            loaded += 1
                        except Exception:
                            continue
            except Exception:
                continue
        logger.info(f"Loaded {loaded} Ember JSONL samples (numeric fields).")

    # Ember (if present): try to load jsonl features (numeric fields only)
    ember_path = Path(ember2017_dir)
    if ember_path.exists():
        logger.info("Loading Ember-style JSONL features if present (numeric fields only).")
        files = sorted(ember_path.glob("*.jsonl"))
        loaded = 0
        for jf in files:
            # if loaded >= ember_samples:
            #     break
            try:
                with open(jf, "r") as fh:
                    for line in fh:
                        # if loaded >= ember_samples:
                        #     break
                        try:
                            sample = __import__("json").loads(line.strip())
                            label = sample.get("label", -1)
                            if label not in [0, 1]:
                                continue
                            # collect numeric fields from common Ember structure (flattened numeric fields)
                            features = {}
                            # generic numeric fields at top-level
                            for k, v in sample.items():
                                if isinstance(v, (int, float)):
                                    features[k] = v
                            # some nested blocks (general/header/strings/section) - best effort numeric flatten
                            if "general" in sample:
                                for k, v in sample["general"].items():
                                    if isinstance(v, (int, float)):
                                        features[f"general_{k}"] = v
                            if "section" in sample and isinstance(sample["section"], dict):
                                secs = sample["section"].get("sections", [])
                                if secs:
                                    entropies = [s.get("entropy", 0) for s in secs if isinstance(s.get("entropy", 0), (int, float))]
                                    features["section_count"] = len(secs)
                                    if entropies:
                                        features["section_avg_entropy"] = float(np.mean(entropies))
                            all_features.append(features)
                            all_labels.append(label)
                            loaded += 1
                        except Exception:
                            continue
            except Exception:
                continue
        logger.info(f"Loaded {loaded} Ember JSONL samples (numeric fields).")

    # Build DataFrame and sanitize
    df = pd.DataFrame(all_features).fillna(0).replace([np.inf, -np.inf], 0)
    y = np.array(all_labels, dtype=int)
    logger.info(f"Total samples collected: {len(df)}; features columns: {len(df.columns)}")
    return df, y


def train_decision_tree_aggressive(
    ember_dir: Path,
    ember2017_dir: Path,
    dike_dir: Path,
    challenge_goodware: Path,
    challenge_malware: Path,
    ember_samples=30000,
    use_bagging=True,
    n_estimators=5,
):
    # Load data
    df, y = load_all_data(ember_dir, ember2017_dir, dike_dir, challenge_goodware, challenge_malware, ember_samples)

    if len(df) < 10:
        logger.error("Not enough data collected. Check paths.")
        return

    # scale features (trees don't require scaling, but we keep the scaler for consistency with other pipelines)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # -----------------------------
    # Aggressive Decision Tree base estimator
    # -----------------------------
    base_dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,            # allow deep tree
        min_samples_split=2,
        min_samples_leaf=2,        # small leaf size (more aggressive)
        class_weight={0: 1.0, 1: 6.0},  # up-weight malware
        random_state=42,
    )

    if use_bagging:
        logger.info(f"Training Bagging ensemble of {n_estimators} aggressive Decision Trees...")
        # For newer scikit-learn versions you can use 'estimator='; 'base_estimator' works across many versions
        model = BaggingClassifier(
            base_estimator=base_dt,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
    else:
        logger.info("Training single aggressive Decision Tree...")
        model = base_dt

    # Fit model
    model.fit(X_train, y_train)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)
        y_proba = preds.astype(float)

    y_pred_default = (y_proba > 0.5).astype(int)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    logger.info("Default-threshold metrics (0.5):")
    logger.info(f"Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f} (TP={tp}, FP={fp})")

    # -----------------------------
    # Threshold tuning to be more aggressive (minimize FNR under constraints)
    # -----------------------------
    best_thresh = 0.5
    best_score = -1.0
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds_t = (y_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_t).ravel()
        fpr_t = fp / (fp + tn) if (fp + tn) else 0.0
        fnr_t = fn / (fn + tp) if (fn + tp) else 0.0
        score = (1 - fnr_t) - 0.5 * fpr_t  # reward recall, moderate FPR penalty
        if score > best_score:
            best_score = score
            best_thresh = thresh

    # Evaluate at best threshold
    y_pred_best = (y_proba > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    fpr_best = fp / (fp + tn) if (fp + tn) else 0.0
    fnr_best = fn / (fn + tp) if (fn + tp) else 0.0
    acc_best = accuracy_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)

    logger.info(f"Best threshold by aggressive score: {best_thresh:.2f}")
    logger.info(f"At best threshold -> Acc={acc_best:.4f}, F1={f1_best:.4f}, FPR={fpr_best:.4f}, FNR={fnr_best:.4f}")

    # -----------------------------
    # Save artifacts (pathlib, OS-neutral)
    # -----------------------------
    model_dir = Path(__file__).resolve().parent / "defender" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = "dt_bagging5.pkl" if use_bagging else "dt_single_aggressive.pkl"
    with open(model_dir / model_name, "wb") as fh:
        pickle.dump(model, fh)

    with open(model_dir / "dt_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    with open(model_dir / "dt_features.pkl", "wb") as fh:
        pickle.dump(list(df.columns), fh)

    with open(model_dir / "dt_best_threshold.pkl", "wb") as fh:
        pickle.dump(best_thresh, fh)

    logger.info(f"Saved model and artifacts to {model_dir}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Decision Tree - Aggressive)")
    print("=" * 80)
    print(f"Model: {'Bagging(5xDT)' if use_bagging else 'Single DecisionTree (aggressive)'}")
    print(f"Accuracy (best threshold): {acc_best:.4f}")
    print(f"F1 (best threshold): {f1_best:.4f}")
    print(f"AUC (probabilities): {auc:.4f}")
    print(f"FPR (best threshold): {fpr_best:.4f}")
    print(f"FNR (best threshold): {fnr_best:.4f}")
    print(f"Best threshold: {best_thresh:.2f}")
    print("=" * 80)

    return {
        "model_path": str(model_dir / model_name),
        "best_threshold": best_thresh,
        "acc_best": acc_best,
        "f1_best": f1_best,
        "auc": auc,
        "fpr_best": fpr_best,
        "fnr_best": fnr_best,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggressive Decision Tree / Bagging training")

    # Windows-native default paths (edit here if your folders differ).
    parser.add_argument('--ember-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2018_2/ember2018')
    parser.add_argument('--ember2017-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2017_2/ember_2017_2')
    parser.add_argument('--dike-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/DikeDataset/files')
    parser.add_argument('--challenge-goodware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/goodware')
    parser.add_argument('--challenge-malware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/malware')    
    parser.add_argument("--ember-samples", type=int, default=30000)
    parser.add_argument("--use-bagging", action="store_true", help="Train Bagging ensemble of 5 DTs (recommended)")
    parser.add_argument("--n-estimators", type=int, default=5, help="Number of trees in Bagging (if used)")
    args = parser.parse_args()

    train_decision_tree_aggressive(
        ember_dir=args.ember_dir,
        ember2017_dir=args.ember2017_dir,
        dike_dir=args.dike_dir,
        challenge_goodware=args.challenge_goodware,
        challenge_malware=args.challenge_malware,
        ember_samples=args.ember_samples,
        use_bagging=args.use_bagging,
        n_estimators=args.n_estimators,
    )
