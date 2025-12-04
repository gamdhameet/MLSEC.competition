#!/usr/bin/env python3


import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from defender.models.enhanced_feature_extractor import EnhancedPEFeatureExtractor


def extract_features_from_directory(directory: str, label: int, max_files: int = None):
    """Extract features from all PE files in a directory"""
    from pathlib import Path
    
    extractor = EnhancedPEFeatureExtractor()
    features_list = []
    labels_list = []
    
    files = list(Path(directory).iterdir())
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Processing {len(files)} files from {directory}...")
    
    for i, file_path in enumerate(files):
        try:
            if not file_path.is_file():
                continue
            
            with open(file_path, 'rb') as f:
                bytez = f.read()
            
            features = extractor.extract(bytez)
            numerical_features = {k: v for k, v in features.items() 
                                if isinstance(v, (int, float))}
            
            features_list.append(numerical_features)
            labels_list.append(label)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files...")
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue
    
    return features_list, labels_list

from pathlib import Path
import json

def load_ember2024_samples_json(ember2024_dir: str, max_samples: int | None = None):
    """
    Stream EMBER2024 JSONL feature files from ember2024_dir, similar to load_ember_samples,
    but without using thrember.read_vectorized_features (avoids 40+ GB RAM).
    """
    logger.info("Loading EMBER2024 raw JSONL features (streaming)...")
    ember_path = Path(ember2024_dir)

    # Grab ALL .jsonl files (train/test/challenge/PE types) and let max_samples cap it.
    jsonl_files = sorted(ember_path.rglob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found under {ember2024_dir}")
        return [], []

    all_features = []
    all_labels = []
    loaded = 0

    logger.info(f"Found {len(jsonl_files)} JSONL files under {ember2024_dir}")

    for feature_file in jsonl_files:
        logger.info(f"Processing EMBER2024 file: {feature_file}")
        with feature_file.open("r") as f:
            for line in f:
                if max_samples is not None and loaded >= max_samples:
                    logger.info(f"Reached max_samples={max_samples} for EMBER2024")
                    logger.info(f"Total EMBER2024 samples loaded: {loaded}")
                    return all_features, all_labels

                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    label = sample.get("label", -1)
                    if label not in [0, 1]:
                        continue

                    features = {}

                    # general block
                    general = sample.get("general", {})
                    for k, v in general.items():
                        if isinstance(v, (int, float)):
                            features[f"ember3_general_{k}"] = v

                    # header info (PE-like, may or may not exist depending on file type)
                    header = sample.get("header", {})
                    coff = header.get("coff", {})
                    opt = header.get("optional", {})

                    if isinstance(coff, dict):
                        ts = coff.get("timestamp")
                        if isinstance(ts, (int, float)):
                            features["ember3_header_timestamp"] = ts

                    if isinstance(opt, dict):
                        for key in ("sizeof_code", "sizeof_headers", "sizeof_image"):
                            val = opt.get(key)
                            if isinstance(val, (int, float)):
                                features[f"ember3_header_{key}"] = val

                    # section stats
                    section = sample.get("section", {})
                    sections = section.get("sections", [])
                    if isinstance(sections, list) and sections:
                        entropies = []
                        raw_sizes = []
                        for s in sections:
                            if not isinstance(s, dict):
                                continue
                            ent = s.get("entropy")
                            size_raw = s.get("size_raw")
                            if isinstance(ent, (int, float)):
                                entropies.append(ent)
                            if isinstance(size_raw, (int, float)):
                                raw_sizes.append(size_raw)

                        features["ember3_section_count"] = len(sections)
                        if entropies:
                            features["ember3_section_avg_entropy"] = float(np.mean(entropies))
                            features["ember3_section_max_entropy"] = float(np.max(entropies))
                        if raw_sizes:
                            features["ember3_section_avg_size_raw"] = float(np.mean(raw_sizes))

                    # strings
                    strings = sample.get("strings", {})
                    if isinstance(strings, dict):
                        for key in ("numstrings", "entropy", "paths", "urls",
                                    "registry", "suspicious_imports", "suspicious_strings"):
                            val = strings.get(key)
                            if isinstance(val, (int, float)):
                                features[f"ember3_strings_{key}"] = val

                    # imports/exports
                    imports = sample.get("imports", {})
                    if isinstance(imports, dict):
                        features["ember3_import_dll_count"] = len(imports)
                        features["ember3_import_func_count"] = sum(
                            len(funcs) for funcs in imports.values()
                            if isinstance(funcs, list)
                        )

                    exports = sample.get("exports", [])
                    if isinstance(exports, list):
                        features["ember3_export_func_count"] = len(exports)

                    # histogram & byteentropy (same idea as your Ember 2018 loader)
                    hist = sample.get("histogram")
                    if isinstance(hist, list) and hist:
                        features["ember3_hist_mean"] = float(np.mean(hist))
                        features["ember3_hist_std"] = float(np.std(hist))

                    be = sample.get("byteentropy")
                    if isinstance(be, list) and be:
                        features["ember3_byteentropy_mean"] = float(np.mean(be))

                    all_features.append(features)
                    all_labels.append(label)
                    loaded += 1

                    if loaded % 5000 == 0:
                        logger.info(f"Loaded {loaded} EMBER2024 samples so far...")

                except Exception as e:
                    logger.warning(f"Error parsing EMBER2024 line in {feature_file}: {e}")
                    continue

    logger.info(f"Total EMBER2024 samples loaded: {loaded}")
    return all_features, all_labels

def iter_ember2024_samples_json(ember2024_dir: str, chunk_size: int = 50000):
    """
    Generator that yields (features_list, labels_list) chunks from EMBER2024 JSONL files.
    Each chunk has at most `chunk_size` samples.
    """
    logger.info("Streaming EMBER2024 JSONL features in chunks...")
    ember_path = Path(ember2024_dir)
    jsonl_files = sorted(ember_path.rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No .jsonl files found under {ember2024_dir}")
        return

    logger.info(f"Found {len(jsonl_files)} JSONL files under {ember2024_dir}")

    features_chunk = []
    labels_chunk = []
    loaded = 0

    for feature_file in jsonl_files:
        logger.info(f"Processing EMBER2024 file: {feature_file}")
        with feature_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    label = sample.get("label", -1)
                    if label not in [0, 1]:
                        continue

                    features = {}

                    # general block
                    general = sample.get("general", {})
                    for k, v in general.items():
                        if isinstance(v, (int, float)):
                            features[f"ember3_general_{k}"] = v

                    # header info
                    header = sample.get("header", {})
                    coff = header.get("coff", {})
                    opt = header.get("optional", {})

                    if isinstance(coff, dict):
                        ts = coff.get("timestamp")
                        if isinstance(ts, (int, float)):
                            features["ember3_header_timestamp"] = ts

                    if isinstance(opt, dict):
                        for key in ("sizeof_code", "sizeof_headers", "sizeof_image"):
                            val = opt.get(key)
                            if isinstance(val, (int, float)):
                                features[f"ember3_header_{key}"] = val

                    # section stats
                    section = sample.get("section", {})
                    sections = section.get("sections", [])
                    if isinstance(sections, list) and sections:
                        entropies = []
                        raw_sizes = []
                        for s in sections:
                            if not isinstance(s, dict):
                                continue
                            ent = s.get("entropy")
                            size_raw = s.get("size_raw")
                            if isinstance(ent, (int, float)):
                                entropies.append(ent)
                            if isinstance(size_raw, (int, float)):
                                raw_sizes.append(size_raw)

                        features["ember3_section_count"] = len(sections)
                        if entropies:
                            features["ember3_section_avg_entropy"] = float(np.mean(entropies))
                            features["ember3_section_max_entropy"] = float(np.max(entropies))
                        if raw_sizes:
                            features["ember3_section_avg_size_raw"] = float(np.mean(raw_sizes))

                    # strings
                    strings = sample.get("strings", {})
                    if isinstance(strings, dict):
                        for key in ("numstrings", "entropy", "paths", "urls",
                                    "registry", "suspicious_imports", "suspicious_strings"):
                            val = strings.get(key)
                            if isinstance(val, (int, float)):
                                features[f"ember3_strings_{key}"] = val

                    # imports/exports
                    imports = sample.get("imports", {})
                    if isinstance(imports, dict):
                        features["ember3_import_dll_count"] = len(imports)
                        features["ember3_import_func_count"] = sum(
                            len(funcs) for funcs in imports.values()
                            if isinstance(funcs, list)
                        )

                    exports = sample.get("exports", [])
                    if isinstance(exports, list):
                        features["ember3_export_func_count"] = len(exports)

                    # histogram & byteentropy
                    hist = sample.get("histogram")
                    if isinstance(hist, list) and hist:
                        features["ember3_hist_mean"] = float(np.mean(hist))
                        features["ember3_hist_std"] = float(np.std(hist))

                    be = sample.get("byteentropy")
                    if isinstance(be, list) and be:
                        features["ember3_byteentropy_mean"] = float(np.mean(be))

                    features_chunk.append(features)
                    labels_chunk.append(label)
                    loaded += 1

                    if len(features_chunk) >= chunk_size:
                        logger.info(f"Yielding chunk with {len(features_chunk)} EMBER2024 samples (total so far: {loaded})")
                        yield features_chunk, labels_chunk
                        features_chunk, labels_chunk = [], []

                except Exception as e:
                    logger.warning(f"Error parsing EMBER2024 line in {feature_file}: {e}")
                    continue

    if features_chunk:
        logger.info(f"Yielding final chunk with {len(features_chunk)} EMBER2024 samples (total: {loaded})")
        yield features_chunk, labels_chunk

def to_aligned_df(features_list, all_columns=None):
    """
    Convert list-of-dicts to DataFrame and align columns.
    If all_columns is None, infer and return them.
    Otherwise, add any missing columns as 0 and reorder.
    """
    df = pd.DataFrame(features_list)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    if all_columns is None:
        all_columns = sorted(df.columns)
        df = df[all_columns]
        return df, all_columns

    # add missing cols as 0
    for col in all_columns:
        if col not in df:
            df[col] = 0

    # drop any extra cols (shouldn’t normally happen, but for safety)
    df = df[all_columns]
    return df, all_columns

def load_ember_samples(ember_dir: str, max_samples: int):
    """Load Ember dataset samples"""
    from pathlib import Path
    import json
    
    logger.info("Loading Ember dataset...")
    ember_path = Path(ember_dir)
    
    feature_files = []
    for i in range(6):
        feature_file = ember_path / f"train_features_{i}.jsonl"
        if feature_file.exists():
            feature_files.append(feature_file)
    
    all_features = []
    all_labels = []
    loaded = 0
    
    for feature_file in feature_files:
        # if loaded >= max_samples:
        #     break
        
        logger.info(f"Processing {feature_file.name}...")
        
        with open(feature_file, 'r') as f:
            for line in f:
                # if loaded >= max_samples:
                #     break
                
                try:
                    sample = json.loads(line.strip())
                    label = sample.get('label', -1)
                    
                    if label not in [0, 1]:
                        continue
                    
                    features = {}
                    
                    if 'general' in sample:
                        for k, v in sample['general'].items():
                            if isinstance(v, (int, float)):
                                features[f'general_{k}'] = v
                    
                    if 'header' in sample:
                        if 'coff' in sample['header']:
                            features['header_timestamp'] = sample['header']['coff'].get('timestamp', 0)
                        if 'optional' in sample['header']:
                            opt = sample['header']['optional']
                            features['header_sizeof_code'] = opt.get('sizeof_code', 0)
                            features['header_sizeof_headers'] = opt.get('sizeof_headers', 0)
                    
                    if 'section' in sample and 'sections' in sample['section']:
                        sections = sample['section']['sections']
                        if sections:
                            entropies = [s.get('entropy', 0) for s in sections]
                            features['section_count'] = len(sections)
                            features['section_avg_entropy'] = np.mean(entropies)
                            features['section_max_entropy'] = np.max(entropies)
                    
                    if 'strings' in sample:
                        strings = sample['strings']
                        features['string_numstrings'] = strings.get('numstrings', 0)
                        features['string_entropy'] = strings.get('entropy', 0)
                        features['string_path_count'] = strings.get('paths', 0)
                        features['string_url_count'] = strings.get('urls', 0)
                        features['string_registry_count'] = strings.get('registry', 0)
                    
                    if 'imports' in sample and isinstance(sample['imports'], dict):
                        features['import_dll_count'] = len(sample['imports'])
                        features['import_func_count'] = sum(len(funcs) for funcs in sample['imports'].values())
                    
                    if 'exports' in sample and isinstance(sample['exports'], list):
                        features['export_func_count'] = len(sample['exports'])
                    
                    if 'histogram' in sample and sample['histogram']:
                        features['histogram_mean'] = np.mean(sample['histogram'])
                        features['histogram_std'] = np.std(sample['histogram'])
                    
                    if 'byteentropy' in sample and sample['byteentropy']:
                        features['entropy_overall'] = np.mean(sample['byteentropy'])
                    
                    all_features.append(features)
                    all_labels.append(label)
                    loaded += 1
                    
                    if loaded % 5000 == 0:
                        logger.info(f"Loaded {loaded} Ember samples...")
                
                except:
                    continue
    
    logger.info(f"Total Ember samples: {loaded}")
    return all_features, all_labels


def train_with_challenge(ember_dir: str, 
                         ember2017_dir: str, 
                         dike_dir: str, 
                         challenge_goodware: str, 
                         challenge_malware: str, 
                         ember_samples: int = 30000,
                         ember2024_dir: str | None = None,
                         ember2024_samples: int | None = None):
    
    logger.info("=" * 80)
    logger.info("TRAINING WITH CHALLENGE DATASET INCLUDED")
    logger.info("=" * 80)
    
    # ------------------------------------------------------------------
    # 1) LOAD CHALLENGE + DIKE + EMBER2018 + EMBER2017 INTO BASE SET
    # ------------------------------------------------------------------
    all_features = []
    all_labels = []

    logger.info("\n### Loading Challenge Dataset ###")
    challenge_good_feat, challenge_good_labels = extract_features_from_directory(
        challenge_goodware, label=0
    )
    challenge_mal_feat, challenge_mal_labels = extract_features_from_directory(
        challenge_malware, label=1
    )
    
    all_features.extend(challenge_good_feat)
    all_features.extend(challenge_mal_feat)
    all_labels.extend(challenge_good_labels)
    all_labels.extend(challenge_mal_labels)
    
    logger.info(f"Challenge dataset: {len(challenge_good_labels)} goodware, {len(challenge_mal_labels)} malware")
    
    logger.info("\n### Loading DikeDataset ###")
    from pathlib import Path
    dike_path = Path(dike_dir)
    
    dike_good_feat, dike_good_labels = extract_features_from_directory(
        dike_path / 'benign', label=0
    )
    dike_mal_feat, dike_mal_labels = extract_features_from_directory(
        dike_path / 'malware', label=1
    )
    
    all_features.extend(dike_good_feat)
    all_features.extend(dike_mal_feat)
    all_labels.extend(dike_good_labels)
    all_labels.extend(dike_mal_labels)
    
    logger.info(f"Dike dataset: {len(dike_good_labels)} goodware, {len(dike_mal_labels)} malware")
    
    logger.info(f"\n### Loading {ember_samples} Ember2018 samples ###")
    ember_feat, ember_labels = load_ember_samples(ember_dir, ember_samples)
    all_features.extend(ember_feat)
    all_labels.extend(ember_labels)
    logger.info(f"Ember2018 dataset: {len(ember_labels)} samples")

    logger.info(f"\n### Loading {ember_samples} Ember2017 samples ###")
    ember_feat, ember_labels = load_ember_samples(ember2017_dir, ember_samples)
    all_features.extend(ember_feat)
    all_labels.extend(ember_labels)
    logger.info(f"Ember2017 dataset: {len(ember_labels)} samples")
    
    # ------------------------------------------------------------------
    # 2) BUILD BASE DATAFRAME + TRAIN/TEST SPLIT
    # ------------------------------------------------------------------
    logger.info("\n### Preparing Base Features ###")
    df_base = pd.DataFrame(all_features).replace([np.inf, -np.inf], 0).fillna(0)
    y_base = np.array(all_labels)
    
    logger.info(f"Base samples: {len(df_base)}")
    logger.info(f"Base features: {len(df_base.columns)}")
    logger.info(f"Base label distribution: Benign={np.sum(y_base == 0)}, Malicious={np.sum(y_base == 1)}")
    
    X_train_base, X_test, y_train_base, y_test = train_test_split(
        df_base, y_base,
        test_size=0.15,
        random_state=42,
        stratify=y_base
    )

    logger.info(f"\nBase training set: {X_train_base.shape[0]} samples")
    logger.info(f"Test set (fixed):   {X_test.shape[0]} samples")
    
    # ------------------------------------------------------------------
    # 3) INITIAL BOOSTER TRAINING ON BASE DATA
    # ------------------------------------------------------------------
    logger.info("\n### Training LightGBM Booster (base-only) ###")

    scale_pos_weight = np.sum(y_train_base == 0) / np.sum(y_train_base == 1)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 40,
        'max_depth': 12,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    # Align feature space and create LightGBM datasets
    df_base_train, all_columns = to_aligned_df(X_train_base.to_dict(orient="records"))
    df_base_test, _ = to_aligned_df(X_test.to_dict(orient="records"), all_columns=all_columns)

    # 🔴 IMPORTANT: keep raw data so we can reuse valid_data in later training calls
    train_data = lgb.Dataset(df_base_train, label=y_train_base, free_raw_data=False)
    valid_data = lgb.Dataset(df_base_test, label=y_test, reference=train_data, free_raw_data=False)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # ------------------------------------------------------------------
    # 4) STREAM EMBER2024 CHUNKS AND CONTINUE TRAINING
    # ------------------------------------------------------------------
    if ember2024_dir is not None:
        chunk_size = ember2024_samples if ember2024_samples is not None else 50000
        logger.info(f"\n### Streaming EMBER2024 chunks (chunk_size={chunk_size}) ###")
        
        for feat_chunk, label_chunk in iter_ember2024_samples_json(ember2024_dir, chunk_size=chunk_size):
            logger.info(f"Streaming EMBER2024 chunk: {len(label_chunk)} samples")

            df_chunk, _ = to_aligned_df(feat_chunk, all_columns=all_columns)

            # 🔴 IMPORTANT: also keep raw data here
            train_chunk = lgb.Dataset(df_chunk, label=np.array(label_chunk), free_raw_data=False)

            model = lgb.train(
                params,
                train_chunk,
                num_boost_round=200,
                init_model=model,
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50)
                ]
            )
    
    # ------------------------------------------------------------------
    # 5) EVALUATION ON FIXED TEST SET (NO EMBER2024)
    # ------------------------------------------------------------------
    logger.info("\n### Evaluation ###")

    y_pred_proba = model.predict(df_base_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}")
    
    # ------------------------------------------------------------------
    # 6) THRESHOLD OPTIMIZATION
    # ------------------------------------------------------------------
    logger.info("\n### Threshold Optimization ###")
    
    best_thresh = 0.5
    best_score = 0.0
    
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_t = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        if fpr_t < 0.01 and fnr_t < 0.10:
            score = 1 - (fpr_t + fnr_t) / 2
            logger.info(f"✓ Thresh {thresh:.2f}: FPR={fpr_t:.4f}, FNR={fnr_t:.4f}")
            if score > best_score:
                best_score = score
                best_thresh = thresh
        elif fpr_t < 0.05 and fnr_t < 0.15:
            logger.info(f"~ Thresh {thresh:.2f}: FPR={fpr_t:.4f}, FNR={fnr_t:.4f}")
    
    logger.info(f"\nBest Threshold: {best_thresh:.2f}")
    
    y_pred_best = (y_pred_proba > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    fpr_best = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_best = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc_best = accuracy_score(y_test, y_pred_best)
    
    logger.info(f"With Best Threshold: Acc={acc_best:.4f}, FPR={fpr_best:.4f}, FNR={fnr_best:.4f}")
    
    # ------------------------------------------------------------------
    # 7) SAVE FINAL MODEL, FEATURE ORDER, THRESHOLD
    # ------------------------------------------------------------------
    logger.info("\n### Saving Model ###")

    model_dir = os.path.join(os.path.dirname(__file__), "defender/models")
    os.makedirs(model_dir, exist_ok=True)

    # Wrap booster into sklearn-compatible class
    final_model = lgb.LGBMClassifier()
    final_model._Booster = model

    with open(os.path.join(model_dir, "advanced_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    scaler = None
    with open(os.path.join(model_dir, "advanced_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(model_dir, "advanced_features.pkl"), "wb") as f:
        pickle.dump(all_columns, f)

    with open(os.path.join(model_dir, "advanced_threshold.pkl"), "wb") as f:
        pickle.dump(best_thresh, f)

    logger.info(f"Saved all artifacts to {model_dir}")
    
    return {
        'accuracy': acc_best,
        'f1_score': f1,
        'auc': auc,
        'fpr': fpr_best,
        'fnr': fnr_best,
        'best_threshold': best_thresh,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with challenge dataset included')
    parser.add_argument('--ember-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2018_2/ember2018')
    parser.add_argument('--ember2017-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/ember_dataset_2017_2/ember_2017_2')
    parser.add_argument('--dike-dir', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/DikeDataset/files')
    parser.add_argument('--challenge-goodware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/goodware')
    parser.add_argument('--challenge-malware', type=str, default='/mnt/c/Users/kotas/Downloads/DACySec/challenge/challenge_ds/malware')    
    parser.add_argument("--ember-samples", type=int, default=30000)
    parser.add_argument(
    '--ember2024-dir',
    type=str,
    default='/mnt/c/Users/kotas/Downloads/DACySec/EMBER2024/data2',
    help='Root directory where thrember.download_dataset(file_type=\"PE\") was run'
    )

    parser.add_argument(
        '--ember2024-samples',
        type=int,
        default=30000,
        help='How many EMBER2024 samples to stream from JSONL'
    )
    
    args = parser.parse_args()
    
    results = train_with_challenge(
        args.ember_dir,
        args.ember2017_dir,
        args.dike_dir,
        args.challenge_goodware,
        args.challenge_malware,
        args.ember_samples,
        args.ember2024_dir,
        args.ember2024_samples
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print(f"FPR: {results['fpr']:.4f} (target: <0.01)")
    print(f"FNR: {results['fnr']:.4f} (target: <0.10)")
    print(f"Best Threshold: {results['best_threshold']:.2f}")
    print("=" * 80)