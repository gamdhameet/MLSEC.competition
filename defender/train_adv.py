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


def train_with_challenge(ember_dir: str, ember2017_dir: str, dike_dir: str, challenge_goodware: str, 
                        challenge_malware: str, ember_samples: int = 30000):
    
    logger.info("=" * 80)
    logger.info("TRAINING WITH CHALLENGE DATASET INCLUDED")
    logger.info("=" * 80)
    
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
    
    # 2. Load DikeDataset (similar distribution)
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
    
    # 3. Load Ember samples (for diversity)
    logger.info(f"\n### Loading {ember_samples} Ember samples ###")
    ember_feat, ember_labels = load_ember_samples(ember_dir, ember_samples)
    
    all_features.extend(ember_feat)
    all_labels.extend(ember_labels)
    
    logger.info(f"Ember dataset: {len(ember_labels)} samples")

    logger.info(f"\n### Loading {ember_samples} Ember samples ###")
    ember_feat, ember_labels = load_ember_samples(ember2017_dir, ember_samples)
    
    all_features.extend(ember_feat)
    all_labels.extend(ember_labels)
    
    logger.info(f"Ember dataset: {len(ember_labels)} samples")
    
    # Convert to DataFrame
    logger.info("\n### Preparing Features ###")
    df = pd.DataFrame(all_features)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    y = np.array(all_labels)
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Features: {len(df.columns)}")
    logger.info(f"Label distribution: Benign={np.sum(y == 0)}, Malicious={np.sum(y == 1)}")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

    # Train-Test Split (stratified to ensure challenge samples in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.15, random_state=42, stratify=y
    )
    
    logger.info(f"\nTraining set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # LightGBM Training
    logger.info("\n### Training LightGBM ###")
    
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 40,
        'max_depth': 12,
        'learning_rate': 0.05,
        'n_estimators': 500,
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
    
    logger.info("Training...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    )
    
    # Evaluation
    logger.info("\n### Evaluation ###")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")
    
    # Threshold Optimization
    logger.info("\n### Threshold Optimization ###")
    
    best_thresh = 0.5
    best_score = 0
    
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_t = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if fpr_t < 0.01 and fnr_t < 0.10:
            score = 1 - (fpr_t + fnr_t) / 2
            logger.info(f"✓ Thresh {thresh:.2f}: FPR={fpr_t:.4f}, FNR={fnr_t:.4f}")
            if score > best_score:
                best_score = score
                best_thresh = thresh
        elif fpr_t < 0.05 and fnr_t < 0.15:  # Relaxed constraints
            logger.info(f"~ Thresh {thresh:.2f}: FPR={fpr_t:.4f}, FNR={fnr_t:.4f}")
    
    logger.info(f"\nBest Threshold: {best_thresh:.2f}")
    
    # Evaluate with best threshold
    y_pred_best = (y_pred_proba > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    fpr_best = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_best = fn / (fn + tp) if (fn + tp) > 0 else 0
    acc_best = accuracy_score(y_test, y_pred_best)
    
    logger.info(f"With Best Threshold: Acc={acc_best:.4f}, FPR={fpr_best:.4f}, FNR={fnr_best:.4f}")
    
    # Save Model
    logger.info("\n### Saving Model ###")
    
    model_dir = os.path.join(os.path.dirname(__file__), 'defender/models')
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'advanced_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(model_dir, 'advanced_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(model_dir, 'advanced_features.pkl'), 'wb') as f:
        pickle.dump(list(df.columns), f)
    
    with open(os.path.join(model_dir, 'advanced_threshold.pkl'), 'wb') as f:
        pickle.dump(best_thresh, f)
    
    logger.info(f"Model saved to {model_dir}")
    
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
    
    args = parser.parse_args()
    
    results = train_with_challenge(
        args.ember_dir,
        args.ember2017_dir,
        args.dike_dir,
        args.challenge_goodware,
        args.challenge_malware,
        args.ember_samples
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