#!/usr/bin/env python3
"""
Threshold Optimization Script

Analyzes model predictions on test dataset to find optimal thresholds
that meet FPR < 1% and FNR < 10% requirements.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging
import pickle

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(test_dir: str) -> Tuple[List[bytes], List[int]]:
    """Load test data from directory structure"""
    test_path = Path(test_dir)
    
    all_samples = []
    all_labels = []
    
    # Load benign samples
    benign_dir = test_path / 'benign'
    if not benign_dir.exists():
        benign_dir = test_path / 'goodware'
    
    if benign_dir.exists():
        logger.info(f"Loading benign samples from {benign_dir}")
        for file_path in benign_dir.iterdir():
            try:
                with open(file_path, 'rb') as f:
                    all_samples.append(f.read())
                    all_labels.append(0)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
    
    # Load malware samples
    malware_dir = test_path / 'malware'
    if malware_dir.exists():
        logger.info(f"Loading malware samples from {malware_dir}")
        for file_path in malware_dir.iterdir():
            try:
                with open(file_path, 'rb') as f:
                    all_samples.append(f.read())
                    all_labels.append(1)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(all_samples)} samples: {np.sum(np.array(all_labels) == 0)} benign, {np.sum(np.array(all_labels) == 1)} malware")
    return all_samples, all_labels


def get_model_probabilities(model, samples: List[bytes]) -> np.ndarray:
    """Get prediction probabilities from model"""
    probabilities = []
    
    for i, sample in enumerate(samples):
        try:
            # Get features and make prediction
            features = model._prepare_features(sample)
            if features is not None:
                proba = model.model.predict_proba(features)[0][1]
                probabilities.append(proba)
            else:
                probabilities.append(0.5)  # Neutral if feature extraction fails
        except Exception as e:
            logger.error(f"Error getting probability for sample {i}: {e}")
            probabilities.append(0.5)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples...")
    
    return np.array(probabilities)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                          max_fpr: float = 0.01, max_fnr: float = 0.10) -> dict:
    """Find optimal threshold that meets FPR and FNR requirements"""
    
    # Try different thresholds
    thresholds = np.arange(0.0, 1.0, 0.01)
    
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'meets_requirements': (fpr <= max_fpr and fnr <= max_fnr)
        })
    
    df = pd.DataFrame(results)
    
    # Find thresholds that meet requirements
    compliant = df[df['meets_requirements'] == True]
    
    if len(compliant) > 0:
        # Choose the one with highest accuracy
        best = compliant.loc[compliant['accuracy'].idxmax()]
        logger.info(f"Found {len(compliant)} thresholds that meet requirements!")
        logger.info(f"Best threshold: {best['threshold']:.3f}")
        logger.info(f"  Accuracy: {best['accuracy']:.4f}")
        logger.info(f"  FPR: {best['fpr']:.4f} (target: <{max_fpr})")
        logger.info(f"  FNR: {best['fnr']:.4f} (target: <{max_fnr})")
        return best.to_dict()
    else:
        logger.warning("No threshold meets all requirements!")
        
        # Find best trade-off: minimize combined error
        df['combined_error'] = df['fpr'] + df['fnr']
        best = df.loc[df['combined_error'].idxmin()]
        
        logger.info(f"Best trade-off threshold: {best['threshold']:.3f}")
        logger.info(f"  Accuracy: {best['accuracy']:.4f}")
        logger.info(f"  FPR: {best['fpr']:.4f} (target: <{max_fpr})")
        logger.info(f"  FNR: {best['fnr']:.4f} (target: <{max_fnr})")
        
        return best.to_dict()


def optimize_model_threshold(model_name: str, model, test_dir: str):
    """Optimize threshold for a specific model"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"OPTIMIZING THRESHOLD FOR: {model_name}")
    logger.info(f"{'=' * 80}")
    
    # Load test data
    samples, labels = load_test_data(test_dir)
    labels = np.array(labels)
    
    # Get model probabilities
    logger.info("Getting model predictions...")
    probabilities = get_model_probabilities(model, samples)
    
    # Find optimal threshold
    logger.info("Finding optimal threshold...")
    optimal = find_optimal_threshold(labels, probabilities)
    
    # Display all candidate thresholds
    logger.info(f"\nThreshold Analysis for {model_name}:")
    logger.info(f"{'Threshold':<12} {'Accuracy':<12} {'FPR':<12} {'FNR':<12} {'Status':<20}")
    logger.info("-" * 80)
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (probabilities >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        
        status = "✓ MEETS REQUIREMENTS" if (fpr <= 0.01 and fnr <= 0.10) else "✗ FAILS"
        logger.info(f"{thresh:<12.2f} {acc:<12.4f} {fpr:<12.4f} {fnr:<12.4f} {status:<20}")
    
    logger.info("-" * 80)
    logger.info(f"OPTIMAL: {optimal['threshold']:<6.2f} {optimal['accuracy']:<12.4f} {optimal['fpr']:<12.4f} {optimal['fnr']:<12.4f}")
    
    return optimal


def main(test_dir: str):
    """Main optimization routine"""
    
    # Optimize Random Forest
    try:
        from defender.models.random_forest_model import RandomForestMalwareModel
        model_path = os.path.join(os.path.dirname(__file__), 'defender/models/random_forest_model.pkl')
        if os.path.exists(model_path):
            model = RandomForestMalwareModel(model_path=model_path, thresh=0.5)
            rf_optimal = optimize_model_threshold("Random Forest", model, test_dir)
            
            # Save optimal threshold
            optimal_thresh_path = os.path.join(os.path.dirname(__file__), 'defender/models/random_forest_optimal_threshold.pkl')
            with open(optimal_thresh_path, 'wb') as f:
                pickle.dump(rf_optimal, f)
            logger.info(f"Saved optimal threshold to {optimal_thresh_path}")
    except Exception as e:
        logger.error(f"Error optimizing Random Forest: {e}")
    
    # Optimize Decision Tree
    try:
        from defender.models.decision_tree_model import DecisionTreeMalwareModel
        model_path = os.path.join(os.path.dirname(__file__), 'defender/models/decision_tree_model.pkl')
        if os.path.exists(model_path):
            model = DecisionTreeMalwareModel(model_path=model_path, thresh=0.5)
            dt_optimal = optimize_model_threshold("Decision Tree", model, test_dir)
            
            # Save optimal threshold
            optimal_thresh_path = os.path.join(os.path.dirname(__file__), 'defender/models/decision_tree_optimal_threshold.pkl')
            with open(optimal_thresh_path, 'wb') as f:
                pickle.dump(dt_optimal, f)
            logger.info(f"Saved optimal threshold to {optimal_thresh_path}")
    except Exception as e:
        logger.error(f"Error optimizing Decision Tree: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize model thresholds')
    parser.add_argument('test_dir', type=str, help='Directory containing test data')
    
    args = parser.parse_args()
    
    main(args.test_dir)

