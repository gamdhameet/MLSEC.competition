#!/usr/bin/env python3
"""
Training script for Random Forest malware detection model

Uses combined datasets: Ember + DikeDataset
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from defender.models.data_loader import CombinedDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_random_forest_model(ember_max_samples: int = 100000, use_validation_split: bool = True):
    """Train the Random Forest model on combined datasets"""
    
    # --- 1. Data Loading ---
    ember_dir = '/home/gamdhameet/ember_dataset/ember'
    dike_dir = '/home/gamdhameet/DikeDataset-main/files'
    
    logger.info("=" * 60)
    logger.info("RANDOM FOREST MODEL TRAINING (Enhanced)")
    logger.info(f"Using {ember_max_samples} ember samples")
    logger.info("=" * 60)
    
    loader = CombinedDataLoader(ember_dir, dike_dir)
    X, y = loader.load_combined(ember_max_samples=ember_max_samples)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {X.shape[0]}")
    logger.info(f"Benign: {np.sum(y == 0)}, Malware: {np.sum(y == 1)}")
    
    # --- 2. Data Preprocessing ---
    # Remove any infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 3. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # --- 4. Model Training with Hyperparameter Tuning ---
    logger.info("Starting Random Forest training with hyperparameter optimization...")
    
    # Define parameter grid (balanced for speed and performance)
    param_grid = {
        'n_estimators': [150, 200],  # Sufficient trees for good performance
        'max_depth': [25, 30],  # Deep enough but not unlimited
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],  # Reduced from 3 variations
        'max_features': ['sqrt'],  # sqrt is usually best for RF
        'class_weight': ['balanced']  # balanced works well
    }
    
    # Create base model with optimized settings
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=4,  # Use more cores for faster training
        warm_start=False,
        bootstrap=True,
        oob_score=True  # Use out-of-bag score for validation
    )
    
    # Grid search with cross-validation
    logger.info("Running GridSearchCV (this may take a while)...")
    logger.info(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['class_weight'])} combinations")
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5,  # More folds for better validation
        scoring='f1', 
        n_jobs=2,  # Limit parallel jobs to avoid memory issues
        verbose=2,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # --- 5. Model Evaluation ---
    logger.info("Evaluating model on test set...")
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Calculate FPR and FNR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TN: {tn}, FP: {fp}")
    logger.info(f"FN: {fn}, TP: {tp}")
    logger.info(f"\nFalse Positive Rate: {fpr:.4f} (target: <0.01)")
    logger.info(f"False Negative Rate: {fnr:.4f} (target: <0.10)")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 20 Most Important Features:")
        logger.info(feature_importance.head(20).to_string())
    
    # --- 6. Save Model and Preprocessors ---
    model_dir = os.path.join(os.path.dirname(__file__), 'defender/models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'random_forest_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(model_dir, 'random_forest_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    logger.info(f"Feature names saved to {features_path}")
    
    logger.info(f"\nAll components saved in {model_dir}")
    
    # --- 7. Test with Different Thresholds ---
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD ANALYSIS")
    logger.info("=" * 60)
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        fpr_thresh = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_thresh = fn / (fn + tp) if (fn + tp) > 0 else 0
        acc_thresh = accuracy_score(y_test, y_pred_thresh)
        
        logger.info(f"Threshold {thresh:.1f}: Acc={acc_thresh:.4f}, FPR={fpr_thresh:.4f}, FNR={fnr_thresh:.4f}")
    
    # Return performance metrics
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest malware detection model')
    parser.add_argument('--ember-samples', type=int, default=100000,
                       help='Maximum number of ember samples to use (default: 100000)')
    
    args = parser.parse_args()
    
    results = train_random_forest_model(ember_max_samples=args.ember_samples)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"FPR: {results['fpr']:.4f} (target: <0.01)")
    print(f"FNR: {results['fnr']:.4f} (target: <0.10)")
    print(f"CV Score: {results['cv_score']:.4f}")
    print(f"Best Parameters: {results['best_params']}")
    print("=" * 60)

