#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
from typing import List
import logging

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from defender.models.reliable_nn_model import ReliablePEFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_simple_reliable_model():
    """Train a simpler but effective sklearn-based neural network model"""
    
    # --- 1. Data Loading and Preparation ---
    malware_dir = '/home/gamdhameet/DikeDataset-main/files/malware'
    benign_dir = '/home/gamdhameet/DikeDataset-main/files/benign'

    malware_files = [os.path.join(malware_dir, f) for f in os.listdir(malware_dir)]
    benign_files = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir)]

    files = malware_files + benign_files
    labels = [1] * len(malware_files) + [0] * len(benign_files)

    # --- 2. Feature Extraction ---
    feature_extractor = ReliablePEFeatureExtractor()
    all_features = []
    textual_features = []
    valid_labels = []
    
    logger.info(f"Starting feature extraction for {len(files)} files...")
    
    for i, f in enumerate(files):
        try:
            with open(f, 'rb') as fp:
                bytez = fp.read()
                features = feature_extractor.extract(bytez)
                if features:
                    numerical_part = {k: v for k, v in features.items() if isinstance(v, (int, float))}
                    textual_part = {k: v for k, v in features.items() if isinstance(v, str)}
                    all_features.append(numerical_part)
                    
                    # Combine all textual features into one string for TF-IDF
                    combined_text = " ".join(textual_part.values())
                    textual_features.append(combined_text)
                    valid_labels.append(labels[i])
                    
            if (i + 1) % 200 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files...")
                
        except Exception as e:
            logger.warning(f"Error processing file {f}: {e}")
            continue
    
    logger.info(f"Successfully extracted features from {len(all_features)} files")
    labels = valid_labels

    # --- 3. Feature Processing ---
    df_numerical = pd.DataFrame(all_features)
    df_numerical = df_numerical.fillna(0)  # Handle missing values

    # TF-IDF for textual features - reduced size for efficiency
    tfidf = TfidfVectorizer(max_features=200, min_df=3, max_df=0.9, ngram_range=(1,1))
    text_features_tfidf = tfidf.fit_transform(textual_features).toarray()

    # Scale numerical features
    scaler_numerical = StandardScaler()
    numerical_features_scaled = scaler_numerical.fit_transform(df_numerical)

    # Combine features
    combined_features = np.hstack((numerical_features_scaled, text_features_tfidf))
    
    logger.info(f"Total feature dimension: {combined_features.shape[1]}")
    
    # --- 4. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, np.array(labels), test_size=0.2, random_state=42, stratify=labels
    )

    # --- 5. Model Training - Simple but effective configuration ---
    logger.info("Starting model training...")
    
    # Simple but effective MLP configuration
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train, y_train)
    
    # --- 6. Model Evaluation ---
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Calculate FPR and FNR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    logger.info(f"False Positive Rate: {fpr:.4f}")
    logger.info(f"False Negative Rate: {fnr:.4f}")

    # --- 7. Save Model and Preprocessors ---
    model_dir = os.path.join(os.path.dirname(__file__), 'defender/models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(model_dir, 'reliable_nn_model.pkl'), 'wb') as f:
        pickle.dump(mlp, f)
    
    # Save TF-IDF vectorizer
    with open(os.path.join(model_dir, 'reliable_tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    
    # Save scaler
    with open(os.path.join(model_dir, 'reliable_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_numerical, f)
        
    logger.info(f"Model and preprocessors saved in {model_dir}")
    
    # Return performance metrics
    return {
        'accuracy': accuracy,
        'fpr': fpr,
        'fnr': fnr,
        'model_params': {
            'hidden_layers': (128, 64),
            'alpha': 0.001,
            'max_iter': mlp.n_iter_
        }
    }


if __name__ == "__main__":
    results = train_simple_reliable_model()
    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"FPR: {results['fpr']:.4f} (Target: <0.01)")
    print(f"FNR: {results['fnr']:.4f} (Target: <0.10)")
    print(f"Model converged in {results['model_params']['max_iter']} iterations")
    
    # Check if requirements are met
    if results['fpr'] < 0.01 and results['fnr'] < 0.10:
        print("✅ Model meets competition requirements!")
    else:
        print("❌ Model needs improvement to meet competition requirements.")
