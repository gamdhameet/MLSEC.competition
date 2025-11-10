"""
Malware prediction using trained Random Forest model
"""
import os
import pickle
import numpy as np
import pandas as pd
import logging
from .feature_extractor import PEFeatureExtractor

logger = logging.getLogger(__name__)

class MalwarePredictor:
    """Predict malware using trained model"""
    
    def __init__(self, model_path, scaler_path, features_path, threshold=0.6):
        """Initialize predictor with model files"""
        logger.info("Loading model components...")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        self.threshold = threshold
        self.feature_extractor = PEFeatureExtractor(expected_feature_count=len(self.feature_names))
        
        logger.info(f"Model loaded: {len(self.feature_names)} features, threshold={threshold}")
    
    def predict(self, file_bytes: bytes) -> dict:
        """
        Predict whether file is malware
        
        Args:
            file_bytes: Raw bytes of PE file
            
        Returns:
            dict with prediction, probability, label
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features_from_bytes(file_bytes)
            
            # Convert to DataFrame
            X = pd.DataFrame([features], columns=self.feature_names)
            
            # Handle NaN/Inf
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probability
            prob = self.model.predict_proba(X_scaled)[0, 1]
            
            # Apply threshold
            prediction = int(prob > self.threshold)
            label = "malware" if prediction == 1 else "benign"
            
            return {
                'prediction': prediction,
                'label': label,
                'probability': float(prob),
                'threshold': self.threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise