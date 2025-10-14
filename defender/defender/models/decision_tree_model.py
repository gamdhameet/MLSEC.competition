"""
Decision Tree Malware Detection Model

This model uses sklearn's DecisionTreeClassifier for malware detection.
Optimized for Docker deployment with very fast inference.
"""

import os
import pickle
import numpy as np
import logging
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionTreeMalwareModel:
    """Decision Tree based malware detection model compatible with defender framework"""
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'Decision-Tree-Malware-Detector'):
        self.thresh = thresh
        self.__name__ = name
        self.model_path = model_path
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        
        # Load pre-trained model if available
        if model_path:
            self._load_model_components()
    
    def _load_model_components(self):
        """Load the Decision Tree model and preprocessing components"""
        try:
            model_dir = os.path.dirname(self.model_path) if self.model_path else os.path.join(os.path.dirname(__file__))
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'decision_tree_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load feature names
            feature_names_path = os.path.join(model_dir, 'decision_tree_features.pkl')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load Decision Tree model
            dt_model_path = os.path.join(model_dir, 'decision_tree_model.pkl')
            if os.path.exists(dt_model_path):
                with open(dt_model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded Decision Tree model from {dt_model_path}")
            
            if self.model and self.scaler and self.feature_names:
                self.is_loaded = True
                logger.info(f"Successfully loaded Decision Tree model components")
            else:
                logger.warning("Some model components missing, using heuristic-based detection")
                
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            self.is_loaded = False
    
    def _extract_features_from_bytes(self, bytez: bytes) -> Dict[str, Any]:
        """Extract features from PE file bytes"""
        try:
            # Import here to avoid circular dependency
            from defender.models.reliable_nn_model import ReliablePEFeatureExtractor
            
            extractor = ReliablePEFeatureExtractor()
            features = extractor.extract(bytez)
            
            # Extract only numerical features
            numerical_features = {k: v for k, v in features.items() 
                                if isinstance(v, (int, float))}
            
            return numerical_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _prepare_features(self, bytez: bytes) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            # Extract features
            features_dict = self._extract_features_from_bytes(bytez)
            if not features_dict:
                return None
            
            # Align features with training feature names
            if self.feature_names:
                # Create feature vector in the same order as training
                feature_vector = []
                for fname in self.feature_names:
                    feature_vector.append(features_dict.get(fname, 0))
                features_array = np.array(feature_vector).reshape(1, -1)
            else:
                # Fallback: use all available features
                features_array = np.array(list(features_dict.values())).reshape(1, -1)
            
            # Scale features
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def predict(self, bytez: bytes) -> int:
        """Predict if a PE file is malicious (0=benign, 1=malicious)"""
        if not self.is_loaded:
            logger.warning("Model not loaded, using heuristic-based detection")
            return self._heuristic_prediction(bytez)
        
        try:
            # Prepare features
            features = self._prepare_features(bytez)
            if features is None:
                return self._heuristic_prediction(bytez)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features)[0]
            prediction = int(prediction_proba[1] > self.thresh)
            
            logger.info(f"Decision Tree Prediction: {prediction} (confidence: {prediction_proba[1]:.4f}, threshold: {self.thresh})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during Decision Tree prediction: {e}")
            return self._heuristic_prediction(bytez)
    
    def _heuristic_prediction(self, bytez: bytes) -> int:
        """Heuristic-based prediction as fallback"""
        try:
            # Basic checks
            if len(bytez) < 1024:
                return 1
            
            if not bytez.startswith(b'MZ'):
                return 1
            
            # Pattern matching with weighted scoring
            malicious_score = 0
            
            suspicious_patterns = [
                (b'cmd.exe', 2), (b'powershell', 3), (b'rundll32', 2),
                (b'regsvr32', 2), (b'certutil', 3), (b'bitsadmin', 3),
                (b'schtasks', 2), (b'wmic', 2), (b'reg.exe', 1),
                (b'http://', 1), (b'https://', 1), (b'ftp://', 2),
                (b'keylog', 4), (b'steal', 3), (b'bypass', 3), 
                (b'inject', 4), (b'payload', 4), (b'exploit', 4),
                (b'shellcode', 5), (b'backdoor', 5), (b'trojan', 5),
                (b'CreateRemoteThread', 3), (b'WriteProcessMemory', 3),
                (b'VirtualAllocEx', 3), (b'SetWindowsHookEx', 2),
            ]
            
            for pattern, weight in suspicious_patterns:
                if pattern in bytez.lower():
                    malicious_score += weight
            
            # Check entropy
            entropy = self._calculate_entropy(bytez)
            if entropy > 7.5:
                malicious_score += 2
            elif entropy > 7.0:
                malicious_score += 1
            
            return 1 if malicious_score >= 3 else 0
            
        except Exception:
            return 1  # Default to malicious for safety
    
    def _calculate_entropy(self, bytez: bytes) -> float:
        """Calculate entropy of byte sequence"""
        if not bytez:
            return 0
        entropy = 0
        for x in range(256):
            p_x = bytez.count(x) / len(bytez)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy
    
    def model_info(self) -> dict:
        """Return model information"""
        info = {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": "sklearn DecisionTreeClassifier",
        }
        
        if self.is_loaded and self.model:
            info.update({
                "max_depth": getattr(self.model, 'max_depth', 'N/A'),
                "n_features": len(self.feature_names) if self.feature_names else 'N/A',
                "n_leaves": getattr(self.model, 'get_n_leaves', lambda: 'N/A')(),
            })
        
        return info

