"""
Advanced Malware Detection Model

This model uses LightGBM for better performance and includes:
1. Enhanced feature extraction
2. YARA rules for heuristic fallback
3. Optimized threshold tuning
"""

import os
import pickle
import numpy as np
import logging
from typing import Dict, Any

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not available, will use fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMalwareModel:
    """Advanced malware detection model with LightGBM and YARA rules"""
    
    # YARA-style patterns (simplified without yara-python dependency for now)
    YARA_PATTERNS = {
        'suspicious_api': [
            (b'CreateRemoteThread', 5), (b'WriteProcessMemory', 5), 
            (b'VirtualAllocEx', 4), (b'SetWindowsHookEx', 4),
            (b'GetAsyncKeyState', 4), (b'URLDownloadToFile', 5),
            (b'WinExec', 3), (b'ShellExecute', 3),
        ],
        'packing_indicators': [
            (b'UPX', 4), (b'MPRESS', 4), (b'ASPack', 4), (b'PECompact', 4),
            (b'themida', 3), (b'VMProtect', 3), (b'Armadillo', 3),
        ],
        'ransomware_indicators': [
            (b'encrypt', 3), (b'decrypt', 2), (b'ransom', 5), (b'bitcoin', 4),
            (b'wallet', 3), (b'payment', 2), (b'.locked', 4), (b'.encrypted', 4),
        ],
        'keylogger_indicators': [
            (b'keylog', 5), (b'GetAsyncKeyState', 4), (b'GetForegroundWindow', 3),
            (b'GetWindowText', 3), (b'keyboard', 3),
        ],
        'network_indicators': [
            (b'http://', 1), (b'https://', 1), (b'ftp://', 2),
            (b'InternetOpen', 2), (b'InternetConnect', 2), 
            (b'HttpOpenRequest', 2), (b'send', 1), (b'recv', 1),
        ],
        'persistence_indicators': [
            (b'RegCreateKey', 2), (b'RegSetValue', 2), (b'CreateService', 3),
            (b'StartService', 3), (b'schtasks', 3), (b'CurrentVersion\\Run', 4),
        ],
        'injection_indicators': [
            (b'CreateRemoteThread', 5), (b'WriteProcessMemory', 5),
            (b'VirtualAllocEx', 4), (b'SetThreadContext', 4),
            (b'QueueUserAPC', 4), (b'NtUnmapViewOfSection', 4),
        ],
        'evasion_indicators': [
            (b'IsDebuggerPresent', 3), (b'CheckRemoteDebuggerPresent', 3),
            (b'OutputDebugString', 2), (b'DebugActiveProcess', 4),
            (b'FindWindow', 2), (b'GetTickCount', 2), (b'Sleep', 1),
        ],
    }
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'Advanced-LightGBM-Detector'):
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
        """Load the LightGBM model and preprocessing components"""
        try:
            model_dir = os.path.dirname(self.model_path) if self.model_path else os.path.join(os.path.dirname(__file__))
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'advanced_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            # Load feature names
            feature_names_path = os.path.join(model_dir, 'advanced_features.pkl')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load LightGBM model
            model_path = os.path.join(model_dir, 'advanced_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded LightGBM model from {model_path}")
            
            if self.model and self.scaler and self.feature_names:
                self.is_loaded = True
                logger.info(f"Successfully loaded advanced model components")
            else:
                logger.warning("Some model components missing, using YARA-based detection")
                
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            self.is_loaded = False
    
    def _extract_features_from_bytes(self, bytez: bytes) -> Dict[str, Any]:
        """Extract features from PE file bytes using enhanced extractor"""
        try:
            from defender.models.enhanced_feature_extractor import EnhancedPEFeatureExtractor
            
            extractor = EnhancedPEFeatureExtractor()
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
            logger.warning("Model not loaded, using YARA-based detection")
            return self._yara_heuristic_prediction(bytez)
        
        try:
            # Prepare features
            features = self._prepare_features(bytez)
            if features is None:
                return self._yara_heuristic_prediction(bytez)
            
            # Make prediction - Pure ML without hybrid
            if HAS_LIGHTGBM and hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features)[0]
                prediction = int(prediction_proba[1] > self.thresh)
                logger.info(f"ML Prediction: {prediction} (confidence: {prediction_proba[1]:.4f}, threshold: {self.thresh})")
            else:
                # Fallback to predict if predict_proba not available
                prediction = int(self.model.predict(features)[0])
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._yara_heuristic_prediction(bytez)
    
    def _get_yara_score(self, bytez: bytes) -> int:
        """Get YARA heuristic score without making a decision"""
        try:
            if len(bytez) < 1024 or not bytez.startswith(b'MZ'):
                return 10  # Definitely suspicious
            
            malicious_score = 0
            bytez_lower = bytez.lower()
            
            # Check each pattern category
            for category, patterns in self.YARA_PATTERNS.items():
                category_hits = 0
                for pattern, weight in patterns:
                    if pattern.lower() in bytez_lower:
                        malicious_score += weight
                        category_hits += 1
                
                if category_hits >= 3:
                    malicious_score += 2
            
            # Entropy check
            entropy = self._calculate_entropy(bytez)
            if entropy > 7.8:
                malicious_score += 3
            elif entropy > 7.2:
                malicious_score += 1
            
            # Size checks
            if len(bytez) < 8192:
                malicious_score += 2
            elif len(bytez) > 20000000:
                malicious_score += 1
            
            # Multiple PE headers
            if bytez.count(b'MZ') > 3:
                malicious_score += 3
            
            return malicious_score
        except:
            return 0
    
    def _yara_heuristic_prediction(self, bytez: bytes) -> int:
        """YARA-style heuristic prediction - used only as fallback"""
        try:
            score = self._get_yara_score(bytez)
            threshold = 7  # High threshold to avoid false positives when used alone
            decision = 1 if score >= threshold else 0
            logger.info(f"YARA Fallback: score={score}, threshold={threshold}, decision={decision}")
            return decision
        except Exception as e:
            logger.error(f"Error in YARA heuristic: {e}")
            return 1  # Default to malicious for safety
    
    def _calculate_entropy(self, bytez: bytes) -> float:
        """Calculate Shannon entropy of byte sequence"""
        if not bytez:
            return 0
        
        from collections import Counter
        byte_counts = Counter(bytez)
        entropy = 0
        total = len(bytez)
        
        for count in byte_counts.values():
            p_x = count / total
            if p_x > 0:
                entropy += -p_x * np.log2(p_x)
        
        return entropy
    
    def model_info(self) -> dict:
        """Return model information"""
        info = {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": "LightGBM Classifier with YARA heuristics",
        }
        
        if self.is_loaded and self.model:
            info.update({
                "n_features": len(self.feature_names) if self.feature_names else 'N/A',
                "has_lightgbm": HAS_LIGHTGBM,
            })
        
        return info

