"""
Enhanced Neural Network Malware Detection Model

This model uses a deep learning approach with comprehensive PE feature extraction
to achieve better malware detection performance than the simple BERT model.
"""

import os
import re
import lief
import math
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any
import logging

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPEFeatureExtractor:
    """
    Extracts a comprehensive set of features from PE files.
    """

    def __init__(self):
        self.lief_binary = None

    def extract(self, bytez: bytes) -> Dict[str, Any]:
        """Extract all features from a PE file."""
        try:
            self.lief_binary = lief.PE.parse(list(bytez))
            if not self.lief_binary:
                return {}
        except Exception as e:
            logger.warning(f"Could not parse PE file: {e}")
            return {}

        features = {}
        features.update(self._extract_general_features(bytez))
        features.update(self._extract_header_features())
        features.update(self._extract_optional_header_features())
        features.update(self._extract_string_metadata(bytez))
        features.update(self._extract_entropy(bytez))
        features.update(self._extract_imports_and_exports())

        return features

    def _extract_general_features(self, bytez: bytes) -> Dict[str, Any]:
        return {
            "size": len(bytez),
            "virtual_size": self.lief_binary.virtual_size,
            "has_debug": int(self.lief_binary.has_debug),
            "has_relocations": int(self.lief_binary.has_relocations),
            "has_resources": int(self.lief_binary.has_resources),
            "has_signature": int(self.lief_binary.has_signatures),
            "has_tls": int(self.lief_binary.has_tls),
            "symbols": len(self.lief_binary.symbols) if self.lief_binary.has_symbol else 0,
        }

    def _extract_header_features(self) -> Dict[str, Any]:
        header = self.lief_binary.header
        return {
            "timestamp": header.time_date_stamps,
            "machine": header.machine.value,
            "numberof_sections": header.numberof_sections,
            "numberof_symbols": header.numberof_symbols,
            "pointerto_symbol_table": header.pointerto_symbol_table,
            "sizeof_optional_header": header.sizeof_optional_header,
            "characteristics": header.characteristics,
        }

    def _extract_optional_header_features(self) -> Dict[str, Any]:
        opt_header = self.lief_binary.optional_header
        try:
            baseof_data = opt_header.baseof_data
        except:
            baseof_data = 0
            
        return {
            "baseof_code": opt_header.baseof_code,
            "baseof_data": baseof_data,
            "dll_characteristics": opt_header.dll_characteristics,
            "file_alignment": opt_header.file_alignment,
            "imagebase": opt_header.imagebase,
            "magic": opt_header.magic.value,
            "major_image_version": opt_header.major_image_version,
            "minor_image_version": opt_header.minor_image_version,
            "major_linker_version": opt_header.major_linker_version,
            "minor_linker_version": opt_header.minor_linker_version,
            "major_operating_system_version": opt_header.major_operating_system_version,
            "minor_operating_system_version": opt_header.minor_operating_system_version,
            "major_subsystem_version": opt_header.major_subsystem_version,
            "minor_subsystem_version": opt_header.minor_subsystem_version,
            "numberof_rva_and_size": opt_header.numberof_rva_and_size,
            "sizeof_code": opt_header.sizeof_code,
            "sizeof_headers": opt_header.sizeof_headers,
            "sizeof_heap_commit": opt_header.sizeof_heap_commit,
            "sizeof_image": opt_header.sizeof_image,
            "sizeof_initialized_data": opt_header.sizeof_initialized_data,
            "sizeof_uninitialized_data": opt_header.sizeof_uninitialized_data,
            "subsystem": opt_header.subsystem.value,
        }

    def _extract_string_metadata(self, bytez: bytes) -> Dict[str, Any]:
        paths = re.compile(b'c:\\\\', re.IGNORECASE)
        urls = re.compile(b'https?://', re.IGNORECASE)
        registry = re.compile(b'HKEY_')
        mz = re.compile(b'MZ')
        return {
            'string_paths': len(paths.findall(bytez)),
            'string_urls': len(urls.findall(bytez)),
            'string_registry': len(registry.findall(bytez)),
            'string_MZ': len(mz.findall(bytez))
        }

    def _extract_entropy(self, bytez: bytes) -> Dict[str, Any]:
        if not bytez:
            return {'entropy': 0}
        entropy = 0
        for x in range(256):
            p_x = bytez.count(x) / len(bytez)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return {'entropy': entropy}

    def _extract_imports_and_exports(self) -> Dict[str, Any]:
        imports = ""
        exports = ""
        try:
            if self.lief_binary.has_imports:
                import_names = []
                for f in self.lief_binary.imported_functions:
                    if f.name:
                        name = f.name
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='ignore')
                        import_names.append(name)
                imports = " ".join(import_names)
        except Exception as e:
            logger.warning(f"Error extracting imports: {e}")
            
        try:
            if self.lief_binary.has_exports:
                export_names = []
                for f in self.lief_binary.exported_functions:
                    if f.name:
                        name = f.name
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='ignore')
                        export_names.append(name)
                exports = " ".join(export_names)
        except Exception as e:
            logger.warning(f"Error extracting exports: {e}")
            
        return {"imports": imports, "exports": exports}


class MalwareClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(MalwareClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x


class EnhancedNNMalwareModel:
    """Main model class compatible with the defender framework"""
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'Enhanced-NN-Malware-Detector'):
        self.thresh = thresh
        self.__name__ = name
        self.model_path = model_path
        
        # Initialize components
        self.feature_extractor = EnhancedPEFeatureExtractor()
        self.model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.is_loaded = False
        
        # Load pre-trained model if available
        if model_path:
            self._load_model_components()
    
    def _load_model_components(self):
        """Load the neural network model and preprocessing components"""
        try:
            model_dir = os.path.dirname(self.model_path) if self.model_path else os.path.join(os.path.dirname(__file__))
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load neural network model
            nn_model_path = os.path.join(model_dir, 'nn_model.pth')
            
            # Determine input dimension from the scaler and tfidf vectorizer
            numerical_features = len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else self.scaler.n_features_in_
            text_features = len(self.tfidf_vectorizer.get_feature_names_out())
            input_dim = numerical_features + text_features
            
            self.model = MalwareClassifierNN(input_dim)
            self.model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded enhanced NN model from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            self.is_loaded = False
    
    def _prepare_features(self, bytez: bytes) -> np.ndarray:
        """Prepare features for prediction"""
        # Extract features
        features = self.feature_extractor.extract(bytez)
        if not features:
            return None
        
        # Separate numerical and textual features
        numerical_part = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        textual_part = {k: v for k, v in features.items() if isinstance(v, str)}
        
        # Create DataFrame for numerical features
        df_numerical = pd.DataFrame([numerical_part])
        df_numerical = df_numerical.fillna(0)
        
        # Transform numerical features
        numerical_features_scaled = self.scaler.transform(df_numerical)
        
        # Transform textual features
        combined_text = " ".join(textual_part.values())
        text_features_tfidf = self.tfidf_vectorizer.transform([combined_text]).toarray()
        
        # Combine features
        combined_features = np.hstack((numerical_features_scaled, text_features_tfidf))
        
        return combined_features
    
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
            
            # Convert to tensor and make prediction
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            with torch.no_grad():
                output = self.model(features_tensor)
                prediction = (output.item() > self.thresh)
            
            result = int(prediction)
            logger.info(f"NN Prediction: {result} (confidence: {output.item():.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during NN prediction: {e}")
            return self._heuristic_prediction(bytez)
    
    def _heuristic_prediction(self, bytez: bytes) -> int:
        """Simple heuristic-based prediction as fallback"""
        try:
            # Basic checks
            if len(bytez) < 1024:
                return 1
            
            if not bytez.startswith(b'MZ'):
                return 1
            
            # Check for suspicious patterns
            suspicious_patterns = [
                b'cmd.exe', b'powershell', b'rundll32',
                b'regsvr32', b'certutil', b'bitsadmin',
                b'http://', b'https://', b'ftp://',
                b'keylog', b'steal', b'bypass', b'inject'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in bytez.lower():
                    return 1
            
            return 0
            
        except Exception:
            return 1
    
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": "Enhanced Neural Network malware detector",
            "architecture": "4-layer fully connected network with dropout",
            "features": "Comprehensive PE features + TF-IDF text features"
        }
