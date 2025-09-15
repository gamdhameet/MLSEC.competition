"""
Reliable Neural Network Malware Detection Model

This model uses sklearn's MLPClassifier (neural network) which is more reliable
in Docker environments and doesn't have the numpy compatibility issues.
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

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReliablePEFeatureExtractor:
    """
    Extracts comprehensive features from PE files optimized for sklearn models.
    """

    def __init__(self):
        self.lief_binary = None

    def extract(self, bytez: bytes) -> Dict[str, Any]:
        """Extract all features from a PE file."""
        try:
            self.lief_binary = lief.PE.parse(list(bytez))
            if not self.lief_binary:
                return self._extract_fallback_features(bytez)
        except Exception as e:
            logger.warning(f"Could not parse PE file with LIEF: {e}")
            return self._extract_fallback_features(bytez)

        features = {}
        features.update(self._extract_general_features(bytez))
        features.update(self._extract_header_features())
        features.update(self._extract_optional_header_features())
        features.update(self._extract_string_metadata(bytez))
        features.update(self._extract_entropy(bytez))
        features.update(self._extract_imports_and_exports())

        return features

    def _extract_fallback_features(self, bytez: bytes) -> Dict[str, Any]:
        """Extract basic features when LIEF parsing fails."""
        features = {
            "size": len(bytez),
            "virtual_size": 0,
            "has_debug": 0,
            "has_relocations": 0,
            "has_resources": 0,
            "has_signature": 0,
            "has_tls": 0,
            "symbols": 0,
            "timestamp": 0,
            "machine": 0,
            "numberof_sections": 0,
            "numberof_symbols": 0,
            "pointerto_symbol_table": 0,
            "sizeof_optional_header": 0,
            "characteristics": 0,
            "baseof_code": 0,
            "baseof_data": 0,
            "dll_characteristics": 0,
            "file_alignment": 0,
            "imagebase": 0,
            "magic": 0,
            "major_image_version": 0,
            "minor_image_version": 0,
            "major_linker_version": 0,
            "minor_linker_version": 0,
            "major_operating_system_version": 0,
            "minor_operating_system_version": 0,
            "major_subsystem_version": 0,
            "minor_subsystem_version": 0,
            "numberof_rva_and_size": 0,
            "sizeof_code": 0,
            "sizeof_headers": 0,
            "sizeof_heap_commit": 0,
            "sizeof_image": 0,
            "sizeof_initialized_data": 0,
            "sizeof_uninitialized_data": 0,
            "subsystem": 0,
            "imports": "",
            "exports": ""
        }
        features.update(self._extract_string_metadata(bytez))
        features.update(self._extract_entropy(bytez))
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
            "symbols": len(self.lief_binary.symbols) if hasattr(self.lief_binary, 'symbols') else 0,
        }

    def _extract_header_features(self) -> Dict[str, Any]:
        header = self.lief_binary.header
        return {
            "timestamp": header.time_date_stamps,
            "machine": header.machine.value if hasattr(header.machine, 'value') else 0,
            "numberof_sections": header.numberof_sections,
            "numberof_symbols": header.numberof_symbols,
            "pointerto_symbol_table": header.pointerto_symbol_table,
            "sizeof_optional_header": header.sizeof_optional_header,
            "characteristics": header.characteristics,
        }

    def _extract_optional_header_features(self) -> Dict[str, Any]:
        try:
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
                "magic": opt_header.magic.value if hasattr(opt_header.magic, 'value') else 0,
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
                "subsystem": opt_header.subsystem.value if hasattr(opt_header.subsystem, 'value') else 0,
            }
        except Exception as e:
            logger.warning(f"Error extracting optional header features: {e}")
            return {
                "baseof_code": 0, "baseof_data": 0, "dll_characteristics": 0,
                "file_alignment": 0, "imagebase": 0, "magic": 0,
                "major_image_version": 0, "minor_image_version": 0,
                "major_linker_version": 0, "minor_linker_version": 0,
                "major_operating_system_version": 0, "minor_operating_system_version": 0,
                "major_subsystem_version": 0, "minor_subsystem_version": 0,
                "numberof_rva_and_size": 0, "sizeof_code": 0, "sizeof_headers": 0,
                "sizeof_heap_commit": 0, "sizeof_image": 0,
                "sizeof_initialized_data": 0, "sizeof_uninitialized_data": 0,
                "subsystem": 0,
            }

    def _extract_string_metadata(self, bytez: bytes) -> Dict[str, Any]:
        paths = re.compile(b'c:\\\\', re.IGNORECASE)
        urls = re.compile(b'https?://', re.IGNORECASE)
        registry = re.compile(b'HKEY_')
        mz = re.compile(b'MZ')
        pe_patterns = re.compile(b'(CreateFile|WriteFile|RegOpenKey|GetProcAddress|LoadLibrary)', re.IGNORECASE)
        
        return {
            'string_paths': len(paths.findall(bytez)),
            'string_urls': len(urls.findall(bytez)),
            'string_registry': len(registry.findall(bytez)),
            'string_MZ': len(mz.findall(bytez)),
            'string_pe_patterns': len(pe_patterns.findall(bytez))
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
                imports = " ".join(import_names[:50])  # Limit to avoid memory issues
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
                exports = " ".join(export_names[:50])  # Limit to avoid memory issues
        except Exception as e:
            logger.warning(f"Error extracting exports: {e}")
            
        return {"imports": imports, "exports": exports}


class ReliableNNMalwareModel:
    """Main model class compatible with the defender framework using sklearn MLPClassifier"""
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'Reliable-NN-Malware-Detector'):
        self.thresh = thresh
        self.__name__ = name
        self.model_path = model_path
        
        # Initialize components
        self.feature_extractor = ReliablePEFeatureExtractor()
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
            tfidf_path = os.path.join(model_dir, 'reliable_tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'reliable_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load neural network model
            nn_model_path = os.path.join(model_dir, 'reliable_nn_model.pkl')
            if os.path.exists(nn_model_path):
                with open(nn_model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            if self.model and self.tfidf_vectorizer and self.scaler:
                self.is_loaded = True
                logger.info(f"Successfully loaded reliable NN model from {model_dir}")
            else:
                logger.warning("Some model components missing, using heuristic-based detection")
                
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            self.is_loaded = False
    
    def _prepare_features(self, bytez: bytes) -> np.ndarray:
        """Prepare features for prediction"""
        try:
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
            
            logger.info(f"Reliable NN Prediction: {prediction} (confidence: {prediction_proba[1]:.4f}, threshold: {self.thresh})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during NN prediction: {e}")
            return self._heuristic_prediction(bytez)
    
    def _heuristic_prediction(self, bytez: bytes) -> int:
        """Improved heuristic-based prediction as fallback"""
        try:
            # Basic checks
            if len(bytez) < 1024:
                return 1
            
            if not bytez.startswith(b'MZ'):
                return 1
            
            # More sophisticated pattern matching
            malicious_score = 0
            
            # Check for suspicious patterns with weighted scoring
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
                (b'GetAsyncKeyState', 3), (b'FindWindow', 1),
                # Additional suspicious patterns
                (b'malware', 3), (b'virus', 3), (b'worm', 3),
                (b'rootkit', 4), (b'botnet', 4), (b'ransomware', 5),
                (b'cryptolocker', 5), (b'miner', 3), (b'coinminer', 4),
                (b'ntdll', 2), (b'kernel32', 1), (b'advapi32', 1),
                (b'LoadLibraryA', 2), (b'GetProcAddress', 2),
                (b'VirtualProtect', 2), (b'CreateProcess', 2),
                (b'RegCreateKey', 2), (b'RegSetValue', 2),
                (b'InternetOpen', 2), (b'URLDownload', 3),
                (b'WinExec', 2), (b'ShellExecute', 2)
            ]
            
            for pattern, weight in suspicious_patterns:
                if pattern in bytez.lower():
                    malicious_score += weight
            
            # Check entropy (high entropy might indicate packing/encryption)
            entropy = self._calculate_entropy(bytez)
            if entropy > 7.5:
                malicious_score += 2
            elif entropy > 7.0:
                malicious_score += 1
            
            # Size-based heuristics
            if len(bytez) < 10000:  # Very small files
                malicious_score += 1
            elif len(bytez) > 10000000:  # Very large files
                malicious_score += 1
            
            # Return prediction based on score
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
                entropy += - p_x * math.log(p_x, 2)
        return entropy
    
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": "Reliable sklearn MLPClassifier neural network",
            "architecture": "Multi-layer perceptron with optimized heuristic fallback",
            "features": "Comprehensive PE features + TF-IDF text features + improved heuristics"
        }
