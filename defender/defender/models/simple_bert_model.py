"""
Simplified BERT-inspired Malware Detection Model

This is a simplified version that uses basic text processing and machine learning
without requiring the full BERT infrastructure, making it more suitable for 
Docker deployment.
"""

import os
import re
import lief
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import pickle
import json
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePETextExtractor:
    """Extract and process textual features from PE files"""
    
    def __init__(self):
        self.max_features = 1000
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    def extract_strings(self, bytez: bytes) -> List[str]:
        """Extract printable strings from binary data"""
        string_pattern = re.compile(b'[!-~]{4,}')
        strings = []
        
        for match in string_pattern.finditer(bytez):
            try:
                string = match.group().decode('utf-8', errors='ignore')
                if len(string) >= 4:
                    strings.append(string.lower())
            except:
                continue
                
        return strings[:50]  # Limit to first 50 strings
    
    def extract_pe_metadata(self, bytez: bytes) -> Dict[str, Any]:
        """Extract metadata from PE file using LIEF"""
        try:
            pe = lief.PE.parse(list(bytez))
            if not pe:
                return {}
            
            metadata = {
                'imports': [],
                'exports': [],
                'sections': [],
                'libraries': []
            }
            
            # Extract imported functions
            if pe.has_imports:
                for lib in pe.imports:
                    if lib.name:
                        metadata['libraries'].append(lib.name.lower())
                    for func in lib.entries:
                        if func.name:
                            metadata['imports'].append(func.name.lower())
            
            # Extract exported functions
            if pe.has_exports:
                for func in pe.exported_functions:
                    if func.name:
                        metadata['exports'].append(func.name.lower())
            
            # Extract section names
            for section in pe.sections:
                if section.name:
                    metadata['sections'].append(section.name.lower())
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting PE metadata: {e}")
            return {}
    
    def create_text_corpus(self, bytez: bytes) -> str:
        """Create a text corpus from PE file for TF-IDF processing"""
        # Extract strings and metadata
        strings = self.extract_strings(bytez)
        metadata = self.extract_pe_metadata(bytez)
        
        # Build text corpus
        text_parts = []
        
        # Add all extracted text
        text_parts.extend(strings)
        text_parts.extend(metadata.get('libraries', []))
        text_parts.extend(metadata.get('imports', []))
        text_parts.extend(metadata.get('exports', []))
        text_parts.extend(metadata.get('sections', []))
        
        # Join all parts
        corpus = ' '.join(text_parts)
        
        return corpus if corpus else "empty_file"


class SimpleBERTMalwareClassifier:
    """Simplified BERT-inspired classifier using TF-IDF and Random Forest"""
    
    def __init__(self):
        self.text_extractor = SimplePETextExtractor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1  # Single thread for Docker
        )
        self.is_trained = False
    
    def extract_numerical_features(self, bytez: bytes) -> np.ndarray:
        """Extract numerical features from PE file"""
        try:
            pe = lief.PE.parse(list(bytez))
            if not pe:
                return np.zeros(15)
            
            features = [
                len(bytez),  # File size
                pe.virtual_size,
                len(pe.imports) if pe.has_imports else 0,
                len(pe.exported_functions) if pe.has_exports else 0,
                len(pe.sections),
                int(pe.has_debug),
                int(pe.has_relocations),
                int(pe.has_resources),
                int(pe.has_signature),
                int(pe.has_tls),
                pe.header.numberof_sections,
                pe.header.numberof_symbols,
                pe.optional_header.sizeof_code,
                pe.optional_header.sizeof_headers,
                pe.optional_header.sizeof_image
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting numerical features: {e}")
            return np.zeros(15)
    
    def prepare_features(self, bytez_list: List[bytes], fit_transform: bool = False) -> np.ndarray:
        """Prepare combined textual and numerical features"""
        # Extract text corpora
        text_corpora = [self.text_extractor.create_text_corpus(bytez) 
                       for bytez in bytez_list]
        
        # Process text features
        if fit_transform:
            text_features = self.tfidf_vectorizer.fit_transform(text_corpora).toarray()
        else:
            text_features = self.tfidf_vectorizer.transform(text_corpora).toarray()
        
        # Extract numerical features
        numerical_features = np.array([
            self.extract_numerical_features(bytez) 
            for bytez in bytez_list
        ])
        
        # Combine features
        combined_features = np.hstack([text_features, numerical_features])
        
        return combined_features
    
    def train(self, bytez_list: List[bytes], labels: List[int]):
        """Train the classifier"""
        logger.info(f"Training simple BERT-inspired classifier on {len(bytez_list)} samples")
        
        # Prepare features
        features = self.prepare_features(bytez_list, fit_transform=True)
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training completed. Test accuracy: {accuracy:.4f}")
        
        self.is_trained = True
    
    def predict(self, bytez: bytes) -> int:
        """Predict if a PE file is malicious"""
        if not self.is_trained:
            logger.warning("Model not trained, using heuristics")
            return self._heuristic_prediction(bytez)
        
        try:
            # Prepare features
            features = self.prepare_features([bytez], fit_transform=False)
            features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.classifier.predict(features)[0]
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
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
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")


class SimpleBERTMalwareModel:
    """Main model class compatible with the defender framework"""
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'Simple-BERT-Malware-Detector'):
        self.thresh = thresh
        self.__name__ = name
        self.model_path = model_path
        
        # Initialize the classifier
        self.classifier = SimpleBERTMalwareClassifier()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                self.classifier.load_model(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                logger.info("Using heuristic-based detection")
        else:
            logger.info("No pre-trained model found, using heuristic-based detection")
    
    def predict(self, bytez: bytes) -> int:
        """Predict if a PE file is malicious (0=benign, 1=malicious)"""
        try:
            prediction = self.classifier.predict(bytez)
            logger.info(f"Prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 1  # Default to malicious for safety
    
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_trained": self.classifier.is_trained,
            "model_type": "Simple BERT-inspired malware detector"
        }


def train_simple_model():
    """Training function for the simple model"""
    # This would use the same synthetic data as before
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from train_bert_model import create_synthetic_dataset, load_dataset
    
    # Create synthetic data
    dataset_path = create_synthetic_dataset("data", 200)
    
    # Load data
    bytez_list, labels = load_dataset(dataset_path)
    
    # Train model
    classifier = SimpleBERTMalwareClassifier()
    classifier.train(bytez_list, labels)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "simple_bert_model.pkl")
    classifier.save_model(model_path)
    
    return model_path


if __name__ == "__main__":
    train_simple_model()

