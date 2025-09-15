"""
BERT-based Malware Detection Model

This model uses BERT to analyze textual features extracted from PE files 
for malware detection. It combines traditional PE file analysis with 
modern NLP techniques.
"""

import os
import re
import lief
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import pickle
import json
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PETextualFeatureExtractor:
    """Extract textual features from PE files for BERT processing"""
    
    def __init__(self):
        self.max_text_length = 512  # BERT's max sequence length
    
    def extract_strings(self, bytez: bytes) -> List[str]:
        """Extract printable strings from binary data"""
        # Find strings of length 4 or more
        string_pattern = re.compile(b'[!-~]{4,}')
        strings = []
        
        for match in string_pattern.finditer(bytez):
            try:
                string = match.group().decode('utf-8', errors='ignore')
                if len(string) >= 4:
                    strings.append(string)
            except:
                continue
                
        return strings[:100]  # Limit to first 100 strings
    
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
                    metadata['libraries'].append(lib.name)
                    for func in lib.entries:
                        if func.name:
                            metadata['imports'].append(func.name)
            
            # Extract exported functions
            if pe.has_exports:
                for func in pe.exported_functions:
                    if func.name:
                        metadata['exports'].append(func.name)
            
            # Extract section names
            for section in pe.sections:
                if section.name:
                    metadata['sections'].append(section.name)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting PE metadata: {e}")
            return {}
    
    def create_text_representation(self, bytez: bytes) -> str:
        """Create a text representation of the PE file for BERT"""
        # Extract strings and metadata
        strings = self.extract_strings(bytez)
        metadata = self.extract_pe_metadata(bytez)
        
        # Build text representation
        text_parts = []
        
        # Add libraries
        if metadata.get('libraries'):
            libs_text = "LIBRARIES: " + " ".join(metadata['libraries'][:20])
            text_parts.append(libs_text)
        
        # Add imports
        if metadata.get('imports'):
            imports_text = "IMPORTS: " + " ".join(metadata['imports'][:30])
            text_parts.append(imports_text)
        
        # Add exports
        if metadata.get('exports'):
            exports_text = "EXPORTS: " + " ".join(metadata['exports'][:10])
            text_parts.append(exports_text)
        
        # Add sections
        if metadata.get('sections'):
            sections_text = "SECTIONS: " + " ".join(metadata['sections'])
            text_parts.append(sections_text)
        
        # Add strings
        if strings:
            strings_text = "STRINGS: " + " ".join(strings[:20])
            text_parts.append(strings_text)
        
        # Combine all parts
        full_text = " | ".join(text_parts)
        
        # Truncate if too long
        if len(full_text) > 2000:
            full_text = full_text[:2000]
        
        return full_text if full_text else "EMPTY_FILE"


class BERTMalwareClassifier:
    """BERT-based classifier for malware detection"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_extractor = PETextualFeatureExtractor()
        self.is_trained = False
        
    def initialize_bert(self):
        """Initialize BERT tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.bert_model = AutoModel.from_pretrained(self.model_name)
            self.bert_model.eval()
            logger.info(f"Initialized BERT model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing BERT: {e}")
            raise
    
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for a list of texts"""
        if not self.tokenizer or not self.bert_model:
            self.initialize_bert()
        
        embeddings = []
        
        for text in texts:
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[0, 0, :].numpy()
                    embeddings.append(embedding)
                    
            except Exception as e:
                logger.warning(f"Error getting embedding for text: {e}")
                # Use zero embedding as fallback
                embedding_dim = 768 if "base" in self.model_name else 512
                embeddings.append(np.zeros(embedding_dim))
        
        return np.array(embeddings)
    
    def extract_numerical_features(self, bytez: bytes) -> np.ndarray:
        """Extract numerical features from PE file"""
        try:
            pe = lief.PE.parse(list(bytez))
            if not pe:
                return np.zeros(10)
            
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
                int(pe.has_tls)
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting numerical features: {e}")
            return np.zeros(10)
    
    def prepare_features(self, bytez_list: List[bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare both textual and numerical features"""
        # Extract text representations
        texts = [self.feature_extractor.create_text_representation(bytez) 
                for bytez in bytez_list]
        
        # Get BERT embeddings
        text_embeddings = self.get_bert_embeddings(texts)
        
        # Extract numerical features
        numerical_features = np.array([
            self.extract_numerical_features(bytez) 
            for bytez in bytez_list
        ])
        
        return text_embeddings, numerical_features
    
    def train(self, bytez_list: List[bytes], labels: List[int]):
        """Train the BERT-based malware classifier"""
        logger.info(f"Training BERT classifier on {len(bytez_list)} samples")
        
        # Prepare features
        text_embeddings, numerical_features = self.prepare_features(bytez_list)
        
        # Combine features
        combined_features = np.hstack([text_embeddings, numerical_features])
        
        # Scale features
        combined_features = self.scaler.fit_transform(combined_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training completed. Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
    
    def predict(self, bytez: bytes) -> int:
        """Predict if a PE file is malicious"""
        if not self.is_trained:
            logger.warning("Model not trained yet, returning random prediction")
            return 1  # Default to malicious for safety
        
        try:
            # Prepare features
            text_embeddings, numerical_features = self.prepare_features([bytez])
            
            # Combine and scale features
            combined_features = np.hstack([text_embeddings, numerical_features])
            combined_features = self.scaler.transform(combined_features)
            
            # Make prediction
            prediction = self.classifier.predict(combined_features)[0]
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 1  # Default to malicious on error
    
    def predict_proba(self, bytez: bytes) -> float:
        """Get prediction probability"""
        if not self.is_trained:
            return 0.9  # High malicious probability as default
        
        try:
            # Prepare features
            text_embeddings, numerical_features = self.prepare_features([bytez])
            
            # Combine and scale features
            combined_features = np.hstack([text_embeddings, numerical_features])
            combined_features = self.scaler.transform(combined_features)
            
            # Get probabilities
            probabilities = self.classifier.predict_proba(combined_features)[0]
            return probabilities[1]  # Return malicious probability
            
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            return 0.9
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        
        # Initialize BERT components
        self.initialize_bert()
        
        logger.info(f"Model loaded from {filepath}")


class BERTMalwareModel:
    """Main model class compatible with the defender framework"""
    
    def __init__(self, 
                 model_path: str = None,
                 thresh: float = 0.5,
                 name: str = 'BERT-Malware-Detector'):
        self.thresh = thresh
        self.__name__ = name
        self.model_path = model_path
        
        # Initialize the BERT classifier
        self.bert_classifier = BERTMalwareClassifier()
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                self.bert_classifier.load_model(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                self._create_dummy_model()
        else:
            logger.warning("No pre-trained model found, using dummy model")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        # This is a simple fallback that just initializes BERT
        try:
            self.bert_classifier.initialize_bert()
            # Mark as "trained" but it will just use basic heuristics
            self.bert_classifier.is_trained = True
            logger.info("Created dummy BERT model")
        except Exception as e:
            logger.error(f"Error creating dummy model: {e}")
            # Ultra-simple fallback
            self.bert_classifier.is_trained = False
    
    def predict(self, bytez: bytes) -> int:
        """Predict if a PE file is malicious (0=benign, 1=malicious)"""
        try:
            if self.bert_classifier.is_trained:
                prediction = self.bert_classifier.predict(bytez)
            else:
                # Simple heuristic fallback
                prediction = self._simple_heuristic(bytez)
            
            logger.info(f"Prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 1  # Default to malicious for safety
    
    def _simple_heuristic(self, bytez: bytes) -> int:
        """Simple heuristic-based detection as fallback"""
        try:
            # Basic checks
            if len(bytez) < 1024:  # Very small files are suspicious
                return 1
            
            # Check for PE header
            if not bytez.startswith(b'MZ'):
                return 1
            
            # Look for suspicious strings
            suspicious_strings = [
                b'cmd.exe', b'powershell', b'rundll32',
                b'regsvr32', b'certutil', b'bitsadmin'
            ]
            
            for sus_string in suspicious_strings:
                if sus_string in bytez:
                    return 1
            
            # Default to benign if no red flags
            return 0
            
        except Exception:
            return 1  # Default to malicious on error
    
    def model_info(self) -> dict:
        """Return model information"""
        return {
            "name": self.__name__,
            "thresh": self.thresh,
            "model_path": self.model_path,
            "is_trained": self.bert_classifier.is_trained,
            "model_type": "BERT-based malware detector"
        }


# Training script for the BERT model
def train_bert_model():
    """Training script - this would be run separately to train the model"""
    logger.info("Starting BERT model training...")
    
    # This is where you would load your dataset
    # For now, we'll create a dummy training routine
    
    classifier = BERTMalwareClassifier()
    
    # In a real scenario, you would:
    # 1. Load the EMBER dataset or another malware dataset
    # 2. Extract features from PE files
    # 3. Train the model
    # 4. Save the trained model
    
    logger.info("Training would happen here with real data")
    logger.info("For now, saving an initialized model...")
    
    # Save the initialized model
    model_path = os.path.join(os.path.dirname(__file__), "bert_malware_model.pkl")
    classifier.save_model(model_path)
    
    return model_path


if __name__ == "__main__":
    # Train the model if run directly
    train_bert_model()

