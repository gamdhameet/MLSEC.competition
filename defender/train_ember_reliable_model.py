#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from defender.models.reliable_nn_model import ReliablePEFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_ember_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from ember dataset sample to match ReliablePEFeatureExtractor format
    """
    features = {}
    
    # Basic file information
    features['size'] = sample.get('general', {}).get('size', 0)
    features['virtual_size'] = sample.get('general', {}).get('vsize', 0)
    
    # General features
    general = sample.get('general', {})
    features['has_debug'] = int(general.get('has_debug', 0))
    features['has_relocations'] = int(general.get('has_relocations', 0))
    features['has_resources'] = int(general.get('has_resources', 0))
    features['has_signature'] = int(general.get('has_signature', 0))
    features['has_tls'] = int(general.get('has_tls', 0))
    features['symbols'] = general.get('symbols', 0)
    
    # Header features
    header = sample.get('header', {})
    coff = header.get('coff', {})
    optional = header.get('optional', {})
    
    features['timestamp'] = coff.get('timestamp', 0)
    features['machine'] = hash(str(coff.get('machine', ''))) % (2**31)  # Convert string to int
    features['numberof_sections'] = len(sample.get('section', {}).get('sections', []))
    features['numberof_symbols'] = 0  # Not available in ember
    features['pointerto_symbol_table'] = 0  # Not available in ember
    features['sizeof_optional_header'] = 0  # Not available in ember
    features['characteristics'] = len(coff.get('characteristics', []))
    
    # Optional header features
    features['baseof_code'] = 0  # Not directly available
    features['baseof_data'] = 0  # Not directly available
    features['dll_characteristics'] = len(optional.get('dll_characteristics', []))
    features['file_alignment'] = 0  # Not available
    features['imagebase'] = 0  # Not available
    features['magic'] = hash(str(optional.get('magic', ''))) % (2**31)
    features['major_image_version'] = optional.get('major_image_version', 0)
    features['minor_image_version'] = optional.get('minor_image_version', 0)
    features['major_linker_version'] = optional.get('major_linker_version', 0)
    features['minor_linker_version'] = optional.get('minor_linker_version', 0)
    features['major_operating_system_version'] = optional.get('major_operating_system_version', 0)
    features['minor_operating_system_version'] = optional.get('minor_operating_system_version', 0)
    features['major_subsystem_version'] = optional.get('major_subsystem_version', 0)
    features['minor_subsystem_version'] = optional.get('minor_subsystem_version', 0)
    features['numberof_rva_and_size'] = 0  # Not available
    features['sizeof_code'] = optional.get('sizeof_code', 0)
    features['sizeof_headers'] = optional.get('sizeof_headers', 0)
    features['sizeof_heap_commit'] = optional.get('sizeof_heap_commit', 0)
    features['sizeof_image'] = 0  # Not available
    features['sizeof_initialized_data'] = 0  # Not available
    features['sizeof_uninitialized_data'] = 0  # Not available
    features['subsystem'] = hash(str(optional.get('subsystem', ''))) % (2**31)
    
    # String metadata from ember strings section
    strings_info = sample.get('strings', {})
    features['string_paths'] = strings_info.get('paths', 0)
    features['string_urls'] = strings_info.get('urls', 0)
    features['string_registry'] = strings_info.get('registry', 0)
    features['string_MZ'] = strings_info.get('MZ', 0)
    features['string_pe_patterns'] = 0  # Not directly available
    
    # Entropy from ember
    features['entropy'] = strings_info.get('entropy', 0)
    
    # Histogram and byte entropy features (these are rich ember features)
    histogram = sample.get('histogram', [])
    byteentropy = sample.get('byteentropy', [])
    
    # Add statistical features from histogram and byteentropy
    if histogram:
        features['histogram_mean'] = np.mean(histogram)
        features['histogram_std'] = np.std(histogram)
        features['histogram_max'] = np.max(histogram)
        features['histogram_min'] = np.min(histogram)
    else:
        features['histogram_mean'] = features['histogram_std'] = 0
        features['histogram_max'] = features['histogram_min'] = 0
        
    if byteentropy:
        features['byteentropy_mean'] = np.mean(byteentropy)
        features['byteentropy_std'] = np.std(byteentropy)
        features['byteentropy_max'] = np.max(byteentropy)
        features['byteentropy_min'] = np.min(byteentropy)
    else:
        features['byteentropy_mean'] = features['byteentropy_std'] = 0
        features['byteentropy_max'] = features['byteentropy_min'] = 0
    
    # Imports and exports as text features
    imports = sample.get('imports', {})
    exports = sample.get('exports', [])
    
    import_text = " ".join([f"{lib}:{func}" for lib, funcs in imports.items() for func in funcs])
    export_text = " ".join(exports) if exports else ""
    
    features['imports'] = import_text
    features['exports'] = export_text
    
    return features


def load_system_goodware(system_dirs: List[str], max_samples: int = 10000) -> tuple:
    """
    Load goodware samples from system directories (like /usr/bin, /bin, etc.)
    """
    feature_extractor = ReliablePEFeatureExtractor()
    goodware_features = []
    goodware_labels = []
    
    logger.info("Loading system goodware samples...")
    
    count = 0
    for system_dir in system_dirs:
        if not os.path.exists(system_dir):
            logger.warning(f"System directory not found: {system_dir}")
            continue
            
        logger.info(f"Processing system directory: {system_dir}")
        
        for root, dirs, files in os.walk(system_dir):
            if count >= max_samples:
                break
                
            for file in files:
                if count >= max_samples:
                    break
                    
                file_path = os.path.join(root, file)
                
                # Skip symbolic links and non-executable files
                if os.path.islink(file_path):
                    continue
                    
                try:
                    # Check if file is executable and reasonable size
                    stat_info = os.stat(file_path)
                    if stat_info.st_size < 1024 or stat_info.st_size > 50 * 1024 * 1024:  # 1KB to 50MB
                        continue
                        
                    with open(file_path, 'rb') as f:
                        bytez = f.read()
                        
                    # Check if it looks like a PE file (starts with MZ) or ELF
                    if not (bytez.startswith(b'MZ') or bytez.startswith(b'\x7fELF')):
                        continue
                        
                    # Extract features using the reliable feature extractor
                    features = feature_extractor.extract(bytez)
                    if features:
                        goodware_features.append(features)
                        goodware_labels.append(0)  # 0 = benign
                        count += 1
                        
                        if count % 100 == 0:
                            logger.info(f"Processed {count} goodware samples...")
                            
                except Exception as e:
                    # Skip files that can't be read or processed
                    continue
    
    logger.info(f"Loaded {len(goodware_features)} goodware samples from system")
    return goodware_features, goodware_labels


def load_ember_dataset(ember_dir: str, max_samples_per_file: int = None) -> tuple:
    """
    Load ember dataset from JSONL files
    """
    train_files = [
        'train_features_0.jsonl',
        'train_features_1.jsonl', 
        'train_features_2.jsonl',
        'train_features_3.jsonl',
        'train_features_4.jsonl',
        'train_features_5.jsonl'
    ]
    
    all_features = []
    all_labels = []
    
    logger.info("Loading ember dataset...")
    
    for train_file in train_files:
        file_path = os.path.join(ember_dir, train_file)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {train_file}...")
        
        with open(file_path, 'r') as f:
            count = 0
            for line in tqdm(f, desc=f"Loading {train_file}"):
                if max_samples_per_file and count >= max_samples_per_file:
                    break
                    
                try:
                    sample = json.loads(line.strip())
                    if 'label' not in sample:
                        continue
                        
                    features = extract_ember_features(sample)
                    all_features.append(features)
                    all_labels.append(sample['label'])
                    count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue
    
    logger.info(f"Loaded {len(all_features)} samples total")
    return all_features, all_labels


def train_ember_reliable_model(ember_dir: str = '/home/gamdhameet/ember_dataset/ember',
                              max_samples_per_file: int = 5000,  # Reduced to balance with goodware
                              max_goodware_samples: int = 15000):  # More goodware samples
    """Train the reliable sklearn-based neural network model using ember dataset + system goodware"""
    
    # --- 1. Data Loading ---
    logger.info("Loading ember malware dataset...")
    ember_features, ember_labels = load_ember_dataset(ember_dir, max_samples_per_file)
    
    # Load system goodware
    system_dirs = [
        '/usr/bin',
        '/bin', 
        '/usr/sbin',
        '/sbin',
        '/usr/local/bin',
        '/usr/lib',
        '/lib',
        '/usr/lib64',
        '/lib64'
    ]
    
    logger.info("Loading system goodware...")
    goodware_features, goodware_labels = load_system_goodware(system_dirs, max_goodware_samples)
    
    # Combine ember and goodware data
    logger.info("Combining datasets...")
    all_features = ember_features + goodware_features
    labels = ember_labels + goodware_labels
    
    logger.info(f"Dataset composition:")
    logger.info(f"  Ember malware samples: {len(ember_features)}")
    logger.info(f"  System goodware samples: {len(goodware_features)}")
    logger.info(f"  Total samples: {len(all_features)}")
    
    if len(all_features) == 0:
        logger.error("No samples loaded!")
        return None
    
    # --- 2. Feature Processing ---
    logger.info("Processing features...")
    
    # Separate numerical and textual features
    numerical_features = []
    textual_features = []
    
    for features in all_features:
        numerical_part = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        textual_part = {k: v for k, v in features.items() if isinstance(v, str)}
        
        numerical_features.append(numerical_part)
        # Combine all textual features into one string for TF-IDF
        combined_text = " ".join(textual_part.values())
        textual_features.append(combined_text)
    
    # Convert to DataFrame and handle missing values
    df_numerical = pd.DataFrame(numerical_features)
    df_numerical = df_numerical.fillna(0)
    
    logger.info(f"Numerical features shape: {df_numerical.shape}")
    logger.info(f"Textual features count: {len(textual_features)}")
    
    # TF-IDF for textual features
    logger.info("Computing TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=300, min_df=2, max_df=0.95, ngram_range=(1,2))
    text_features_tfidf = tfidf.fit_transform(textual_features).toarray()
    
    # Scale numerical features
    logger.info("Scaling numerical features...")
    scaler_numerical = StandardScaler()
    numerical_features_scaled = scaler_numerical.fit_transform(df_numerical)
    
    # Combine features
    combined_features = np.hstack((numerical_features_scaled, text_features_tfidf))
    
    logger.info(f"Total feature dimension: {combined_features.shape[1]}")
    
    # Check and clean labels
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    logger.info(f"Unique labels found: {unique_labels}")
    
    # Filter out any invalid labels (keep only 0 and 1)
    valid_mask = np.isin(labels, [0, 1])
    if not np.all(valid_mask):
        logger.warning(f"Found {np.sum(~valid_mask)} invalid labels, filtering them out")
        combined_features = combined_features[valid_mask]
        labels = labels[valid_mask]
    
    logger.info(f"Final dataset size: {len(labels)}")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # --- 3. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, np.array(labels), test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- 4. Model Training ---
    logger.info("Starting model training...")
    
    # Use a simpler approach to avoid memory issues
    # Train with good default parameters
    best_model = MLPClassifier(
        hidden_layer_sizes=(150, 75),
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )
    
    logger.info("Training neural network...")
    best_model.fit(X_train, y_train)
    
    logger.info("Training completed!")
    
    # --- 5. Model Evaluation ---
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Calculate FPR and FNR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    logger.info(f"False Positive Rate: {fpr:.4f}")
    logger.info(f"False Negative Rate: {fnr:.4f}")

    # --- 6. Save Model and Preprocessors ---
    model_dir = os.path.join(os.path.dirname(__file__), 'defender/models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(model_dir, 'reliable_nn_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
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
            'hidden_layer_sizes': (150, 75),
            'alpha': 0.001,
            'learning_rate': 'adaptive'
        },
        'samples_used': len(all_features),
        'malware_samples': len(ember_features),
        'goodware_samples': len(goodware_features)
    }


if __name__ == "__main__":
    results = train_ember_reliable_model()
    if results:
        print(f"\nFinal Results:")
        print(f"Total samples used: {results['samples_used']}")
        print(f"  - Malware samples: {results['malware_samples']}")
        print(f"  - Goodware samples: {results['goodware_samples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"FPR: {results['fpr']:.4f}")
        print(f"FNR: {results['fnr']:.4f}")
        print(f"Model Parameters: {results['model_params']}")
    else:
        print("Training failed!")
