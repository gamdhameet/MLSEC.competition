"""
Data Loading Module for Combined Datasets

This module loads and combines features from:
1. Ember dataset (pre-extracted features in JSON)
2. DikeDataset (raw PE files requiring feature extraction)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmberDataLoader:
    """Load pre-extracted features from Ember dataset"""
    
    # Key features to extract from ember JSON
    EMBER_FEATURE_KEYS = [
        'general', 'header', 'section', 'imports', 'exports', 
        'histogram', 'byteentropy', 'strings'
    ]
    
    def __init__(self, ember_dir: str):
        self.ember_dir = Path(ember_dir)
        self.feature_files = []
        
        # Find all train_features_*.jsonl files
        for i in range(6):  # train_features_0.jsonl to train_features_5.jsonl
            feature_file = self.ember_dir / f"train_features_{i}.jsonl"
            if feature_file.exists():
                self.feature_files.append(feature_file)
        
        logger.info(f"Found {len(self.feature_files)} ember feature files")
    
    def _flatten_ember_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested ember features into a flat dictionary"""
        flat_features = {}
        
        # Extract label
        label = sample.get('label', -1)
        
        # General features
        if 'general' in sample:
            for key, value in sample['general'].items():
                flat_features[f'general_{key}'] = value
        
        # Header features
        if 'header' in sample and 'coff' in sample['header']:
            coff = sample['header']['coff']
            flat_features['header_timestamp'] = coff.get('timestamp', 0)
            flat_features['header_machine'] = 1 if coff.get('machine') == 'I386' else 0
            
        if 'header' in sample and 'optional' in sample['header']:
            opt = sample['header']['optional']
            flat_features['header_subsystem'] = 1 if 'GUI' in str(opt.get('subsystem', '')) else 0
            flat_features['header_magic'] = 1 if opt.get('magic') == 'PE32' else 0
            flat_features['header_major_linker_version'] = opt.get('major_linker_version', 0)
            flat_features['header_minor_linker_version'] = opt.get('minor_linker_version', 0)
            flat_features['header_sizeof_code'] = opt.get('sizeof_code', 0)
            flat_features['header_sizeof_headers'] = opt.get('sizeof_headers', 0)
        
        # Section features - aggregate statistics
        if 'section' in sample and 'sections' in sample['section']:
            sections = sample['section']['sections']
            if sections:
                flat_features['section_count'] = len(sections)
                flat_features['section_avg_entropy'] = np.mean([s.get('entropy', 0) for s in sections])
                flat_features['section_max_entropy'] = np.max([s.get('entropy', 0) for s in sections])
                flat_features['section_avg_size'] = np.mean([s.get('size', 0) for s in sections])
            else:
                flat_features['section_count'] = 0
                flat_features['section_avg_entropy'] = 0
                flat_features['section_max_entropy'] = 0
                flat_features['section_avg_size'] = 0
        
        # Histogram features - statistical summary
        if 'histogram' in sample:
            hist = sample['histogram']
            if hist:
                flat_features['histogram_mean'] = np.mean(hist)
                flat_features['histogram_std'] = np.std(hist)
                flat_features['histogram_max'] = np.max(hist)
                flat_features['histogram_min'] = np.min(hist)
        
        # Byte entropy features - statistical summary
        if 'byteentropy' in sample:
            entropy = sample['byteentropy']
            if entropy:
                flat_features['byteentropy_mean'] = np.mean(entropy)
                flat_features['byteentropy_std'] = np.std(entropy)
                flat_features['byteentropy_max'] = np.max(entropy)
        
        # String features
        if 'strings' in sample:
            strings = sample['strings']
            flat_features['strings_numstrings'] = strings.get('numstrings', 0)
            flat_features['strings_avlength'] = strings.get('avlength', 0)
            flat_features['strings_entropy'] = strings.get('entropy', 0)
            flat_features['strings_paths'] = strings.get('paths', 0)
            flat_features['strings_urls'] = strings.get('urls', 0)
            flat_features['strings_registry'] = strings.get('registry', 0)
            flat_features['strings_MZ'] = strings.get('MZ', 0)
        
        # Import count
        if 'imports' in sample and isinstance(sample['imports'], dict):
            flat_features['imports_count'] = sum(len(funcs) for funcs in sample['imports'].values())
            flat_features['imports_dll_count'] = len(sample['imports'])
        else:
            flat_features['imports_count'] = 0
            flat_features['imports_dll_count'] = 0
        
        # Export count
        if 'exports' in sample and isinstance(sample['exports'], list):
            flat_features['exports_count'] = len(sample['exports'])
        else:
            flat_features['exports_count'] = 0
        
        return flat_features, label
    
    def load_samples(self, max_samples: int = 100000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and flatten ember samples"""
        all_features = []
        all_labels = []
        
        total_loaded = 0
        
        for feature_file in self.feature_files:
            if total_loaded >= max_samples:
                break
            
            logger.info(f"Loading {feature_file.name}...")
            
            with open(feature_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if total_loaded >= max_samples:
                        break
                    
                    try:
                        sample = json.loads(line.strip())
                        features, label = self._flatten_ember_features(sample)
                        
                        # Only include labeled samples
                        if label in [0, 1]:
                            all_features.append(features)
                            all_labels.append(label)
                            total_loaded += 1
                            
                            if total_loaded % 10000 == 0:
                                logger.info(f"Loaded {total_loaded} samples...")
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num} in {feature_file.name}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
        
        logger.info(f"Total ember samples loaded: {total_loaded}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        df = df.fillna(0)  # Fill missing values
        
        return df, np.array(all_labels)


class DikeDataLoader:
    """Load and extract features from DikeDataset (raw PE files)"""
    
    def __init__(self, dike_dir: str, feature_extractor=None):
        self.dike_dir = Path(dike_dir)
        self.benign_dir = self.dike_dir / 'benign'
        self.malware_dir = self.dike_dir / 'malware'
        self.feature_extractor = feature_extractor
        
        # Count files
        self.benign_count = len(list(self.benign_dir.iterdir())) if self.benign_dir.exists() else 0
        self.malware_count = len(list(self.malware_dir.iterdir())) if self.malware_dir.exists() else 0
        
        logger.info(f"Dike dataset: {self.benign_count} benign, {self.malware_count} malware")
    
    def load_samples(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and extract features from DikeDataset PE files"""
        if self.feature_extractor is None:
            # Import here to avoid circular dependency
            from defender.models.reliable_nn_model import ReliablePEFeatureExtractor
            self.feature_extractor = ReliablePEFeatureExtractor()
        
        all_features = []
        all_labels = []
        
        # Process benign files
        logger.info("Processing benign files...")
        if self.benign_dir.exists():
            for i, file_path in enumerate(self.benign_dir.iterdir()):
                try:
                    with open(file_path, 'rb') as f:
                        bytez = f.read()
                        features = self.feature_extractor.extract(bytez)
                        
                        # Extract only numerical features
                        numerical_features = {k: v for k, v in features.items() 
                                            if isinstance(v, (int, float))}
                        
                        all_features.append(numerical_features)
                        all_labels.append(0)  # benign
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"Processed {i + 1} benign files...")
                            
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
        
        # Process malware files
        logger.info("Processing malware files...")
        if self.malware_dir.exists():
            for i, file_path in enumerate(self.malware_dir.iterdir()):
                try:
                    with open(file_path, 'rb') as f:
                        bytez = f.read()
                        features = self.feature_extractor.extract(bytez)
                        
                        # Extract only numerical features
                        numerical_features = {k: v for k, v in features.items() 
                                            if isinstance(v, (int, float))}
                        
                        all_features.append(numerical_features)
                        all_labels.append(1)  # malware
                        
                        if (i + 1) % 100 == 0:
                            logger.info(f"Processed {i + 1} malware files...")
                            
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Total Dike samples loaded: {len(all_features)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        df = df.fillna(0)
        
        return df, np.array(all_labels)


class CombinedDataLoader:
    """Combine Ember and DikeDataset"""
    
    def __init__(self, ember_dir: str, dike_dir: str):
        self.ember_loader = EmberDataLoader(ember_dir)
        self.dike_loader = DikeDataLoader(dike_dir)
    
    def load_combined(self, ember_max_samples: int = 100000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and combine both datasets"""
        logger.info("Loading Ember dataset...")
        ember_df, ember_labels = self.ember_loader.load_samples(max_samples=ember_max_samples)
        
        logger.info("Loading Dike dataset...")
        dike_df, dike_labels = self.dike_loader.load_samples()
        
        # Find common columns
        common_cols = list(set(ember_df.columns) & set(dike_df.columns))
        logger.info(f"Common features: {len(common_cols)}")
        
        if not common_cols:
            logger.warning("No common columns found, using all columns with alignment")
            # Align columns
            all_cols = list(set(ember_df.columns) | set(dike_df.columns))
            ember_df = ember_df.reindex(columns=all_cols, fill_value=0)
            dike_df = dike_df.reindex(columns=all_cols, fill_value=0)
        else:
            # Use only common columns for consistency
            ember_df = ember_df[common_cols]
            dike_df = dike_df[common_cols]
        
        # Combine datasets
        combined_df = pd.concat([ember_df, dike_df], ignore_index=True)
        combined_labels = np.concatenate([ember_labels, dike_labels])
        
        logger.info(f"Combined dataset: {len(combined_df)} samples, {len(combined_df.columns)} features")
        logger.info(f"Label distribution: {np.bincount(combined_labels)}")
        
        return combined_df, combined_labels

