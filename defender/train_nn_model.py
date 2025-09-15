import os
import re
import lief
import math
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPEFeatureExtractor:
    """
    Extracts a comprehensive set of features from PE files.
    This class is inspired by the feature extractors found in the project,
    but is adapted for use with a PyTorch neural network.
    """

    def __init__(self):
        self.lief_binary = None

    def extract(self, bytez: bytes) -> Dict[str, Any]:
        """Extract all features from a PE file."""
        try:
            self.lief_binary = lief.PE.parse(list(bytez))
            if not self.lief_binary:
                return {}
        except lief.bad_file as e:
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
        return {
            "baseof_code": opt_header.baseof_code,
            "baseof_data": opt_header.baseof_data,
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


class PEDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def train_model():
    # --- 1. Data Loading and Preparation ---
    malware_dir = '/home/gamdhameet/DikeDataset-main/files/malware'
    benign_dir = '/home/gamdhameet/DikeDataset-main/files/benign'

    malware_files = [os.path.join(malware_dir, f) for f in os.listdir(malware_dir)]
    benign_files = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir)]

    files = malware_files + benign_files
    labels = [1] * len(malware_files) + [0] * len(benign_files)

    # --- 2. Feature Extraction ---
    feature_extractor = EnhancedPEFeatureExtractor()
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
                    
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files...")
                
        except Exception as e:
            logger.warning(f"Error processing file {f}: {e}")
            continue
    
    logger.info(f"Successfully extracted features from {len(all_features)} files")
    labels = valid_labels


    df_numerical = pd.DataFrame(all_features)
    df_numerical = df_numerical.fillna(0) # Handle missing values

    # --- 3. Feature Vectorization and Scaling ---
    tfidf = TfidfVectorizer(max_features=500)
    text_features_tfidf = tfidf.fit_transform(textual_features).toarray()

    scaler_numerical = StandardScaler()
    numerical_features_scaled = scaler_numerical.fit_transform(df_numerical)

    combined_features = np.hstack((numerical_features_scaled, text_features_tfidf))
    
    # --- 4. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, np.array(labels), test_size=0.2, random_state=42, stratify=labels
    )

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = PEDataset(X_train_tensor, y_train_tensor)
    test_dataset = PEDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    # --- 5. Model Training ---
    input_dim = combined_features.shape[1]
    model = MalwareClassifierNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 25
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # --- 6. Model Evaluation ---
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            outputs = model(features)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    logger.info(f'Test Accuracy: {accuracy:.2f}%')

    # --- 7. Save Model and Preprocessors ---
    model_dir = os.path.join(os.path.dirname(__file__), 'defender/models')
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'nn_model.pth'))
    
    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_numerical, f)
        
    logger.info(f"Model and preprocessors saved in {model_dir}")


if __name__ == "__main__":
    train_model()
