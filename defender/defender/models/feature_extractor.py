"""
Extract features from PE files for malware detection
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import lief  # Better than pefile for feature extraction
    HAS_LIEF = True
except ImportError:
    HAS_LIEF = False
    logger.warning("LIEF not available, using basic extraction")

class PEFeatureExtractor:
    """Extract features from PE files compatible with EMBER 2024"""
    
    def __init__(self, expected_feature_count=1167):
        self.expected_feature_count = expected_feature_count
        
    def extract_features_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """
        Extract features from PE file bytes
        
        Args:
            file_bytes: Raw bytes of PE file
            
        Returns:
            numpy array of features (1D vector)
        """
        logger.info(f"Extracting features from PE file ({len(file_bytes)} bytes)")
        
        if HAS_LIEF:
            features = self._extract_with_lief(file_bytes)
        else:
            features = self._extract_basic(file_bytes)
        
        # Ensure correct feature count
        if len(features) != self.expected_feature_count:
            if len(features) < self.expected_feature_count:
                # Pad with zeros
                padded = np.zeros(self.expected_feature_count, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                # Truncate
                features = features[:self.expected_feature_count]
        
        return features
    
    def _extract_with_lief(self, file_bytes: bytes) -> np.ndarray:
        """Extract features using LIEF"""
        try:
            binary = lief.parse(list(file_bytes))
            if binary is None:
                logger.warning("LIEF failed to parse, using basic extraction")
                return self._extract_basic(file_bytes)
            
            features = []
            
            # Header features
            if binary.header:
                features.extend([
                    float(binary.header.machine),
                    float(binary.header.numberof_sections),
                    float(binary.header.time_date_stamps),
                    float(binary.header.pointerto_symbol_table),
                    float(binary.header.numberof_symbols),
                    float(binary.header.sizeof_optional_header),
                    float(binary.header.characteristics),
                ])
            
            # Optional header
            if binary.optional_header:
                oh = binary.optional_header
                features.extend([
                    float(oh.magic),
                    float(oh.sizeof_code),
                    float(oh.sizeof_initialized_data),
                    float(oh.sizeof_uninitialized_data),
                    float(oh.addressof_entrypoint),
                    float(oh.baseof_code),
                    float(oh.imagebase),
                    float(oh.section_alignment),
                    float(oh.file_alignment),
                ])
            
            # Section features (pad to 10 sections max)
            for i in range(10):
                if i < len(binary.sections):
                    section = binary.sections[i]
                    features.extend([
                        float(section.virtual_address),
                        float(section.virtual_size),
                        float(section.size),
                        float(section.characteristics),
                        float(section.entropy),
                    ])
                else:
                    features.extend([0.0] * 5)
            
            # Import features
            features.append(float(len(binary.imports)))
            
            # Export features
            features.append(float(len(binary.exported_functions)))
            
            # Pad to expected length
            while len(features) < self.expected_feature_count:
                features.append(0.0)
            
            return np.array(features[:self.expected_feature_count], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"LIEF extraction failed: {e}")
            return self._extract_basic(file_bytes)
    
    def _extract_basic(self, file_bytes: bytes) -> np.ndarray:
        """Basic feature extraction from raw bytes"""
        features = []
        
        # Basic byte statistics
        features.append(float(len(file_bytes)))
        
        # Byte value distribution (0-255)
        byte_counts = np.bincount(np.frombuffer(file_bytes, dtype=np.uint8), minlength=256)
        byte_probs = byte_counts / len(file_bytes)
        features.extend(byte_probs.astype(np.float32).tolist())
        
        # Entropy
        entropy = -np.sum(byte_probs[byte_probs > 0] * np.log2(byte_probs[byte_probs > 0]))
        features.append(float(entropy))
        
        # Pad to expected length
        while len(features) < self.expected_feature_count:
            features.append(0.0)
        
        return np.array(features[:self.expected_feature_count], dtype=np.float32)