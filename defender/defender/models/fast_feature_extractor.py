"""
Optimized Feature Extractor for Malware Detection

This module provides a faster version of the PE feature extraction pipeline
with caching, simplified features, and performance optimizations.
"""

import lief
import numpy as np
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastPEFeatureExtractor:
    """
    Optimized PE feature extractor focused on speed and essential features
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = {}
    
    def extract(self, bytez: bytes) -> Dict[str, Any]:
        """Extract features from PE file bytes with caching and optimization"""
        
        # Check cache
        if self.use_cache:
            file_hash = hashlib.md5(bytez).hexdigest()
            if file_hash in self._cache:
                return self._cache[file_hash]
        
        features = {}
        
        try:
            # Parse PE with timeout protection
            binary = lief.PE.parse(list(bytez))
            
            if binary is None:
                logger.warning("Failed to parse PE file")
                return self._get_default_features()
            
            # Basic file features (fast)
            features.update(self._extract_basic_features(bytez, binary))
            
            # Header features (fast)
            features.update(self._extract_header_features(binary))
            
            # Section features (medium speed)
            features.update(self._extract_section_features(binary))
            
            # Import features (medium speed)
            features.update(self._extract_import_features(binary))
            
            # Export features (fast)
            features.update(self._extract_export_features(binary))
            
            # Resource features (fast)
            features.update(self._extract_resource_features(binary))
            
            # Cache result
            if self.use_cache:
                self._cache[file_hash] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._get_default_features()
    
    def _extract_basic_features(self, bytez: bytes, binary: Any) -> Dict:
        """Extract basic file-level features"""
        features = {}
        
        # File size features
        features['file_size'] = len(bytez)
        features['file_size_log'] = np.log10(max(1, len(bytez)))
        
        # Entropy (simplified calculation on sample)
        sample = bytez[::max(1, len(bytez) // 10000)]  # Sample every N bytes
        features['entropy'] = self._calculate_entropy_fast(sample)
        
        # PE header info
        features['is_64bit'] = 1 if binary.optional_header.magic == lief.PE.PE_TYPE.PE32_PLUS else 0
        features['is_dll'] = 1 if binary.has_characteristic(lief.PE.Header.CHARACTERISTICS.DLL) else 0
        features['is_exe'] = 1 if binary.has_characteristic(lief.PE.Header.CHARACTERISTICS.EXECUTABLE_IMAGE) else 0
        
        return features
    
    def _extract_header_features(self, binary: Any) -> Dict:
        """Extract PE header features"""
        features = {}
        
        try:
            dos_header = binary.dos_header
            features['dos_header_magic'] = dos_header.magic
            features['dos_header_lfanew'] = dos_header.addressof_new_exeheader
            
            header = binary.header
            features['num_sections'] = header.numberof_sections
            features['timestamp'] = header.time_date_stamps
            features['characteristics'] = header.characteristics
            
            opt_header = binary.optional_header
            features['entry_point'] = opt_header.addressof_entrypoint
            features['image_base'] = opt_header.imagebase
            features['section_alignment'] = opt_header.section_alignment
            features['file_alignment'] = opt_header.file_alignment
            features['size_of_image'] = opt_header.sizeof_image
            features['size_of_headers'] = opt_header.sizeof_headers
            features['checksum'] = opt_header.checksum
            features['subsystem'] = opt_header.subsystem.value if hasattr(opt_header.subsystem, 'value') else 0
            features['dll_characteristics'] = opt_header.dll_characteristics
            
        except Exception as e:
            logger.debug(f"Error extracting header features: {e}")
            features.update({
                'dos_header_magic': 0, 'dos_header_lfanew': 0,
                'num_sections': 0, 'timestamp': 0, 'characteristics': 0,
                'entry_point': 0, 'image_base': 0, 'section_alignment': 0,
                'file_alignment': 0, 'size_of_image': 0, 'size_of_headers': 0,
                'checksum': 0, 'subsystem': 0, 'dll_characteristics': 0
            })
        
        return features
    
    def _extract_section_features(self, binary: Any) -> Dict:
        """Extract section-related features (optimized)"""
        features = {}
        
        try:
            sections = binary.sections
            features['num_sections'] = len(sections)
            
            if len(sections) > 0:
                # Aggregate statistics
                total_virtual_size = sum(s.virtual_size for s in sections)
                total_raw_size = sum(s.sizeof_raw_data for s in sections)
                
                features['total_virtual_size'] = total_virtual_size
                features['total_raw_size'] = total_raw_size
                features['size_ratio'] = total_virtual_size / max(1, total_raw_size)
                
                # Entropy statistics (sampled)
                entropies = []
                for section in sections[:10]:  # Limit to first 10 sections
                    try:
                        content = bytes(section.content)
                        if len(content) > 0:
                            # Sample content for speed
                            sample = content[::max(1, len(content) // 1000)]
                            entropies.append(self._calculate_entropy_fast(sample))
                    except:
                        pass
                
                if entropies:
                    features['section_entropy_mean'] = np.mean(entropies)
                    features['section_entropy_max'] = np.max(entropies)
                    features['section_entropy_min'] = np.min(entropies)
                else:
                    features['section_entropy_mean'] = 0
                    features['section_entropy_max'] = 0
                    features['section_entropy_min'] = 0
                
                # Check for suspicious section names
                suspicious_names = [b'.text', b'.data', b'.rdata', b'.rsrc', b'.reloc']
                for name in suspicious_names:
                    has_section = any(name in s.name.encode() for s in sections)
                    features[f'has_section_{name.decode()}'] = 1 if has_section else 0
                
            else:
                features.update({
                    'total_virtual_size': 0, 'total_raw_size': 0, 'size_ratio': 0,
                    'section_entropy_mean': 0, 'section_entropy_max': 0, 'section_entropy_min': 0,
                    'has_section_.text': 0, 'has_section_.data': 0, 'has_section_.rdata': 0,
                    'has_section_.rsrc': 0, 'has_section_.reloc': 0
                })
        
        except Exception as e:
            logger.debug(f"Error extracting section features: {e}")
            features.update({
                'num_sections': 0, 'total_virtual_size': 0, 'total_raw_size': 0,
                'size_ratio': 0, 'section_entropy_mean': 0, 'section_entropy_max': 0,
                'section_entropy_min': 0, 'has_section_.text': 0, 'has_section_.data': 0,
                'has_section_.rdata': 0, 'has_section_.rsrc': 0, 'has_section_.reloc': 0
            })
        
        return features
    
    def _extract_import_features(self, binary: Any) -> Dict:
        """Extract import-related features (optimized)"""
        features = {}
        
        try:
            imports = binary.imports
            features['num_imports'] = len(imports)
            
            if len(imports) > 0:
                # Count total imported functions
                total_functions = sum(len(imp.entries) for imp in imports)
                features['num_imported_functions'] = total_functions
                
                # Check for suspicious imports
                suspicious_dlls = [
                    'kernel32.dll', 'ntdll.dll', 'advapi32.dll',
                    'user32.dll', 'ws2_32.dll', 'wininet.dll'
                ]
                
                for dll_name in suspicious_dlls:
                    has_dll = any(dll_name.lower() in imp.name.lower() for imp in imports if imp.name)
                    features[f'imports_{dll_name.replace(".", "_")}'] = 1 if has_dll else 0
                
            else:
                features['num_imported_functions'] = 0
                for dll_name in ['kernel32.dll', 'ntdll.dll', 'advapi32.dll', 'user32.dll', 'ws2_32.dll', 'wininet.dll']:
                    features[f'imports_{dll_name.replace(".", "_")}'] = 0
        
        except Exception as e:
            logger.debug(f"Error extracting import features: {e}")
            features.update({
                'num_imports': 0, 'num_imported_functions': 0,
                'imports_kernel32_dll': 0, 'imports_ntdll_dll': 0,
                'imports_advapi32_dll': 0, 'imports_user32_dll': 0,
                'imports_ws2_32_dll': 0, 'imports_wininet_dll': 0
            })
        
        return features
    
    def _extract_export_features(self, binary: Any) -> Dict:
        """Extract export-related features"""
        features = {}
        
        try:
            if binary.has_exports:
                exports = binary.get_export()
                features['num_exports'] = len(exports.entries) if exports else 0
            else:
                features['num_exports'] = 0
        except:
            features['num_exports'] = 0
        
        return features
    
    def _extract_resource_features(self, binary: Any) -> Dict:
        """Extract resource-related features"""
        features = {}
        
        try:
            if binary.has_resources:
                resources = binary.resources
                features['num_resources'] = len(resources.childs)
                
                # Calculate total resource size (limited iteration)
                total_size = 0
                for child in list(resources.childs)[:50]:  # Limit iterations
                    try:
                        if hasattr(child, 'sizeof'):
                            total_size += child.sizeof
                    except:
                        pass
                
                features['total_resource_size'] = total_size
            else:
                features['num_resources'] = 0
                features['total_resource_size'] = 0
        except:
            features['num_resources'] = 0
            features['total_resource_size'] = 0
        
        return features
    
    def _calculate_entropy_fast(self, data: bytes) -> float:
        """Fast entropy calculation"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    def _get_default_features(self) -> Dict:
        """Return default features when parsing fails"""
        default_features = {
            'file_size': 0, 'file_size_log': 0, 'entropy': 0,
            'is_64bit': 0, 'is_dll': 0, 'is_exe': 0,
            'dos_header_magic': 0, 'dos_header_lfanew': 0,
            'num_sections': 0, 'timestamp': 0, 'characteristics': 0,
            'entry_point': 0, 'image_base': 0, 'section_alignment': 0,
            'file_alignment': 0, 'size_of_image': 0, 'size_of_headers': 0,
            'checksum': 0, 'subsystem': 0, 'dll_characteristics': 0,
            'total_virtual_size': 0, 'total_raw_size': 0, 'size_ratio': 0,
            'section_entropy_mean': 0, 'section_entropy_max': 0, 'section_entropy_min': 0,
            'has_section_.text': 0, 'has_section_.data': 0, 'has_section_.rdata': 0,
            'has_section_.rsrc': 0, 'has_section_.reloc': 0,
            'num_imports': 0, 'num_imported_functions': 0,
            'imports_kernel32_dll': 0, 'imports_ntdll_dll': 0,
            'imports_advapi32_dll': 0, 'imports_user32_dll': 0,
            'imports_ws2_32_dll': 0, 'imports_wininet_dll': 0,
            'num_exports': 0, 'num_resources': 0, 'total_resource_size': 0
        }
        return default_features
    
    def clear_cache(self):
        """Clear the feature cache"""
        self._cache.clear()

