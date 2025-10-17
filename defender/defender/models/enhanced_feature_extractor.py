"""
Enhanced PE Feature Extractor with Advanced Malware Detection Features

This module implements comprehensive feature extraction including:
1. PE Import/Export Analysis with suspicious function detection
2. String Analysis for IP addresses, domains, registry keys
3. Section-wise Entropy Features
4. Opcode N-grams (disassembly-based features)
"""

import os
import re
import lief
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPEFeatureExtractor:
    """
    Comprehensive PE feature extractor with advanced malware detection capabilities.
    """
    
    # Suspicious Windows API functions commonly used by malware
    SUSPICIOUS_IMPORTS = {
        # Process/Thread manipulation
        'CreateRemoteThread': 5, 'WriteProcessMemory': 5, 'VirtualAllocEx': 4,
        'OpenProcess': 3, 'CreateProcess': 2, 'TerminateProcess': 2,
        'SetWindowsHookEx': 4, 'GetAsyncKeyState': 4, 'SetThreadContext': 4,
        'QueueUserAPC': 4, 'ResumeThread': 3, 'SuspendThread': 3,
        
        # Memory manipulation
        'VirtualAlloc': 3, 'VirtualProtect': 4, 'VirtualProtectEx': 4,
        'HeapAlloc': 1, 'MapViewOfFile': 3, 'VirtualQuery': 2,
        
        # Registry manipulation
        'RegCreateKey': 2, 'RegSetValue': 2, 'RegOpenKey': 2, 
        'RegDeleteKey': 3, 'RegSetValueEx': 2, 'RegOpenKeyEx': 2,
        
        # File operations
        'CreateFile': 1, 'WriteFile': 1, 'ReadFile': 1, 'DeleteFile': 2,
        'MoveFile': 2, 'CopyFile': 1, 'FindFirstFile': 1,
        
        # Network operations
        'InternetOpen': 2, 'InternetConnect': 2, 'HttpOpenRequest': 2,
        'InternetReadFile': 2, 'URLDownloadToFile': 4, 'send': 2, 'recv': 2,
        'WSAStartup': 2, 'socket': 2, 'connect': 2, 'bind': 2,
        
        # Cryptography
        'CryptAcquireContext': 2, 'CryptEncrypt': 3, 'CryptDecrypt': 3,
        'CryptCreateHash': 2, 'CryptHashData': 2,
        
        # DLL manipulation
        'LoadLibrary': 2, 'LoadLibraryEx': 2, 'GetProcAddress': 3,
        'FreeLibrary': 1, 'GetModuleHandle': 2, 'GetModuleFileName': 1,
        
        # Debugging/Anti-debugging
        'IsDebuggerPresent': 3, 'CheckRemoteDebuggerPresent': 3,
        'OutputDebugString': 2, 'DebugActiveProcess': 4,
        
        # Service manipulation
        'CreateService': 3, 'StartService': 3, 'ControlService': 3,
        'DeleteService': 3, 'OpenService': 2,
        
        # Shell/System
        'WinExec': 3, 'ShellExecute': 3, 'system': 4, 'ShellExecuteEx': 3,
        
        # Privilege escalation
        'AdjustTokenPrivileges': 4, 'LookupPrivilegeValue': 3,
        'OpenProcessToken': 3,
    }
    
    # DLLs commonly imported by malware
    SUSPICIOUS_DLLS = {
        'ntdll.dll': 2, 'kernel32.dll': 1, 'advapi32.dll': 2, 
        'ws2_32.dll': 2, 'wininet.dll': 3, 'urlmon.dll': 4,
        'user32.dll': 1, 'shell32.dll': 2, 'psapi.dll': 2,
    }

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
        features.update(self._extract_section_features())
        features.update(self._extract_string_analysis(bytez))
        features.update(self._extract_entropy_features(bytez))
        features.update(self._extract_import_export_analysis())
        features.update(self._extract_resource_features())
        features.update(self._extract_opcode_features(bytez))

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
        }
        features.update(self._extract_string_analysis(bytez))
        features.update(self._extract_entropy_features(bytez))
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
            return {k: 0 for k in ["baseof_code", "baseof_data", "dll_characteristics", "file_alignment", 
                                   "imagebase", "magic", "major_image_version", "minor_image_version",
                                   "major_linker_version", "minor_linker_version", "major_operating_system_version",
                                   "minor_operating_system_version", "major_subsystem_version", "minor_subsystem_version",
                                   "numberof_rva_and_size", "sizeof_code", "sizeof_headers", "sizeof_heap_commit",
                                   "sizeof_image", "sizeof_initialized_data", "sizeof_uninitialized_data", "subsystem"]}

    def _extract_section_features(self) -> Dict[str, Any]:
        """Extract section-wise entropy and characteristics."""
        features = {}
        
        try:
            sections = self.lief_binary.sections
            if not sections:
                return {f'section_{key}': 0 for key in ['count', 'avg_entropy', 'max_entropy', 'min_entropy',
                                                         'text_entropy', 'data_entropy', 'rsrc_entropy',
                                                         'executable_sections', 'writable_sections']}
            
            entropies = []
            text_entropy = 0
            data_entropy = 0
            rsrc_entropy = 0
            executable_count = 0
            writable_count = 0
            
            for section in sections:
                # Get section entropy
                entropy = section.entropy
                entropies.append(entropy)
                
                # Check section name
                name = section.name.lower() if section.name else ''
                if '.text' in name or 'code' in name:
                    text_entropy = entropy
                elif '.data' in name:
                    data_entropy = entropy
                elif '.rsrc' in name or 'resource' in name:
                    rsrc_entropy = entropy
                
                # Check section characteristics
                characteristics = section.characteristics
                if characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                    executable_count += 1
                if characteristics & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                    writable_count += 1
            
            features.update({
                'section_count': len(sections),
                'section_avg_entropy': np.mean(entropies) if entropies else 0,
                'section_max_entropy': np.max(entropies) if entropies else 0,
                'section_min_entropy': np.min(entropies) if entropies else 0,
                'section_text_entropy': text_entropy,
                'section_data_entropy': data_entropy,
                'section_rsrc_entropy': rsrc_entropy,
                'section_executable_count': executable_count,
                'section_writable_count': writable_count,
            })
            
        except Exception as e:
            logger.warning(f"Error extracting section features: {e}")
            features = {f'section_{key}': 0 for key in ['count', 'avg_entropy', 'max_entropy', 'min_entropy',
                                                         'text_entropy', 'data_entropy', 'rsrc_entropy',
                                                         'executable_count', 'writable_count']}
        
        return features

    def _extract_string_analysis(self, bytez: bytes) -> Dict[str, Any]:
        """Enhanced string analysis for IP addresses, domains, registry keys, file paths."""
        # IP address patterns
        ip_pattern = re.compile(rb'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        
        # URL patterns
        url_pattern = re.compile(rb'(?:https?|ftp)://[^\s]+', re.IGNORECASE)
        
        # Domain patterns
        domain_pattern = re.compile(rb'[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(?:\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+', re.IGNORECASE)
        
        # Registry key patterns
        registry_pattern = re.compile(rb'(?:HKEY_|HKLM|HKCU|HKCR|HKU|HKCC)', re.IGNORECASE)
        
        # File path patterns
        path_pattern = re.compile(rb'[a-zA-Z]:\\\\[^\x00-\x1f]+', re.IGNORECASE)
        
        # PE header pattern
        mz_pattern = re.compile(rb'MZ')
        
        # Suspicious command patterns
        cmd_patterns = re.compile(rb'(?:cmd\.exe|powershell|rundll32|regsvr32|wscript|cscript)', re.IGNORECASE)
        
        # Common API patterns
        api_patterns = re.compile(rb'(?:CreateFile|WriteFile|RegOpenKey|GetProcAddress|LoadLibrary|VirtualAlloc|CreateProcess|InternetOpen)', re.IGNORECASE)
        
        return {
            'string_ip_count': len(ip_pattern.findall(bytez)),
            'string_url_count': len(url_pattern.findall(bytez)),
            'string_domain_count': len(domain_pattern.findall(bytez)),
            'string_registry_count': len(registry_pattern.findall(bytez)),
            'string_path_count': len(path_pattern.findall(bytez)),
            'string_MZ_count': len(mz_pattern.findall(bytez)),
            'string_cmd_count': len(cmd_patterns.findall(bytez)),
            'string_api_count': len(api_patterns.findall(bytez)),
        }

    def _extract_entropy_features(self, bytez: bytes) -> Dict[str, Any]:
        """Calculate overall entropy and entropy distribution."""
        if not bytez:
            return {'entropy_overall': 0, 'entropy_std': 0, 'entropy_high_blocks': 0}
        
        # Overall entropy
        entropy_overall = 0
        byte_counts = Counter(bytez)
        total = len(bytez)
        
        for count in byte_counts.values():
            p_x = count / total
            if p_x > 0:
                entropy_overall += -p_x * math.log(p_x, 2)
        
        # Block-wise entropy (divide file into 256-byte blocks)
        block_size = 256
        block_entropies = []
        high_entropy_blocks = 0
        
        for i in range(0, len(bytez), block_size):
            block = bytez[i:i+block_size]
            if len(block) < block_size // 2:  # Skip very small blocks
                continue
            
            block_entropy = 0
            block_counts = Counter(block)
            block_total = len(block)
            
            for count in block_counts.values():
                p_x = count / block_total
                if p_x > 0:
                    block_entropy += -p_x * math.log(p_x, 2)
            
            block_entropies.append(block_entropy)
            if block_entropy > 7.0:  # High entropy threshold
                high_entropy_blocks += 1
        
        return {
            'entropy_overall': entropy_overall,
            'entropy_std': np.std(block_entropies) if block_entropies else 0,
            'entropy_high_blocks': high_entropy_blocks,
            'entropy_block_mean': np.mean(block_entropies) if block_entropies else 0,
        }

    def _extract_import_export_analysis(self) -> Dict[str, Any]:
        """Enhanced import/export analysis with suspicious function detection."""
        features = {
            'import_dll_count': 0,
            'import_func_count': 0,
            'import_suspicious_score': 0,
            'import_suspicious_count': 0,
            'export_func_count': 0,
            'import_has_kernel32': 0,
            'import_has_ntdll': 0,
            'import_has_wininet': 0,
            'import_has_ws2_32': 0,
        }
        
        try:
            # Import analysis
            if self.lief_binary.has_imports:
                import_names = []
                dll_names = []
                suspicious_score = 0
                suspicious_count = 0
                
                for import_entry in self.lief_binary.imports:
                    dll_name = import_entry.name.lower() if import_entry.name else ''
                    dll_names.append(dll_name)
                    
                    # Check for suspicious DLLs
                    for sus_dll, score in self.SUSPICIOUS_DLLS.items():
                        if sus_dll in dll_name:
                            suspicious_score += score
                            break
                    
                    # Check imported functions
                    for func_entry in import_entry.entries:
                        if func_entry.name:
                            func_name = func_entry.name
                            if isinstance(func_name, bytes):
                                func_name = func_name.decode('utf-8', errors='ignore')
                            
                            import_names.append(func_name)
                            
                            # Check for suspicious functions
                            for sus_func, score in self.SUSPICIOUS_IMPORTS.items():
                                if sus_func.lower() in func_name.lower():
                                    suspicious_score += score
                                    suspicious_count += 1
                                    break
                
                features['import_dll_count'] = len(dll_names)
                features['import_func_count'] = len(import_names)
                features['import_suspicious_score'] = suspicious_score
                features['import_suspicious_count'] = suspicious_count
                
                # Check for specific DLLs
                features['import_has_kernel32'] = int('kernel32.dll' in dll_names)
                features['import_has_ntdll'] = int('ntdll.dll' in dll_names)
                features['import_has_wininet'] = int('wininet.dll' in dll_names)
                features['import_has_ws2_32'] = int('ws2_32.dll' in dll_names)
        
        except Exception as e:
            logger.warning(f"Error extracting imports: {e}")
        
        try:
            # Export analysis
            if self.lief_binary.has_exports:
                export_count = 0
                for export_entry in self.lief_binary.exported_functions:
                    if export_entry.name:
                        export_count += 1
                
                features['export_func_count'] = export_count
        
        except Exception as e:
            logger.warning(f"Error extracting exports: {e}")
        
        return features

    def _extract_resource_features(self) -> Dict[str, Any]:
        """Extract features from PE resources."""
        features = {
            'resource_count': 0,
            'resource_total_size': 0,
            'resource_avg_entropy': 0,
        }
        
        try:
            if self.lief_binary.has_resources:
                # Get resource node (may be a directory)
                resource_node = self.lief_binary.resources
                
                # Try to count resources by iterating through childs
                resource_count = 0
                if hasattr(resource_node, 'childs'):
                    try:
                        resource_count = len(list(resource_node.childs))
                    except:
                        resource_count = 1
                
                features['resource_count'] = resource_count
                # Note: Cannot easily extract size/entropy from resource directory
                features['resource_total_size'] = 0
                features['resource_avg_entropy'] = 0
        
        except Exception as e:
            # Silently handle resource extraction errors
            pass
        
        return features

    def _extract_opcode_features(self, bytez: bytes) -> Dict[str, Any]:
        """
        Extract opcode-based features (simplified without full disassembly).
        Looks for common opcode patterns in the code section.
        """
        features = {
            'opcode_call_count': 0,
            'opcode_jmp_count': 0,
            'opcode_push_count': 0,
            'opcode_mov_count': 0,
            'opcode_nop_count': 0,
            'opcode_int_count': 0,
        }
        
        try:
            # Common x86 opcodes
            # CALL: E8, FF (with ModR/M)
            # JMP: E9, EB, FF
            # PUSH: 50-57, 68, 6A
            # MOV: 88, 89, 8A, 8B, 8C, 8D, 8E
            # NOP: 90
            # INT: CD
            
            # Count approximate opcode occurrences
            features['opcode_call_count'] = bytez.count(b'\xE8')  # CALL relative
            features['opcode_jmp_count'] = bytez.count(b'\xE9') + bytez.count(b'\xEB')  # JMP
            features['opcode_push_count'] = sum(bytez.count(bytes([i])) for i in range(0x50, 0x58))  # PUSH reg
            features['opcode_mov_count'] = sum(bytez.count(bytes([i])) for i in range(0x88, 0x8F))  # MOV variants
            features['opcode_nop_count'] = bytez.count(b'\x90')  # NOP
            features['opcode_int_count'] = bytez.count(b'\xCD')  # INT (interrupts)
            
        except Exception as e:
            logger.warning(f"Error extracting opcode features: {e}")
        
        return features

