#!/usr/bin/env python3
"""
Goodware Collection Script

This script helps collect benign PE files from Windows systems for training.
It can be run on a Windows machine to collect executables from system directories.
"""

import os
import shutil
import argparse
import hashlib
from pathlib import Path
from typing import List, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoodwareCollector:
    """Collect benign executables from Windows system directories"""
    
    # Common Windows system directories with benign executables
    DEFAULT_WINDOWS_DIRS = [
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
        r"C:\Program Files\Windows Defender",
        r"C:\Program Files\Windows NT",
        r"C:\Program Files\Common Files\Microsoft Shared",
        r"C:\Program Files (x86)\Common Files\Microsoft Shared",
    ]
    
    # File extensions to collect
    EXECUTABLE_EXTENSIONS = {'.exe', '.dll', '.sys', '.ocx'}
    
    # Minimum and maximum file sizes (to avoid very small or very large files)
    MIN_FILE_SIZE = 1024  # 1 KB
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    
    def __init__(self, output_dir: str, max_files: int = 5000):
        self.output_dir = Path(output_dir)
        self.max_files = max_files
        self.collected_hashes: Set[str] = set()
        self.file_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def is_valid_file(self, file_path: Path) -> bool:
        """Check if file meets criteria for collection"""
        try:
            # Check extension
            if file_path.suffix.lower() not in self.EXECUTABLE_EXTENSIONS:
                return False
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size < self.MIN_FILE_SIZE or file_size > self.MAX_FILE_SIZE:
                return False
            
            # Check if it's a PE file (starts with MZ)
            with open(file_path, 'rb') as f:
                header = f.read(2)
                if header != b'MZ':
                    return False
            
            return True
        except Exception as e:
            logger.debug(f"Error validating {file_path}: {e}")
            return False
    
    def collect_from_directory(self, source_dir: str) -> int:
        """Collect files from a specific directory"""
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Directory does not exist: {source_dir}")
            return 0
        
        collected = 0
        logger.info(f"Scanning: {source_dir}")
        
        try:
            for file_path in source_path.iterdir():
                if self.file_count >= self.max_files:
                    logger.info(f"Reached maximum file limit: {self.max_files}")
                    return collected
                
                if not file_path.is_file():
                    continue
                
                if not self.is_valid_file(file_path):
                    continue
                
                # Calculate hash to avoid duplicates
                file_hash = self.calculate_hash(file_path)
                if not file_hash or file_hash in self.collected_hashes:
                    continue
                
                # Copy file to output directory
                try:
                    output_filename = f"{file_hash[:16]}_{file_path.name}"
                    output_path = self.output_dir / output_filename
                    shutil.copy2(file_path, output_path)
                    
                    self.collected_hashes.add(file_hash)
                    self.file_count += 1
                    collected += 1
                    
                    if collected % 100 == 0:
                        logger.info(f"Collected {collected} files from {source_dir}")
                        
                except Exception as e:
                    logger.warning(f"Error copying {file_path}: {e}")
                    
        except PermissionError:
            logger.warning(f"Permission denied: {source_dir}")
        except Exception as e:
            logger.error(f"Error scanning {source_dir}: {e}")
        
        logger.info(f"Collected {collected} files from {source_dir}")
        return collected
    
    def collect_from_directories(self, directories: List[str]) -> int:
        """Collect files from multiple directories"""
        total_collected = 0
        
        for directory in directories:
            if self.file_count >= self.max_files:
                break
            collected = self.collect_from_directory(directory)
            total_collected += collected
        
        return total_collected
    
    def generate_report(self):
        """Generate a summary report"""
        logger.info("=" * 60)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files collected: {self.file_count}")
        logger.info(f"Unique files: {len(self.collected_hashes)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect benign PE files from Windows system directories"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='goodware_collected',
        help='Output directory for collected files (default: goodware_collected)'
    )
    parser.add_argument(
        '-d', '--directories',
        nargs='+',
        help='Custom directories to scan (default: Windows system directories)'
    )
    parser.add_argument(
        '-m', '--max-files',
        type=int,
        default=5000,
        help='Maximum number of files to collect (default: 5000)'
    )
    parser.add_argument(
        '--use-defaults',
        action='store_true',
        help='Use default Windows directories'
    )
    
    args = parser.parse_args()
    
    # Determine directories to scan
    if args.directories:
        scan_dirs = args.directories
    elif args.use_defaults:
        scan_dirs = GoodwareCollector.DEFAULT_WINDOWS_DIRS
    else:
        # Interactive mode
        print("Available options:")
        print("1. Use default Windows directories")
        print("2. Enter custom directories")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            scan_dirs = GoodwareCollector.DEFAULT_WINDOWS_DIRS
        else:
            custom_dirs = input("Enter directories (space-separated): ").strip().split()
            scan_dirs = custom_dirs if custom_dirs else GoodwareCollector.DEFAULT_WINDOWS_DIRS
    
    # Create collector and collect files
    collector = GoodwareCollector(args.output, args.max_files)
    
    logger.info(f"Starting collection from {len(scan_dirs)} directories...")
    total_collected = collector.collect_from_directories(scan_dirs)
    
    collector.generate_report()
    
    if total_collected == 0:
        logger.warning("No files were collected. Check permissions and paths.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

