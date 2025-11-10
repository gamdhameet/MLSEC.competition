import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class EmberDataLoader:
    """Load EMBER 2024 dataset from numpy files"""
    
    def __init__(self, ember_dir: str):
        self.ember_dir = ember_dir
        
    def load_ember(self, max_samples: int = None):
        """
        Load EMBER 2024 dataset from .npy files
        
        Args:
            max_samples: Maximum number of samples to load (None = all)
            
        Returns:
            X: Features DataFrame
            y: Labels array (0=benign, 1=malware)
        """
        logger.info(f"Loading EMBER 2024 dataset from {self.ember_dir}")
        
        # Load numpy arrays
        X = np.load(os.path.join(self.ember_dir, "X_train.npy"))
        y = np.load(os.path.join(self.ember_dir, "y_train.npy"))
        
        logger.info(f"Loaded {len(y)} samples with {X.shape[1]} features")
        logger.info(f"Benign (0): {np.sum(y == 0)}")
        logger.info(f"Malware (1): {np.sum(y == 1)}")
        
        # Limit samples if specified
        if max_samples and len(y) > max_samples:
            logger.info(f"Sampling {max_samples} samples...")
            from sklearn.model_selection import train_test_split
            X, _, y, _ = train_test_split(
                X, y, 
                train_size=max_samples, 
                stratify=y, 
                random_state=42
            )
            logger.info(f"After sampling - Benign: {np.sum(y == 0)}, Malware: {np.sum(y == 1)}")
        
        # Convert to DataFrame for compatibility with your training script
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y


class CombinedDataLoader:
    """Loader for combined datasets (currently just EMBER 2024)"""
    
    def __init__(self, ember_dir: str, dike_dir: str = None):
        self.ember_loader = EmberDataLoader(ember_dir)
        self.dike_dir = dike_dir
        
    def load_combined(self, ember_max_samples: int = None):
        """Load EMBER 2024 dataset"""
        logger.info("Loading EMBER 2024 dataset")
        if self.dike_dir:
            logger.warning("DikeDataset not implemented yet, loading EMBER only")
        return self.ember_loader.load_ember(max_samples=ember_max_samples)