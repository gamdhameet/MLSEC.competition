"""
Ensemble Malware Detection Model

Combines multiple models (Random Forest, Reliable NN, etc.) to achieve better
accuracy and meet FPR < 1% and FNR < 10% requirements.
"""

import os
import logging
import numpy as np
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleMalwareModel:
    """Ensemble model combining multiple malware detectors"""
    
    def __init__(self, 
                 model_weights: dict = None,
                 voting_strategy: str = 'weighted',
                 thresh: float = 0.5,
                 name: str = 'Ensemble-Malware-Detector'):
        """
        Initialize ensemble model
        
        Args:
            model_weights: Dictionary mapping model names to weights (e.g., {'rf': 0.5, 'nn': 0.5})
            voting_strategy: 'weighted', 'majority', or 'unanimous'
            thresh: Decision threshold (0.0-1.0)
            name: Model name
        """
        self.__name__ = name
        self.thresh = thresh
        self.voting_strategy = voting_strategy
        self.model_weights = model_weights or {'random_forest': 0.6, 'reliable_nn': 0.4}
        
        self.models = {}
        self.is_loaded = False
        
        # Load component models
        self._load_models()
    
    def _load_models(self):
        """Load all component models"""
        logger.info("Loading ensemble component models...")
        
        model_dir = os.path.dirname(__file__)
        
        # Load Random Forest
        try:
            from defender.models.random_forest_model import RandomForestMalwareModel
            rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
            if os.path.exists(rf_path):
                # Use optimized threshold from optimization
                optimal_thresh_path = os.path.join(model_dir, 'random_forest_optimal_threshold.pkl')
                if os.path.exists(optimal_thresh_path):
                    import pickle
                    with open(optimal_thresh_path, 'rb') as f:
                        optimal_info = pickle.load(f)
                        rf_thresh = optimal_info.get('threshold', 0.21)
                else:
                    rf_thresh = 0.21  # From optimization analysis
                
                self.models['random_forest'] = RandomForestMalwareModel(
                    model_path=rf_path, 
                    thresh=rf_thresh
                )
                logger.info(f"Loaded Random Forest (threshold: {rf_thresh})")
        except Exception as e:
            logger.warning(f"Could not load Random Forest: {e}")
        
        # Load Reliable NN
        try:
            from defender.models.reliable_nn_model import ReliableNNMalwareModel
            nn_path = os.path.join(model_dir, 'reliable_nn_model.pkl')
            if os.path.exists(nn_path):
                self.models['reliable_nn'] = ReliableNNMalwareModel(
                    model_path=nn_path,
                    thresh=0.3  # Lower threshold to reduce FPR
                )
                logger.info("Loaded Reliable NN (threshold: 0.3)")
        except Exception as e:
            logger.warning(f"Could not load Reliable NN: {e}")
        
        # Load NFS Behemot (if available)
        try:
            from defender.models.nfs_behemot_model import NFSBehemotModel
            self.models['nfs_behemot'] = NFSBehemotModel()
            logger.info("Loaded NFS Behemot")
        except Exception as e:
            logger.debug(f"Could not load NFS Behemot: {e}")
        
        # Load Ember (if available)
        try:
            from defender.models.ember_model import EmberModel
            self.models['ember'] = EmberModel()
            logger.info("Loaded Ember")
        except Exception as e:
            logger.debug(f"Could not load Ember: {e}")
        
        if len(self.models) > 0:
            self.is_loaded = True
            logger.info(f"Ensemble loaded with {len(self.models)} models: {list(self.models.keys())}")
        else:
            logger.error("No models could be loaded!")
            self.is_loaded = False
    
    def predict(self, bytez: bytes) -> int:
        """
        Make ensemble prediction
        
        Args:
            bytez: PE file bytes
            
        Returns:
            0 for benign, 1 for malicious
        """
        if not self.is_loaded:
            logger.warning("Ensemble not loaded, defaulting to malicious")
            return 1
        
        try:
            predictions = []
            confidences = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(bytez)
                    predictions.append((model_name, pred))
                    
                    # Try to get probability/confidence if available
                    if hasattr(model, '_prepare_features') and hasattr(model, 'model'):
                        try:
                            features = model._prepare_features(bytez)
                            if features is not None:
                                proba = model.model.predict_proba(features)[0][1]
                                confidences.append((model_name, proba))
                        except:
                            confidences.append((model_name, float(pred)))
                    else:
                        confidences.append((model_name, float(pred)))
                        
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if not predictions:
                logger.error("No predictions obtained from any model")
                return 1
            
            # Combine predictions based on strategy
            final_prediction = self._combine_predictions(predictions, confidences)
            
            logger.info(f"Ensemble prediction: {final_prediction} (from {len(predictions)} models)")
            return final_prediction
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 1
    
    def _combine_predictions(self, predictions: List[Tuple[str, int]], 
                           confidences: List[Tuple[str, float]]) -> int:
        """Combine predictions from multiple models"""
        
        if self.voting_strategy == 'majority':
            # Simple majority voting
            votes = [pred for _, pred in predictions]
            return 1 if sum(votes) > len(votes) / 2 else 0
        
        elif self.voting_strategy == 'unanimous':
            # All models must agree on malicious
            votes = [pred for _, pred in predictions]
            return 1 if all(v == 1 for v in votes) else 0
        
        elif self.voting_strategy == 'weighted':
            # Weighted voting based on model confidences and weights
            total_score = 0.0
            total_weight = 0.0
            
            for model_name, confidence in confidences:
                weight = self.model_weights.get(model_name, 1.0)
                total_score += confidence * weight
                total_weight += weight
            
            if total_weight == 0:
                return 1  # Default to malicious if no weights
            
            avg_score = total_score / total_weight
            prediction = 1 if avg_score > self.thresh else 0
            
            logger.debug(f"Weighted score: {avg_score:.4f}, threshold: {self.thresh}, prediction: {prediction}")
            return prediction
        
        else:
            # Default: majority voting
            votes = [pred for _, pred in predictions]
            return 1 if sum(votes) > len(votes) / 2 else 0
    
    def model_info(self) -> dict:
        """Return ensemble information"""
        return {
            "name": self.__name__,
            "voting_strategy": self.voting_strategy,
            "threshold": self.thresh,
            "num_models": len(self.models),
            "models": list(self.models.keys()),
            "model_weights": self.model_weights,
            "is_loaded": self.is_loaded
        }


class AdaptiveEnsembleModel(EnsembleMalwareModel):
    """
    Adaptive ensemble that adjusts weights based on confidence
    
    This model dynamically adjusts model weights based on:
    1. Individual model confidence
    2. Agreement between models
    3. File characteristics
    """
    
    def __init__(self, thresh: float = 0.5, name: str = 'Adaptive-Ensemble'):
        super().__init__(
            model_weights={'random_forest': 0.5, 'reliable_nn': 0.5},
            voting_strategy='adaptive',
            thresh=thresh,
            name=name
        )
    
    def _combine_predictions(self, predictions: List[Tuple[str, int]], 
                           confidences: List[Tuple[str, float]]) -> int:
        """Adaptive combination with confidence-based weighting"""
        
        if len(confidences) == 0:
            return 1
        
        # Calculate weighted average with adaptive weights
        scores = []
        for model_name, confidence in confidences:
            base_weight = self.model_weights.get(model_name, 1.0)
            
            # Boost weight if model is very confident
            confidence_boost = 1.0
            if confidence > 0.9 or confidence < 0.1:
                confidence_boost = 1.5
            elif confidence > 0.8 or confidence < 0.2:
                confidence_boost = 1.2
            
            adjusted_weight = base_weight * confidence_boost
            scores.append(confidence * adjusted_weight)
        
        # Normalize
        avg_score = sum(scores) / len(scores)
        
        # Check for unanimous agreement (all models agree)
        all_votes = [pred for _, pred in predictions]
        if len(set(all_votes)) == 1:  # All agree
            # Be more confident in unanimous decisions
            if all_votes[0] == 1:
                avg_score = max(avg_score, 0.7)
            else:
                avg_score = min(avg_score, 0.3)
        
        prediction = 1 if avg_score > self.thresh else 0
        
        logger.debug(f"Adaptive score: {avg_score:.4f}, threshold: {self.thresh}, prediction: {prediction}")
        return prediction

