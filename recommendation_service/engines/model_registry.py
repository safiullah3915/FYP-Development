"""
Model Registry for Two-Tower Recommendation Models
Manages model versions and loading from database
"""
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for managing recommendation model versions
    Handles model loading and version management
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model registry
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}  # Cache for loaded models
    
    def get_model_path(
        self, 
        model_name: str, 
        use_case: str = None,
        model_type: str = 'two_tower'
    ) -> Optional[Dict[str, str]]:
        """
        Get paths for model and encoder files
        
        Args:
            model_name: Name of the model (e.g., 'two_tower_v1')
            use_case: Use case filter (optional)
            model_type: Type of model (default: 'two_tower')
            
        Returns:
            Dictionary with 'model_path' and 'encoder_path', or None if not found
        """
        try:
            # Construct paths
            model_path = self.models_dir / f"{model_name}.pth"
            encoder_path = self.models_dir / f"{model_name}_encoder.pkl"
            config_path = self.models_dir / f"{model_name}_config.json"
            
            # Check if files exist
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            if not encoder_path.exists():
                logger.warning(f"Encoder file not found: {encoder_path}")
                return None
            
            logger.info(f"Found model: {model_name}")
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  Encoder path: {encoder_path}")
            
            return {
                'model_path': str(model_path),
                'encoder_path': str(encoder_path),
                'config_path': str(config_path) if config_path.exists() else None,
            }
            
        except Exception as e:
            logger.error(f"Error getting model path: {e}")
            return None
    
    def get_active_model(
        self,
        use_case: str = 'developer_startup',
        model_type: str = 'two_tower'
    ) -> Optional[Dict[str, str]]:
        """
        Get active model for a specific use case
        Currently returns the default model, but can be extended to query database
        
        Args:
            use_case: Use case (e.g., 'developer_startup')
            model_type: Model type (e.g., 'two_tower')
            
        Returns:
            Dictionary with model paths, or None if not found
        """
        try:
            # Try to load from database (future implementation)
            # For now, return default model based on naming convention
            model_name = f"{model_type}_{use_case}_v1"
            
            paths = self.get_model_path(model_name, use_case, model_type)
            
            # Fallback to generic model if use-case specific not found
            if not paths:
                logger.info(f"Use-case specific model not found, trying generic model")
                model_name = f"{model_type}_v1"
                paths = self.get_model_path(model_name, use_case, model_type)
            
            return paths
            
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None
    
    def list_available_models(self) -> list:
        """
        List all available models in the models directory
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return models
        
        # Find all .pth files
        for model_file in self.models_dir.glob("*.pth"):
            if '_latest' in model_file.name:
                continue  # Skip latest checkpoints
            
            model_name = model_file.stem
            encoder_path = self.models_dir / f"{model_name}_encoder.pkl"
            config_path = self.models_dir / f"{model_name}_config.json"
            
            model_info = {
                'name': model_name,
                'model_path': str(model_file),
                'encoder_path': str(encoder_path) if encoder_path.exists() else None,
                'config_path': str(config_path) if config_path.exists() else None,
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            }
            
            models.append(model_info)
        
        logger.info(f"Found {len(models)} models in {self.models_dir}")
        return models
    
    def load_model_metadata(self, model_name: str) -> Optional[Dict]:
        """
        Load model metadata from config file
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model metadata, or None
        """
        try:
            config_path = self.models_dir / f"{model_name}_config.json"
            
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return None
            
            import json
            with open(config_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            return None
    
    def validate_model(self, model_name: str) -> bool:
        """
        Validate that all required files exist for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if valid, False otherwise
        """
        paths = self.get_model_path(model_name)
        
        if not paths:
            return False
        
        required_files = ['model_path', 'encoder_path']
        
        for file_key in required_files:
            file_path = paths.get(file_key)
            if not file_path or not Path(file_path).exists():
                logger.warning(f"Missing required file: {file_key}")
                return False
        
        logger.info(f"Model {model_name} is valid")
        return True
    
    def get_fallback_model(self) -> Optional[Dict[str, str]]:
        """
        Get fallback model (content-based or default)
        
        Returns:
            Dictionary with model paths, or None
        """
        # For now, return None since we'll fallback to content-based
        # In future, could return a default two-tower model
        logger.info("Using content-based fallback")
        return None


# Global registry instance
_registry = None


def get_registry(models_dir: str = 'models') -> ModelRegistry:
    """
    Get global model registry instance
    
    Args:
        models_dir: Directory where models are stored
        
    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
    return _registry

