"""
Training Configuration for Two-Tower Model
Defines hyperparameters and training settings
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class TwoTowerConfig:
    """Configuration for Two-Tower model training"""
    
    # Model architecture
    user_feature_dim: int = 0  # Will be set dynamically
    startup_feature_dim: int = 0  # Will be set dynamically
    embedding_dim: int = 128
    hidden_dims: list = None  # [512, 256]
    dropout_rate: float = 0.3
    dropout_rate_middle: float = 0.2
    
    # Training hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 256
    num_epochs: int = 50
    early_stopping_patience: int = 5
    
    # Loss function
    loss_type: str = 'weighted_bce'  # 'weighted_bce' or 'warp'
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'cosine'  # 'cosine' or 'step'
    lr_warmup_epochs: int = 5
    
    # Evaluation
    eval_every_n_epochs: int = 1
    metrics_k: list = None  # [10, 20, 50]
    
    # Optimization
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = False
    
    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Paths
    model_save_dir: str = 'models'
    model_name: str = 'two_tower_v1'
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]
        if self.metrics_k is None:
            self.metrics_k = [10, 20, 50]
    
    def to_dict(self):
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_json(self):
        """Convert config to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str):
        """Create config from JSON string"""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())


# Default configurations for different scenarios
DEFAULT_CONFIG = TwoTowerConfig()

FAST_CONFIG = TwoTowerConfig(
    hidden_dims=[256, 128],
    batch_size=512,
    num_epochs=20,
    early_stopping_patience=3,
)

DEEP_CONFIG = TwoTowerConfig(
    hidden_dims=[1024, 512, 256],
    dropout_rate=0.4,
    dropout_rate_middle=0.3,
    batch_size=128,
    num_epochs=100,
    early_stopping_patience=10,
)

