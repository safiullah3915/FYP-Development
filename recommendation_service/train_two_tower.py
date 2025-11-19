#!/usr/bin/env python3
"""
Two-Tower Model Training Script
Standalone script for training the two-tower recommendation model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add current directory to path to avoid circular imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import directly from module files to avoid circular imports through __init__.py
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a module directly from file to avoid __init__.py imports"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules directly
training_config_module = load_module_from_file(
    "training_config",
    current_dir / "engines" / "training_config.py"
)
feature_engineering_module = load_module_from_file(
    "feature_engineering",
    current_dir / "engines" / "feature_engineering.py"
)
evaluation_module = load_module_from_file(
    "evaluation",
    current_dir / "engines" / "evaluation.py"
)
logger_module = load_module_from_file(
    "logger",
    current_dir / "utils" / "logger.py"
)

TwoTowerConfig = training_config_module.TwoTowerConfig
FeatureEncoder = feature_engineering_module.FeatureEncoder
DatasetProcessor = feature_engineering_module.DatasetProcessor
load_dataset_from_csv = feature_engineering_module.load_dataset_from_csv
RankingEvaluator = evaluation_module.RankingEvaluator
get_logger = logger_module.get_logger

logger = get_logger(__name__)


# ============================================================================
# Model Architecture (copied to avoid circular imports)
# ============================================================================

class Tower(nn.Module):
    """Generic tower (encoder) for user or item features"""
    
    def __init__(
        self, 
        input_dim: int, 
        embedding_dim: int,
        hidden_dims: list,
        dropout_rate: float = 0.3,
        dropout_rate_middle: float = 0.2
    ):
        super(Tower, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(nn.Dropout(dropout_rate_middle))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding = self.model(x)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Two-Tower model for recommendation"""
    
    def __init__(self, config: TwoTowerConfig, is_reverse: bool = False):
        super(TwoTowerModel, self).__init__()
        
        self.config = config
        self.is_reverse = is_reverse
        
        self.user_tower = Tower(
            input_dim=config.user_feature_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            dropout_rate_middle=config.dropout_rate_middle
        )
        
        self.startup_tower = Tower(
            input_dim=config.startup_feature_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            dropout_rate_middle=config.dropout_rate_middle
        )
    
    def forward(self, user_features: torch.Tensor, startup_features: torch.Tensor) -> torch.Tensor:
        # In reverse mode: user_features are actually startup features, startup_features are user features
        # But the towers are initialized with swapped dimensions, so we need to swap inputs
        if self.is_reverse:
            # Swap inputs: user_features (startup data) goes to startup_tower, startup_features (user data) goes to user_tower
            user_emb = self.user_tower(startup_features)  # User tower receives user features (from startup_features)
            startup_emb = self.startup_tower(user_features)  # Startup tower receives startup features (from user_features)
        else:
            user_emb = self.user_tower(user_features)
            startup_emb = self.startup_tower(startup_features)
        
        similarity = torch.sum(user_emb * startup_emb, dim=1)
        scores = torch.sigmoid(similarity)
        return scores
    
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)
    
    def encode_startups(self, startup_features: torch.Tensor) -> torch.Tensor:
        return self.startup_tower(startup_features)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss"""
    
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy(predictions, targets, reduction='none')
        weighted_bce = bce * weights
        return weighted_bce.mean()


# ============================================================================
# Dataset and Training
# ============================================================================

class TwoTowerDataset(Dataset):
    """PyTorch Dataset for Two-Tower model"""
    
    def __init__(self, user_features, startup_features, labels, weights):
        """
        Initialize dataset
        
        Args:
            user_features: User feature array (N, D_user)
            startup_features: Startup feature array (N, D_startup)
            labels: Label array (N,) - 0.0 to 1.0 based on interaction type
            weights: Weight array (N,) - base_weight * rank_weight
                     rank_weight = 1 / log2(rank + 1) corrects for exposure bias
        """
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.startup_features = torch.tensor(startup_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'user_features': self.user_features[idx],
            'startup_features': self.startup_features[idx],
            'labels': self.labels[idx],
            'weights': self.weights[idx],
        }


class Trainer:
    """Two-Tower model trainer"""
    
    def __init__(
        self, 
        model: TwoTowerModel,
        config: TwoTowerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize trainer
        
        Args:
            model: Two-Tower model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = WeightedBCELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.use_lr_scheduler:
            if config.lr_scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.num_epochs,
                    eta_min=config.learning_rate * 0.01
                )
            elif config.lr_scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=10,
                    gamma=0.5
                )
        else:
            self.scheduler = None
        
        # Evaluator
        self.evaluator = RankingEvaluator(k_values=config.metrics_k)
        
        # Training state
        self.best_val_ndcg = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            user_features = batch['user_features'].to(self.device)
            startup_features = batch['startup_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            weights = batch['weights'].to(self.device)
            
            # Forward pass
            # Weighted loss: weights include rank-based exposure correction
            # Hard negatives (high score but no interaction) are prioritized
            predictions = self.model(user_features, startup_features)
            loss = self.criterion(predictions, labels, weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                user_features = batch['user_features'].to(self.device)
                startup_features = batch['startup_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                weights = batch['weights'].to(self.device)
                
                # Forward pass
                predictions = self.model(user_features, startup_features)
                loss = self.criterion(predictions, labels, weights)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute ranking metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        metrics = self.evaluator.evaluate_batch(all_labels, all_predictions)
        
        return avg_loss, metrics
    
    def train(self) -> dict:
        """Train model for multiple epochs"""
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Device: {self.device}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            logger.info(f"  Train loss: {train_loss:.4f}")
            
            # Validate
            if epoch % self.config.eval_every_n_epochs == 0:
                val_loss, val_metrics = self.validate()
                self.val_losses.append(val_loss)
                self.val_metrics_history.append(val_metrics)
                
                logger.info(f"  Val loss: {val_loss:.4f}")
                logger.info(f"  Val metrics:")
                for metric_name, value in val_metrics.items():
                    logger.info(f"    {metric_name}: {value:.4f}")
                
                # Check for improvement
                val_ndcg = val_metrics.get('ndcg@10', 0.0)
                if val_ndcg > self.best_val_ndcg:
                    self.best_val_ndcg = val_ndcg
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"  New best model! NDCG@10: {val_ndcg:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
                logger.info(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        logger.info(f"\nTraining completed!")
        logger.info(f"  Best epoch: {self.best_epoch}")
        logger.info(f"  Best NDCG@10: {self.best_val_ndcg:.4f}")
        
        return {
            'best_epoch': self.best_epoch,
            'best_val_ndcg': self.best_val_ndcg,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics_history': self.val_metrics_history,
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        # Create models directory
        models_dir = Path(self.config.model_save_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'best_val_ndcg': self.best_val_ndcg,
        }
        
        if is_best:
            # For reverse models, save with _best suffix
            if 'reverse' in self.config.model_name:
                checkpoint_path = models_dir / f"{self.config.model_name}_best.pth"
            else:
                checkpoint_path = models_dir / f"{self.config.model_name}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"  Saved best model to {checkpoint_path}")
        
        # Also save latest
        latest_path = models_dir / f"{self.config.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Two-Tower recommendation model')
    parser.add_argument('--data', type=str, required=True, help='Path to training dataset CSV')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--model-name', type=str, default=None, help='Model name (auto-generated if not provided)')
    parser.add_argument('--use-case', type=str, default='developer_startup', 
                        choices=['developer_startup', 'investor_startup', 'startup_developer', 'startup_investor'],
                        help='Use case for training (default: developer_startup)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--model', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # Determine if this is a reverse use case
    reverse_use_cases = ['startup_developer', 'startup_investor']
    is_reverse = args.use_case in reverse_use_cases
    
    # Auto-generate model name if not provided
    if args.model_name is None:
        if is_reverse:
            args.model_name = f'two_tower_reverse_v1'
        else:
            args.model_name = f'two_tower_v1'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Use case: {args.use_case} (reverse: {is_reverse})")
    
    # Load configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TwoTowerConfig.load(args.config)
    else:
        config = TwoTowerConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            embedding_dim=args.embedding_dim,
            model_save_dir=args.output_dir,
            model_name=args.model_name,
        )
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data}")
    raw_data = load_dataset_from_csv(args.data)
    
    if len(raw_data) == 0:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(raw_data)} samples")
    
    # Process dataset
    logger.info("Processing dataset...")
    encoder = FeatureEncoder()
    processor = DatasetProcessor(encoder)
    
    train_data, val_data, test_data = processor.prepare_dataset(raw_data)
    
    # Update config with feature dimensions
    # For reverse: user features are actually startup features, startup features are user features
    if is_reverse:
        # For reverse use cases, swap feature dimensions
        config.user_feature_dim = encoder.get_startup_feature_dim()  # Startups as "users"
        config.startup_feature_dim = encoder.get_user_feature_dim()  # Users as "items"
        logger.info(f"Feature dimensions (reverse mode):")
        logger.info(f"  Entity (Startup): {config.user_feature_dim}")
        logger.info(f"  Item (User): {config.startup_feature_dim}")
    else:
        config.user_feature_dim = encoder.get_user_feature_dim()
        config.startup_feature_dim = encoder.get_startup_feature_dim()
        logger.info(f"Feature dimensions:")
        logger.info(f"  User: {config.user_feature_dim}")
        logger.info(f"  Startup: {config.startup_feature_dim}")
    
    # Save encoder
    encoder_path = Path(config.model_save_dir) / f"{config.model_name}_encoder.pkl"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Register module for pickle compatibility
    import sys
    sys.modules['engines.feature_engineering'] = feature_engineering_module
    sys.modules['feature_engineering'] = feature_engineering_module
    
    encoder.save(str(encoder_path))
    
    # Save config
    config_path = Path(config.model_save_dir) / f"{config.model_name}_config.json"
    config.save(str(config_path))
    
    if args.evaluate:
        # Evaluation mode
        if not args.model:
            logger.error("--model required for evaluation")
            return
        
        logger.info(f"Evaluating model from {args.model}")
        
        # Load model
        checkpoint = torch.load(args.model, map_location=device)
        # Determine if reverse from config or model name
        is_reverse_eval = 'reverse' in config.model_name or 'reverse' in str(args.model)
        model = TwoTowerModel(config, is_reverse=is_reverse_eval)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Create test dataset
        test_dataset = TwoTowerDataset(
            test_data['user_features'],
            test_data['startup_features'],
            test_data['labels'],
            test_data['weights']
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Evaluate
        evaluator = RankingEvaluator(k_values=config.metrics_k)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                user_features = batch['user_features'].to(device)
                startup_features = batch['startup_features'].to(device)
                labels = batch['labels']
                
                predictions = model(user_features, startup_features)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        metrics = evaluator.evaluate_batch(all_labels, all_predictions)
        
        logger.info("\nTest Set Evaluation:")
        for metric_name, value in sorted(metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save metrics
        metrics_path = Path(config.model_save_dir) / f"{config.model_name}_test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nMetrics saved to {metrics_path}")
        
    else:
        # Training mode
        logger.info("Creating data loaders...")
        
        train_dataset = TwoTowerDataset(
            train_data['user_features'],
            train_data['startup_features'],
            train_data['labels'],
            train_data['weights']
        )
        
        val_dataset = TwoTowerDataset(
            val_data['user_features'],
            val_data['startup_features'],
            val_data['labels'],
            val_data['weights']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Initialize model
        logger.info("Initializing model...")
        model = TwoTowerModel(config, is_reverse=is_reverse)
        
        # Initialize trainer
        trainer = Trainer(model, config, train_loader, val_loader, device)
        
        # Train
        training_results = trainer.train()
        
        # Save training history
        history_path = Path(config.model_save_dir) / f"{config.model_name}_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history = {
                'best_epoch': int(training_results['best_epoch']),
                'best_val_ndcg': float(training_results['best_val_ndcg']),
                'train_losses': [float(x) for x in training_results['train_losses']],
                'val_losses': [float(x) for x in training_results['val_losses']],
            }
            json.dump(history, f, indent=2)
        logger.info(f"\nTraining history saved to {history_path}")


if __name__ == '__main__':
    main()

