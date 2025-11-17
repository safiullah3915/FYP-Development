"""
Ranker Training Script
Trains neural ranker using explicit feedback
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from engines.ranker import RankerMLP, PairwiseRankingLoss
import time


class RankerDataset(Dataset):
    """Dataset for ranker training"""
    
    def __init__(self, data_path):
        """
        Load ranker training data
        
        Expected CSV columns:
        - user_id, startup_id, model_score, recency_score, popularity_score, diversity_score, label
        """
        self.data = pd.read_csv(data_path)
        
        # Features
        if 'original_score' not in self.data.columns:
            self.data['original_score'] = self.data['model_score']
        self.data['original_score'] = self.data['original_score'].fillna(self.data['model_score'])
        if 'exposure_weight' not in self.data.columns:
            self.data['exposure_weight'] = 1.0
        self.data['exposure_weight'] = self.data['exposure_weight'].fillna(1.0)
        
        feature_cols = ['model_score', 'recency_score', 'popularity_score', 'diversity_score', 'original_score']
        self.features = self.data[feature_cols].values.astype(np.float32)
        
        # Labels (1 = positive, 0 = negative)
        self.labels = self.data['label'].values.astype(np.float32)
        self.weights = self.data['exposure_weight'].values.astype(np.float32)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"  Positive: {(self.labels == 1).sum()}")
        print(f"  Negative: {(self.labels == 0).sum()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.features[idx]),
            'label': torch.tensor(self.labels[idx]),
            'exposure_weight': torch.tensor(self.weights[idx])
        }


def create_pairwise_batches(batch, device):
    """
    Convert batch to pairwise format for ranking loss
    
    Args:
        batch: Dict with 'features' and 'label'
        device: torch device
    
    Returns:
        pos_features, neg_features tensors
    """
    features = batch['features'].to(device)
    labels = batch['label'].to(device)
    weights = batch['exposure_weight'].to(device)
    
    # Separate positive and negative examples
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_features = features[pos_mask]
    neg_features = features[neg_mask]
    pos_weights = weights[pos_mask]
    neg_weights = weights[neg_mask]
    
    return pos_features, neg_features, pos_weights, neg_weights


def train_ranker(train_path, val_path=None, epochs=20, batch_size=128, lr=0.001, output_dir='models'):
    """
    Train neural ranker model
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        output_dir: Directory to save model
    """
    print("=" * 60)
    print("RANKER MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load data
    print("Loading datasets...")
    train_dataset = RankerDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_path and os.path.exists(val_path):
        val_dataset = RankerDataset(val_path)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    print("Initializing model...")
    model = RankerMLP(input_dim=5, hidden_dim1=32, hidden_dim2=16).to(device)
    
    # Loss and optimizer
    criterion = PairwiseRankingLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # Training loop
    print("Starting training...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        start_time = time.time()
        
        for batch in train_loader:
            pos_features, neg_features, pos_weights, neg_weights = create_pairwise_batches(batch, device)
            
            # Skip if no pairs
            if len(pos_features) == 0 or len(neg_features) == 0:
                continue
            
            # Forward pass
            pos_scores = model(pos_features)
            neg_scores = model(neg_features)
            
            # Make sure we have equal number of pos and neg for pairing
            min_len = min(len(pos_scores), len(neg_scores))
            if min_len == 0:
                continue
            
            pos_scores = pos_scores[:min_len]
            neg_scores = neg_scores[:min_len]
            pair_weight = torch.minimum(
                pos_weights[:min_len], neg_weights[:min_len]
            )
            
            # Loss
            raw_loss = criterion(pos_scores, neg_scores)
            loss = (raw_loss * pair_weight).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        
        # Validation
        val_loss_str = "N/A"
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    pos_features, neg_features, pos_weights, neg_weights = create_pairwise_batches(batch, device)
                    
                    if len(pos_features) == 0 or len(neg_features) == 0:
                        continue
                    
                    pos_scores = model(pos_features)
                    neg_scores = model(neg_features)
                    
                    min_len = min(len(pos_scores), len(neg_scores))
                    if min_len == 0:
                        continue
                    
                    pos_scores = pos_scores[:min_len]
                    neg_scores = neg_scores[:min_len]
                    pair_weight = torch.minimum(
                        pos_weights[:min_len], neg_weights[:min_len]
                    )
                    
                    raw_loss = criterion(pos_scores, neg_scores)
                    val_loss += (raw_loss * pair_weight).mean().item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            val_loss_str = f"{avg_val_loss:.4f}"
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, 'ranker_v1_best.pth')
                torch.save(model.state_dict(), model_path)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss_str} | "
              f"Time: {epoch_time:.2f}s")
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, 'ranker_v1.pth')
    torch.save(model.state_dict(), final_path)
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print(f"Final model saved: {final_path}")
    if val_loader:
        best_path = os.path.join(output_dir, 'ranker_v1_best.pth')
        print(f"Best model saved: {best_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Train Ranker Model')
    parser.add_argument('--data', type=str, default='data/ranker_train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--val', type=str, default=None,
                        help='Path to validation data CSV (optional)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained model')
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.data):
        print(f"Error: Training data not found at {args.data}")
        print()
        print("Please generate training data first:")
        print("  cd backend")
        print("  python manage.py generate_ranker_dataset")
        print()
        return
    
    # Train
    train_ranker(
        train_path=args.data,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

