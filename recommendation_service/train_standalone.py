#!/usr/bin/env python3
"""
Standalone Two-Tower Training Script (No Dependencies)
This script can run independently without circular import issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# ============================================================================
# Model Architecture
# ============================================================================

class Tower(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims, dropout_rate=0.3, dropout_rate_middle=0.2):
        super(Tower, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate if i == 0 else dropout_rate_middle))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding = self.model(x)
        return F.normalize(embedding, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, startup_dim, embedding_dim=128, hidden_dims=[512, 256], dropout_rate=0.3):
        super(TwoTowerModel, self).__init__()
        self.user_tower = Tower(user_dim, embedding_dim, hidden_dims, dropout_rate)
        self.startup_tower = Tower(startup_dim, embedding_dim, hidden_dims, dropout_rate)
    
    def forward(self, user_features, startup_features):
        user_emb = self.user_tower(user_features)
        startup_emb = self.startup_tower(startup_features)
        similarity = torch.sum(user_emb * startup_emb, dim=1)
        return torch.sigmoid(similarity)


class WeightedBCELoss(nn.Module):
    def forward(self, predictions, targets, weights):
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        return (bce * weights).mean()


class TwoTowerDataset(Dataset):
    def __init__(self, user_features, startup_features, labels, weights):
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


# ============================================================================
# Feature Processing
# ============================================================================

def parse_json_list(json_str):
    if not json_str or json_str == 'null' or pd.isna(json_str):
        return []
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except:
        return []


def parse_embedding(emb_str):
    if not emb_str or emb_str == 'null' or pd.isna(emb_str):
        return None
    try:
        emb = json.loads(emb_str)
        if isinstance(emb, list) and len(emb) > 0:
            return emb
    except:
        pass
    return None


def process_features(df):
    """Process raw CSV data into feature matrices"""
    print("Processing features...")
    
    # Extract all categorical values
    user_categories_all = [parse_json_list(x) for x in df['user_categories']]
    user_fields_all = [parse_json_list(x) for x in df['user_fields']]
    user_tags_all = [parse_json_list(x) for x in df['user_tags']]
    user_stages_all = [parse_json_list(x) for x in df['user_stages']]
    user_engagement_all = [parse_json_list(x) for x in df['user_engagement']]
    user_skills_all = [parse_json_list(x) for x in df['user_skills']]
    startup_tags_all = [parse_json_list(x) for x in df['startup_tags']]
    startup_stages_all = [parse_json_list(x) for x in df['startup_stages']]
    
    # Fit multi-label binarizers
    mlb_user_cat = MultiLabelBinarizer().fit(user_categories_all)
    mlb_user_fields = MultiLabelBinarizer().fit(user_fields_all)
    mlb_user_tags = MultiLabelBinarizer().fit(user_tags_all)
    mlb_user_stages = MultiLabelBinarizer().fit(user_stages_all)
    mlb_user_engagement = MultiLabelBinarizer().fit(user_engagement_all)
    mlb_user_skills = MultiLabelBinarizer().fit(user_skills_all)
    mlb_startup_tags = MultiLabelBinarizer().fit(startup_tags_all)
    mlb_startup_stages = MultiLabelBinarizer().fit(startup_stages_all)
    
    # Fit single value encoders
    roles = sorted(df['user_role'].unique())
    types = sorted(df['startup_type'].unique())
    categories = sorted(df['startup_category'].unique())
    fields = sorted(df['startup_field'].unique())
    phases = sorted([str(x) for x in df['startup_phase'].unique() if x and str(x) != '' and not pd.isna(x)])
    
    role_encoder = {role: idx for idx, role in enumerate(roles)}
    type_encoder = {t: idx for idx, t in enumerate(types)}
    category_encoder = {cat: idx for idx, cat in enumerate(categories)}
    field_encoder = {field: idx for idx, field in enumerate(fields)}
    phase_encoder = {phase: idx for idx, phase in enumerate(phases)} if phases else {'': 0}
    
    print(f"  Feature dimensions:")
    print(f"    User categories: {len(mlb_user_cat.classes_)}")
    print(f"    User skills: {len(mlb_user_skills.classes_)}")
    print(f"    Startup tags: {len(mlb_startup_tags.classes_)}")
    
    # Process each sample
    user_features_list = []
    startup_features_list = []
    
    for idx, row in df.iterrows():
        # User features
        user_emb = parse_embedding(row['user_embedding'])
        user_emb = np.array(user_emb, dtype=np.float32) if user_emb else np.zeros(384, dtype=np.float32)
        
        role_onehot = np.zeros(len(role_encoder), dtype=np.float32)
        role_onehot[role_encoder.get(row['user_role'], 0)] = 1.0
        
        user_cat = mlb_user_cat.transform([parse_json_list(row['user_categories'])])[0].astype(np.float32)
        user_fields = mlb_user_fields.transform([parse_json_list(row['user_fields'])])[0].astype(np.float32)
        user_tags = mlb_user_tags.transform([parse_json_list(row['user_tags'])])[0].astype(np.float32)
        user_stages = mlb_user_stages.transform([parse_json_list(row['user_stages'])])[0].astype(np.float32)
        user_eng = mlb_user_engagement.transform([parse_json_list(row['user_engagement'])])[0].astype(np.float32)
        user_skills = mlb_user_skills.transform([parse_json_list(row['user_skills'])])[0].astype(np.float32)
        
        user_feat = np.concatenate([user_emb, role_onehot, user_cat, user_fields, user_tags, user_stages, user_eng, user_skills])
        user_features_list.append(user_feat)
        
        # Startup features
        startup_emb = parse_embedding(row['startup_embedding'])
        startup_emb = np.array(startup_emb, dtype=np.float32) if startup_emb else np.zeros(384, dtype=np.float32)
        
        type_onehot = np.zeros(len(type_encoder), dtype=np.float32)
        type_onehot[type_encoder.get(row['startup_type'], 0)] = 1.0
        
        cat_onehot = np.zeros(len(category_encoder), dtype=np.float32)
        cat_onehot[category_encoder.get(row['startup_category'], 0)] = 1.0
        
        field_onehot = np.zeros(len(field_encoder), dtype=np.float32)
        field_onehot[field_encoder.get(row['startup_field'], 0)] = 1.0
        
        phase_onehot = np.zeros(len(phase_encoder), dtype=np.float32)
        phase_val = row['startup_phase'] if row['startup_phase'] and row['startup_phase'] != '' else ''
        phase_onehot[phase_encoder.get(phase_val, 0)] = 1.0
        
        startup_tags = mlb_startup_tags.transform([parse_json_list(row['startup_tags'])])[0].astype(np.float32)
        startup_stages = mlb_startup_stages.transform([parse_json_list(row['startup_stages'])])[0].astype(np.float32)
        
        startup_feat = np.concatenate([startup_emb, type_onehot, cat_onehot, field_onehot, phase_onehot, startup_tags, startup_stages])
        startup_features_list.append(startup_feat)
    
    user_features = np.array(user_features_list, dtype=np.float32)
    startup_features = np.array(startup_features_list, dtype=np.float32)
    labels = df['label'].values.astype(np.float32)
    weights = df['weight'].values.astype(np.float32)
    
    print(f"  User features shape: {user_features.shape}")
    print(f"  Startup features shape: {startup_features.shape}")
    
    return user_features, startup_features, labels, weights


# ============================================================================
# Training
# ============================================================================

def train_model(train_loader, val_loader, model, device, epochs=50, lr=0.001):
    criterion = WeightedBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    best_val_loss = float('inf')
    patience = 0
    patience_limit = 5
    
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            user_features = batch['user_features'].to(device)
            startup_features = batch['startup_features'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device)
            
            optimizer.zero_grad()
            predictions = model(user_features, startup_features)
            loss = criterion(predictions, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user_features = batch['user_features'].to(device)
                startup_features = batch['startup_features'].to(device)
                labels = batch['labels'].to(device)
                weights = batch['weights'].to(device)
                
                predictions = model(user_features, startup_features)
                loss = criterion(predictions, labels, weights)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'models/two_tower_v1_best.pth')
            print(f"  >>> New best model saved!")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        scheduler.step()
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print(f"Loading dataset from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples\n")
    
    # Process features
    user_features, startup_features, labels, weights = process_features(df)
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_dataset = TwoTowerDataset(
        user_features[train_idx], startup_features[train_idx],
        labels[train_idx], weights[train_idx]
    )
    val_dataset = TwoTowerDataset(
        user_features[val_idx], startup_features[val_idx],
        labels[val_idx], weights[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    
    # Create model
    user_dim = user_features.shape[1]
    startup_dim = startup_features.shape[1]
    model = TwoTowerModel(user_dim, startup_dim).to(device)
    
    print(f"Model architecture:")
    print(f"  User input dim: {user_dim}")
    print(f"  Startup input dim: {startup_dim}")
    print(f"  Embedding dim: 128")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train
    model = train_model(train_loader, val_loader, model, device, args.epochs, args.lr)
    
    print("\n>>> Training complete!")
    print(f"Best model saved to: models/two_tower_v1_best.pth")


if __name__ == '__main__':
    main()

