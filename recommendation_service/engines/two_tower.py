"""
Two-Tower Neural Network for Recommendation
Implements PyTorch-based two-tower architecture for user-item matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json

from .base_recommender import BaseRecommender
from .training_config import TwoTowerConfig
from .feature_engineering import FeatureEncoder
from database.models import Startup, User, Position
from database.connection import SessionLocal
from utils.logger import get_logger

logger = get_logger(__name__)


class Tower(nn.Module):
    """Generic tower (encoder) for user or item features"""
    
    def __init__(
        self, 
        input_dim: int, 
        embedding_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.3,
        dropout_rate_middle: float = 0.2
    ):
        """
        Initialize tower
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for first layer
            dropout_rate_middle: Dropout rate for middle layers
        """
        super(Tower, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Apply dropout
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(nn.Dropout(dropout_rate_middle))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation, will normalize)
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        embedding = self.model(x)
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class TwoTowerModel(nn.Module):
    """Two-Tower model for recommendation"""
    
    def __init__(self, config: TwoTowerConfig):
        """
        Initialize Two-Tower model
        
        Args:
            config: Training configuration
        """
        super(TwoTowerModel, self).__init__()
        
        self.config = config
        
        # User tower
        self.user_tower = Tower(
            input_dim=config.user_feature_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            dropout_rate_middle=config.dropout_rate_middle
        )
        
        # Startup/Item tower
        self.startup_tower = Tower(
            input_dim=config.startup_feature_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            dropout_rate_middle=config.dropout_rate_middle
        )
    
    def forward(
        self, 
        user_features: torch.Tensor, 
        startup_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute similarity scores
        
        Args:
            user_features: User feature tensor (batch_size, user_feature_dim)
            startup_features: Startup feature tensor (batch_size, startup_feature_dim)
            
        Returns:
            Similarity scores (batch_size,)
        """
        # Encode user and startup
        user_emb = self.user_tower(user_features)  # (batch_size, embedding_dim)
        startup_emb = self.startup_tower(startup_features)  # (batch_size, embedding_dim)
        
        # Compute dot product similarity
        similarity = torch.sum(user_emb * startup_emb, dim=1)  # (batch_size,)
        
        # Apply sigmoid to get scores in [0, 1]
        scores = torch.sigmoid(similarity)
        
        return scores
    
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode users to embeddings"""
        return self.user_tower(user_features)
    
    def encode_startups(self, startup_features: torch.Tensor) -> torch.Tensor:
        """Encode startups to embeddings"""
        return self.startup_tower(startup_features)
    
    def predict_batch(
        self, 
        user_features: torch.Tensor, 
        startup_features: torch.Tensor
    ) -> np.ndarray:
        """
        Predict scores for batch (inference mode)
        
        Args:
            user_features: User feature tensor
            startup_features: Startup feature tensor
            
        Returns:
            Numpy array of scores
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_features, startup_features)
            return scores.cpu().numpy()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss"""
    
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss
        
        Args:
            predictions: Predicted scores (batch_size,)
            targets: Target labels (batch_size,)
            weights: Sample weights (batch_size,)
            
        Returns:
            Scalar loss
        """
        # Binary cross-entropy
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Apply weights
        weighted_bce = bce * weights
        
        # Return mean
        return weighted_bce.mean()


class TwoTowerRecommender(BaseRecommender):
    """Two-Tower recommendation engine with inference capabilities"""
    
    def __init__(self, db_session=None, model_path: str = None, encoder_path: str = None):
        """
        Initialize Two-Tower recommender
        
        Args:
            db_session: Database session
            model_path: Path to trained model file
            encoder_path: Path to feature encoder file
        """
        self.db = db_session if db_session else SessionLocal()
        self.model = None
        self.encoder = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if path provided
        if model_path and encoder_path:
            self.load_model(model_path, encoder_path)
    
    def load_model(self, model_path: str, encoder_path: str):
        """
        Load trained model and encoder
        
        Args:
            model_path: Path to model checkpoint (.pth)
            encoder_path: Path to feature encoder (.pkl)
        """
        try:
            logger.info(f"Loading Two-Tower model from {model_path}")
            
            # Load encoder
            self.encoder = FeatureEncoder.load(encoder_path)
            logger.info(f"Loaded encoder from {encoder_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct config
            self.config = TwoTowerConfig.from_dict(checkpoint['config'])
            
            # Initialize model
            self.model = TwoTowerModel(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
            logger.info(f"  User feature dim: {self.config.user_feature_dim}")
            logger.info(f"  Startup feature dim: {self.config.startup_feature_dim}")
            logger.info(f"  Embedding dim: {self.config.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def recommend(
        self, 
        user_id: str, 
        use_case: str, 
        limit: int, 
        filters: Dict = None
    ) -> Dict:
        """
        Generate recommendations using Two-Tower model
        
        Args:
            user_id: User ID
            use_case: Use case type
            limit: Maximum number of recommendations
            filters: Optional filters
            
        Returns:
            dict with item_ids, scores, match_reasons
        """
        if not self.model or not self.encoder:
            logger.error("Model not loaded. Call load_model() first.")
            return self._empty_result()
        
        try:
            logger.info(f"Generating Two-Tower recommendations for user {user_id}")
            
            # Get user from database
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User {user_id} not found")
                return self._empty_result()
            
            # Get candidate startups
            candidate_startups = self._get_candidate_startups(filters)
            if not candidate_startups:
                logger.warning("No candidate startups found")
                return self._empty_result()
            
            logger.info(f"Scoring {len(candidate_startups)} candidate startups")
            
            # Score all candidates
            scores_dict = self._score_candidates(user, candidate_startups)
            
            if not scores_dict:
                logger.warning("No scores computed")
                return self._empty_result()
            
            # Sort by score and take top-k
            sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:limit]
            
            item_ids = [item_id for item_id, _ in top_items]
            scores = {item_id: score for item_id, score in top_items}
            
            # Generate match reasons (placeholder)
            match_reasons = {
                item_id: ["Neural network prediction", f"Match score: {score:.3f}"]
                for item_id, score in top_items
            }
            
            logger.info(f"Generated {len(item_ids)} recommendations")
            
            return {
                'item_ids': item_ids,
                'scores': scores,
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in recommend: {e}", exc_info=True)
            return self._empty_result()
    
    def _get_candidate_startups(self, filters: Dict = None) -> List[Startup]:
        """Get candidate startups from database"""
        query = self.db.query(Startup).filter(Startup.status == 'active')
        
        if filters:
            if 'type' in filters:
                query = query.filter(Startup.type == filters['type'])
            if 'category' in filters:
                if isinstance(filters['category'], list):
                    query = query.filter(Startup.category.in_(filters['category']))
                else:
                    query = query.filter(Startup.category == filters['category'])
            if filters.get('require_open_positions'):
                query = query.join(Position, Position.startup_id == Startup.id).filter(
                    Position.is_active == True
                ).distinct()
        
        return query.all()
    
    def _score_candidates(self, user: User, startups: List[Startup]) -> Dict[str, float]:
        """Score all candidate startups for a user"""
        scores = {}
        
        # Prepare user sample
        user_sample = self._prepare_user_sample(user)
        
        # Process in batches for efficiency
        batch_size = 128
        for i in range(0, len(startups), batch_size):
            batch_startups = startups[i:i+batch_size]
            
            # Prepare batch samples
            batch_samples = []
            for startup in batch_startups:
                sample = {**user_sample, **self._prepare_startup_sample(startup)}
                batch_samples.append(sample)
            
            # Extract features
            batch_user_features = []
            batch_startup_features = []
            
            for sample in batch_samples:
                user_feats = self.encoder.transform_user_features(sample)
                startup_feats = self.encoder.transform_startup_features(sample)
                
                # Concatenate features
                user_feat_vector = np.concatenate([
                    user_feats['user_embedding'],
                    user_feats['user_role'],
                    user_feats['user_categories'],
                    user_feats['user_fields'],
                    user_feats['user_tags'],
                    user_feats['user_stages'],
                    user_feats['user_engagement'],
                    user_feats['user_skills'],
                ])
                
                startup_feat_vector = np.concatenate([
                    startup_feats['startup_embedding'],
                    startup_feats['startup_type'],
                    startup_feats['startup_category'],
                    startup_feats['startup_field'],
                    startup_feats['startup_phase'],
                    startup_feats['startup_tags'],
                    startup_feats['startup_stages'],
                ])
                
                batch_user_features.append(user_feat_vector)
                batch_startup_features.append(startup_feat_vector)
            
            # Convert to tensors
            user_tensor = torch.tensor(np.array(batch_user_features), dtype=torch.float32).to(self.device)
            startup_tensor = torch.tensor(np.array(batch_startup_features), dtype=torch.float32).to(self.device)
            
            # Predict scores
            batch_scores = self.model.predict_batch(user_tensor, startup_tensor)
            
            # Store scores
            for startup, score in zip(batch_startups, batch_scores):
                scores[str(startup.id)] = float(score)
        
        return scores
    
    def _prepare_user_sample(self, user: User) -> Dict:
        """Prepare user sample dictionary"""
        # Get preferences
        preferences = {}
        try:
            prefs = user.onboarding_preferences
            preferences = {
                'selected_categories': prefs.selected_categories or [],
                'selected_fields': prefs.selected_fields or [],
                'selected_tags': prefs.selected_tags or [],
                'preferred_stages': prefs.preferred_startup_stages or [],
                'preferred_engagement': prefs.preferred_engagement_types or [],
                'preferred_skills': prefs.preferred_skills or [],
            }
        except:
            preferences = {
                'selected_categories': [],
                'selected_fields': [],
                'selected_tags': [],
                'preferred_stages': [],
                'preferred_engagement': [],
                'preferred_skills': [],
            }
        
        # Get profile skills
        profile_skills = []
        try:
            profile = user.profile
            profile_skills = profile.skills or []
        except:
            profile_skills = []
        
        # Parse embedding
        embedding = None
        if user.profile_embedding:
            try:
                embedding = json.loads(user.profile_embedding)
            except:
                pass
        
        return {
            'user_role': user.role,
            'user_embedding': json.dumps(embedding) if embedding else None,
            'user_categories': json.dumps(preferences['selected_categories']),
            'user_fields': json.dumps(preferences['selected_fields']),
            'user_tags': json.dumps(preferences['selected_tags']),
            'user_stages': json.dumps(preferences['preferred_stages']),
            'user_engagement': json.dumps(preferences['preferred_engagement']),
            'user_skills': json.dumps(profile_skills),
        }
    
    def _prepare_startup_sample(self, startup: Startup) -> Dict:
        """Prepare startup sample dictionary"""
        # Get tags
        tags = list(startup.tags.values_list('tag', flat=True))
        
        # Get positions
        positions = startup.positions.filter(is_active=True)
        position_titles = [p.title for p in positions]
        position_requirements = [p.requirements for p in positions if p.requirements]
        
        # Parse embedding
        embedding = None
        if startup.profile_embedding:
            try:
                embedding = json.loads(startup.profile_embedding)
            except:
                pass
        
        return {
            'startup_type': startup.type,
            'startup_category': startup.category,
            'startup_field': startup.field,
            'startup_phase': startup.phase or '',
            'startup_embedding': json.dumps(embedding) if embedding else None,
            'startup_stages': json.dumps(startup.stages or []),
            'startup_tags': json.dumps(tags),
            'startup_positions': json.dumps(position_titles),
            'startup_position_requirements': json.dumps(position_requirements),
        }
    
    def explain(self, user_id: str, item_id: str, use_case: str) -> List[str]:
        """Generate explanations for recommendation"""
        if not self.model:
            return ["Model not loaded"]
        
        return [
            "Recommended by neural network",
            "Based on your preferences and interaction history",
            "Matched using deep learning model"
        ]
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'item_ids': [],
            'scores': {},
            'match_reasons': {}
        }
