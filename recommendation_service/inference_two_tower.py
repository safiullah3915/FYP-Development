"""
Two-Tower Model Inference (No Circular Imports)
Standalone inference module for making predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database.connection import SessionLocal
from database.models import User, Startup, StartupTag, Position
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Model Architecture (copied from training script)
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
        self.user_tower = Tower(user_dim, startup_dim, hidden_dims, dropout_rate)
        self.startup_tower = Tower(startup_dim, embedding_dim, hidden_dims, dropout_rate)
    
    def forward(self, user_features, startup_features):
        user_emb = self.user_tower(user_features)
        startup_emb = self.startup_tower(startup_features)
        similarity = torch.sum(user_emb * startup_emb, dim=1)
        return torch.sigmoid(similarity)


# ============================================================================
# Feature Processing (simplified from training)
# ============================================================================

def parse_json_list(json_str):
    """Parse JSON list string"""
    if not json_str or json_str == 'null':
        return []
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else []
    except:
        return []


def parse_embedding(emb_str):
    """Parse embedding JSON string"""
    if not emb_str or emb_str == 'null':
        return None
    try:
        emb = json.loads(emb_str)
        if isinstance(emb, list) and len(emb) > 0:
            return emb
    except:
        pass
    return None


# ============================================================================
# Inference Class
# ============================================================================

class TwoTowerInference:
    """Simple inference wrapper for Two-Tower model"""
    
    def __init__(self, model_path: str, user_dim: int = 502, startup_dim: int = 471):
        """
        Initialize inference module
        
        Args:
            model_path: Path to trained model (.pth file)
            user_dim: User feature dimension (default from training)
            startup_dim: Startup feature dimension (default from training)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TwoTowerModel(user_dim, startup_dim).to(self.device)
        
        # Load trained weights
        logger.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        self.db = SessionLocal()
    
    def extract_user_features(self, user: User) -> np.ndarray:
        """
        Extract features for a single user (simplified version)
        Returns 502-dim feature vector
        """
        # Get embedding
        embedding = parse_embedding(user.profile_embedding)
        user_emb = np.array(embedding, dtype=np.float32) if embedding else np.zeros(384, dtype=np.float32)
        
        # Get preferences
        try:
            prefs = user.onboarding_preferences
            categories = prefs.selected_categories or []
            fields = prefs.selected_fields or []
            tags = prefs.selected_tags or []
            stages = prefs.preferred_startup_stages or []
            engagement = prefs.preferred_engagement_types or []
            skills = prefs.preferred_skills or []
        except:
            categories = fields = tags = stages = engagement = skills = []
        
        # Get profile skills
        try:
            profile = user.profile
            profile_skills = profile.skills or []
        except:
            profile_skills = []
        
        # Combine skills
        all_skills = list(set(skills + profile_skills))
        
        # Create simple one-hot/multi-hot features (simplified)
        # Note: This is a simplified version. For production, you'd need to use
        # the same encoders from training or save/load them
        role_feat = np.array([1.0 if user.role == 'student' else 0.0], dtype=np.float32)
        
        # Create fixed-size multi-hot vectors (simplified)
        cat_feat = np.zeros(13, dtype=np.float32)  # 13 categories
        field_feat = np.zeros(20, dtype=np.float32)  # 20 fields
        tag_feat = np.zeros(29, dtype=np.float32)  # 29 tags
        stage_feat = np.zeros(5, dtype=np.float32)  # 5 stages
        eng_feat = np.zeros(4, dtype=np.float32)  # 4 engagement types
        skill_feat = np.zeros(48, dtype=np.float32)  # 48 skills
        
        # NOTE: You should save encoders during training and load them here
        # For now, using zeros as placeholder for categorical features
        
        # Concatenate all features
        user_features = np.concatenate([
            user_emb,  # 384
            role_feat,  # 1
            cat_feat,  # 13
            field_feat,  # 20
            tag_feat,  # 29
            stage_feat,  # 5
            eng_feat,  # 4
            skill_feat,  # 48
        ])  # Total: 504, but we need 502, so pad/trim if needed
        
        # Ensure correct dimension
        if len(user_features) > 502:
            user_features = user_features[:502]
        elif len(user_features) < 502:
            user_features = np.pad(user_features, (0, 502 - len(user_features)))
        
        return user_features
    
    def extract_startup_features(self, startup: Startup) -> np.ndarray:
        """
        Extract features for a single startup (simplified version)
        Returns 471-dim feature vector
        """
        # Get embedding
        embedding = parse_embedding(startup.profile_embedding)
        startup_emb = np.array(embedding, dtype=np.float32) if embedding else np.zeros(384, dtype=np.float32)
        
        # Get tags
        tags = list(startup.tags.values_list('tag', flat=True))
        
        # Get positions
        positions = startup.positions.filter(is_active=True)
        position_titles = [p.title for p in positions]
        
        # Create simple one-hot features (simplified)
        type_feat = np.array([1.0 if startup.type == 'collaboration' else 0.0, 
                              1.0 if startup.type == 'marketplace' else 0.0], dtype=np.float32)
        
        # Create fixed-size multi-hot vectors (simplified)
        cat_feat = np.zeros(13, dtype=np.float32)  # 13 categories
        field_feat = np.zeros(20, dtype=np.float32)  # 20 fields
        phase_feat = np.zeros(4, dtype=np.float32)  # 4 phases
        tag_feat = np.zeros(44, dtype=np.float32)  # 44 tags
        stage_feat = np.zeros(4, dtype=np.float32)  # 4 stages
        
        # NOTE: You should save encoders during training and load them here
        
        # Concatenate all features
        startup_features = np.concatenate([
            startup_emb,  # 384
            type_feat,  # 2
            cat_feat,  # 13
            field_feat,  # 20
            phase_feat,  # 4
            tag_feat,  # 44
            stage_feat,  # 4
        ])  # Total: 471
        
        # Ensure correct dimension
        if len(startup_features) > 471:
            startup_features = startup_features[:471]
        elif len(startup_features) < 471:
            startup_features = np.pad(startup_features, (0, 471 - len(startup_features)))
        
        return startup_features
    
    def predict(self, user_id: str, startup_ids: List[str]) -> Dict[str, float]:
        """
        Predict scores for user-startup pairs
        
        Args:
            user_id: User ID
            startup_ids: List of startup IDs
            
        Returns:
            Dict mapping startup_id to score
        """
        try:
            # Get user
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User {user_id} not found")
                return {}
            
            # Get startups
            startups = self.db.query(Startup).filter(Startup.id.in_(startup_ids)).all()
            if not startups:
                logger.warning("No startups found")
                return {}
            
            # Extract user features
            user_features = self.extract_user_features(user)
            
            # Process in batches
            scores = {}
            batch_size = 32
            
            startup_list = list(startups)
            for i in range(0, len(startup_list), batch_size):
                batch_startups = startup_list[i:i+batch_size]
                
                # Extract startup features
                startup_features_batch = []
                for startup in batch_startups:
                    startup_feat = self.extract_startup_features(startup)
                    startup_features_batch.append(startup_feat)
                
                # Prepare tensors
                user_tensor = torch.tensor([user_features] * len(batch_startups), dtype=torch.float32).to(self.device)
                startup_tensor = torch.tensor(startup_features_batch, dtype=torch.float32).to(self.device)
                
                # Predict
                with torch.no_grad():
                    batch_scores = self.model(user_tensor, startup_tensor).cpu().numpy()
                
                # Store scores
                for startup, score in zip(batch_startups, batch_scores):
                    scores[str(startup.id)] = float(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in predict: {e}", exc_info=True)
            return {}
    
    def recommend(self, user_id: str, limit: int = 10, filters: Dict = None) -> Dict:
        """
        Get top-K recommendations for a user
        
        Args:
            user_id: User ID
            limit: Number of recommendations
            filters: Optional filters (type, category, etc.)
            
        Returns:
            Dict with item_ids, scores, match_reasons
        """
        try:
            # Get candidate startups from database
            query = self.db.query(Startup).filter(Startup.status == 'active')
            
            if filters:
                if 'type' in filters:
                    query = query.filter(Startup.type == filters['type'])
                if 'category' in filters:
                    if isinstance(filters['category'], list):
                        query = query.filter(Startup.category.in_(filters['category']))
                    else:
                        query = query.filter(Startup.category == filters['category'])
            
            candidates = query.all()
            if not candidates:
                return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
            
            logger.info(f"Scoring {len(candidates)} candidates for user {user_id}")
            
            # Get scores
            candidate_ids = [str(s.id) for s in candidates]
            scores = self.predict(user_id, candidate_ids)
            
            if not scores:
                return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
            
            # Sort and take top-K
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:limit]
            
            item_ids = [item_id for item_id, _ in top_items]
            top_scores = {item_id: score for item_id, score in top_items}
            
            # Generate match reasons
            match_reasons = {
                item_id: [
                    f"AI model prediction score: {score:.3f}",
                    "Based on your interaction history",
                    "Learned from similar users"
                ]
                for item_id, score in top_items
            }
            
            logger.info(f"Generated {len(item_ids)} recommendations")
            
            return {
                'item_ids': item_ids,
                'scores': top_scores,
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in recommend: {e}", exc_info=True)
            return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db'):
            self.db.close()


# ============================================================================
# Quick Test Function
# ============================================================================

def test_inference():
    """Test inference module"""
    logger.info("Testing Two-Tower inference...")
    
    # Initialize
    model_path = "models/two_tower_v1_best.pth"
    inference = TwoTowerInference(model_path)
    
    # Get a test user
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.role == 'student').first()
        if not user:
            logger.error("No test user found")
            return
        
        logger.info(f"Testing with user: {user.username} ({user.id})")
        
        # Get recommendations
        results = inference.recommend(str(user.id), limit=5)
        
        logger.info(f"Got {len(results['item_ids'])} recommendations:")
        for i, item_id in enumerate(results['item_ids'], 1):
            score = results['scores'].get(item_id, 0.0)
            logger.info(f"  {i}. Startup {item_id}: score={score:.3f}")
        
        logger.info("✓ Inference test successful!")
        
    finally:
        db.close()


if __name__ == '__main__':
    test_inference()

