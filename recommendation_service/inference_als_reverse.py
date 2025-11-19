"""
ALS Reverse Inference Module (SVD-powered)
Provides Startup → User recommendations without implicit dependency.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from database.connection import SessionLocal
from database.models import User
from utils.logger import get_logger

logger = get_logger(__name__)


class ALSReverseInference:
    """
    Standalone reverse inference using SVD embeddings
    """

    def __init__(self, model_path: str):
        """
        Initialize ALS Reverse inference

        Args:
            model_path: Path to config JSON or legacy prefix (e.g., models/als_reverse_v1_config.json)
        """
        self.user_factors = None  # Startups
        self.item_factors = None  # Users
        self.user_mapping = None
        self.item_mapping = None
        self.user_reverse = None
        self.item_reverse = None
        self.config = {}

        self._load_model(model_path)
    
    def _resolve_prefix(self, model_path: str) -> Path:
        path = Path(model_path)
        if path.is_dir():
            raise ValueError("Model path cannot be a directory; supply the config file or prefix.")

        if path.suffix == '.json' and path.stem.endswith('_config'):
            return path.with_name(path.stem.replace('_config', ''))

        if path.suffix == '.pkl':
            return path.with_suffix('')

        return path

    def _load_model(self, model_path: str):
        """Load embeddings and mappings"""
        try:
            prefix = self._resolve_prefix(model_path)

            startup_factors_path = f"{prefix}_user_factors.npy"
            user_factors_path = f"{prefix}_item_factors.npy"
            startup_mapping_path = f"{prefix}_user_mapping.json"
            user_mapping_path = f"{prefix}_item_mapping.json"
            config_path = f"{prefix}_config.json"

            for path in [startup_factors_path, user_factors_path, startup_mapping_path, user_mapping_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing artifact: {path}")

            self.user_factors = np.load(startup_factors_path)
            self.item_factors = np.load(user_factors_path)

            with open(startup_mapping_path, 'r') as f:
                self.user_mapping = {str(k): int(v) for k, v in json.load(f).items()}

            with open(user_mapping_path, 'r') as f:
                self.item_mapping = {str(k): int(v) for k, v in json.load(f).items()}

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)

            self.user_reverse = {idx: startup_id for startup_id, idx in self.user_mapping.items()}
            self.item_reverse = {idx: user_id for user_id, idx in self.item_mapping.items()}

            # ALS Reverse model loaded silently - status logged at app startup

        except Exception as e:
            logger.error(f"Failed to load ALS Reverse artifacts: {e}")
            raise
    
    def recommend(self, startup_id: str, limit: int = 10, filters: Optional[Dict] = None, fetch_multiplier: int = 1) -> Dict:
        """
        Generate user recommendations for a startup
        
        Args:
            startup_id: Startup ID (UUID string)
            limit: Number of recommendations to return
            filters: Optional filters (role: 'student' or 'investor')
            fetch_multiplier: Multiplier for fetching more candidates (for reranking)
        
        Returns:
            dict with:
                - users: List of user recommendations
                - total: Number of recommendations
                - scores: Dict of user_id -> score
                - method_used: 'als_reverse'
        """
        db = SessionLocal()
        try:
            actual_limit = limit * fetch_multiplier
            logger.info(f"ALS Reverse inference for startup {startup_id}, limit {limit} (fetching {actual_limit} candidates)")
            
            # Check if startup exists in model
            if startup_id not in self.user_mapping:
                logger.warning(f"Startup {startup_id} not in ALS Reverse model, using fallback")
                return self._fallback_recent(limit, filters, db)
            
            # Get startup embedding
            startup_idx = self.user_mapping[startup_id]
            startup_embedding = self.user_factors[startup_idx]
            
            # Compute scores for all users
            scores = self.item_factors.dot(startup_embedding)
            
            # Get top candidates (more than limit for filtering)
            # Use actual_limit * 5 to allow for filtering
            top_k_indices = np.argsort(scores)[::-1][:actual_limit * 5]
            
            # Convert to user IDs and scores
            candidate_ids = []
            candidate_scores = {}
            for idx in top_k_indices:
                if idx in self.item_reverse:
                    user_id = self.item_reverse[idx]
                    candidate_ids.append(user_id)
                    candidate_scores[user_id] = float(scores[idx])
            
            # Query users from database
            query = db.query(User).filter(
                User.id.in_(candidate_ids),
                User.is_active == True
            )
            
            # Apply role filter
            if filters and filters.get('role'):
                role = filters['role']
                # Map 'student' to 'student' or 'developer' roles
                if role == 'student':
                    query = query.filter(User.role == 'student')
                elif role == 'investor':
                    query = query.filter(User.role == 'investor')
            
            users = query.all()
            
            # Sort by ALS scores
            users_dict = {str(u.id): u for u in users}
            sorted_users = []
            for user_id in candidate_ids:
                if user_id in users_dict:
                    sorted_users.append(users_dict[user_id])
                if len(sorted_users) >= actual_limit:
                    break
            
            # Format response
            recommendations = []
            for user in sorted_users:
                user_id = str(user.id)
                recommendations.append({
                    'id': user_id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'score': candidate_scores.get(user_id, 0.0),
                    'match_reasons': [
                        f"ALS Reverse score: {candidate_scores.get(user_id, 0.0):.3f}",
                        "This user has shown interest in similar startups"
                    ]
                })
            
            return {
                'users': recommendations,
                'total': len(recommendations),
                'scores': {r['id']: r['score'] for r in recommendations},
                'method_used': 'als_reverse'
            }
            
        except Exception as e:
            logger.error(f"Error in ALS Reverse inference: {e}", exc_info=True)
            return self._fallback_recent(limit, filters, db)
        finally:
            db.close()
    
    def _fallback_recent(self, limit: int, filters: Optional[Dict], db) -> Dict:
        """Fallback to recent active users when ALS Reverse cannot be used"""
        logger.info("Using recent active users fallback")
        
        try:
            query = db.query(User).filter(User.is_active == True)
            
            # Apply role filter
            if filters and filters.get('role'):
                role = filters['role']
                if role == 'student':
                    query = query.filter(User.role == 'student')
                elif role == 'investor':
                    query = query.filter(User.role == 'investor')
            
            # Order by most recently updated (active users)
            users = query.order_by(User.updated_at.desc()).limit(limit).all()
            
            recommendations = []
            for user in users:
                recommendations.append({
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'score': 1.0,  # Default score
                    'match_reasons': ["Recently active user (ALS Reverse fallback)"]
                })
            
            return {
                'users': recommendations,
                'total': len(recommendations),
                'scores': {r['id']: r['score'] for r in recommendations},
                'method_used': 'recent_active_fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback: {e}")
            return {
                'users': [],
                'total': 0,
                'scores': {},
                'method_used': 'error'
            }



