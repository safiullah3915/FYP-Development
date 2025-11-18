"""
SVD-based ALS Inference Module
Loads precomputed embeddings produced by train_als.py and serves recommendations.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from database.connection import SessionLocal
from database.models import Startup
from utils.logger import get_logger

logger = get_logger(__name__)


class ALSInference:
    """
    Standalone inference for collaborative filtering embeddings
    """

    def __init__(self, model_path: str):
        """
        Initialize ALS/SVD inference

        Args:
            model_path: Path to config JSON or legacy model prefix (e.g., models/als_v1_config.json)
        """
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_reverse = None
        self.item_reverse = None
        self.config = {}

        self._load_model(model_path)
    
    def _resolve_prefix(self, model_path: str) -> Path:
        """
        Derive the artifact prefix (without suffix) from whichever path was provided.
        Supports:
            - .../als_v1_config.json
            - .../als_v1.pkl (legacy)
            - .../als_v1 (prefix directly)
        """
        path = Path(model_path)
        if path.is_dir():
            raise ValueError(f"Provided model path '{model_path}' is a directory; please pass a config file or prefix.")

        if path.suffix == '.json' and path.stem.endswith('_config'):
            return path.with_name(path.stem.replace('_config', ''))

        if path.suffix == '.pkl':
            # Legacy pointer: simply drop the extension and use the prefix
            return path.with_suffix('')

        return path

    def _load_model(self, model_path: str):
        """Load embeddings, mappings, and config metadata"""
        try:
            prefix = self._resolve_prefix(model_path)
            logger.info(f"Loading ALS/SVD artifacts using prefix {prefix}")

            user_factors_path = f"{prefix}_user_factors.npy"
            item_factors_path = f"{prefix}_item_factors.npy"
            user_mapping_path = f"{prefix}_user_mapping.json"
            item_mapping_path = f"{prefix}_item_mapping.json"
            config_path = f"{prefix}_config.json"

            for path in [user_factors_path, item_factors_path, user_mapping_path, item_mapping_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing artifact: {path}")

            self.user_factors = np.load(user_factors_path)
            self.item_factors = np.load(item_factors_path)

            with open(user_mapping_path, 'r') as f:
                raw_user_mapping = json.load(f)
                self.user_mapping = {str(k): int(v) for k, v in raw_user_mapping.items()}

            with open(item_mapping_path, 'r') as f:
                raw_item_mapping = json.load(f)
                self.item_mapping = {str(k): int(v) for k, v in raw_item_mapping.items()}

            config_exists = os.path.exists(config_path)
            if config_exists:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file not found at {config_path}; continuing with defaults.")

            self.user_reverse = {idx: user_id for user_id, idx in self.user_mapping.items()}
            self.item_reverse = {idx: item_id for item_id, idx in self.item_mapping.items()}

            logger.info("ALS/SVD embeddings loaded successfully:")
            logger.info(f"  Users: {len(self.user_mapping)}")
            logger.info(f"  Items: {len(self.item_mapping)}")
            logger.info(f"  Factors: {self.user_factors.shape[1]}")

        except Exception as e:
            logger.error(f"Failed to load ALS/SVD artifacts: {e}")
            raise
    
    def recommend(self, user_id: str, limit: int = 10, filters: Optional[Dict] = None, fetch_multiplier: int = 1) -> Dict:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID (UUID string)
            limit: Number of recommendations to return
            filters: Optional filters (type, category, field, etc.)
            fetch_multiplier: Multiplier for fetching more candidates (for reranking)
        
        Returns:
            dict with:
                - startups: List of startup recommendations
                - total: Number of recommendations
                - scores: Dict of startup_id -> score
                - method_used: 'als'
        """
        db = SessionLocal()
        try:
            actual_limit = limit * fetch_multiplier
            logger.info(f"ALS inference for user {user_id}, limit {limit} (fetching {actual_limit} candidates)")
            
            # Normalize UUID format (remove dashes) since SQLite stores UUIDs without dashes
            # and the ALS model was trained with normalized IDs
            normalized_user_id = str(user_id).replace('-', '')
            
            # Check if user exists in model
            if normalized_user_id not in self.user_mapping:
                logger.warning(f"User {user_id} (normalized: {normalized_user_id}) not in ALS model, using fallback")
                return self._fallback_popular(limit, filters, db)
            
            # Get user embedding
            user_idx = self.user_mapping[normalized_user_id]
            user_embedding = self.user_factors[user_idx]
            
            # Compute scores for all items
            scores = self.item_factors.dot(user_embedding)
            
            # Get top candidates (more than limit for filtering)
            # Use actual_limit * 5 to allow for filtering
            top_k_indices = np.argsort(scores)[::-1][:actual_limit * 5]
            
            # Convert to startup IDs and scores
            candidate_ids = []
            candidate_scores = {}
            for idx in top_k_indices:
                if idx in self.item_reverse:
                    startup_id = self.item_reverse[idx]
                    candidate_ids.append(startup_id)
                    candidate_scores[startup_id] = float(scores[idx])
            
            # Query startups from database
            # candidate_ids from ALS model are already normalized (without dashes)
            query = db.query(Startup).filter(
                Startup.id.in_(candidate_ids),
                Startup.status == 'active'
            )
            
            # Apply filters
            if filters:
                if filters.get('type'):
                    query = query.filter(Startup.type == filters['type'])
                if filters.get('category'):
                    categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                    query = query.filter(Startup.category.in_(categories))
                if filters.get('field'):
                    fields = filters['field'] if isinstance(filters['field'], list) else [filters['field']]
                    query = query.filter(Startup.field.in_(fields))
            
            startups = query.all()
            
            # Sort by ALS scores
            startups_dict = {str(s.id): s for s in startups}
            sorted_startups = []
            for startup_id in candidate_ids:
                if startup_id in startups_dict:
                    sorted_startups.append(startups_dict[startup_id])
                if len(sorted_startups) >= actual_limit:
                    break
            
            # Format response
            recommendations = []
            for startup in sorted_startups:
                startup_id = str(startup.id)
                recommendations.append({
                    'id': startup_id,
                    'title': startup.title,
                    'description': startup.description,
                    'type': startup.type,
                    'category': startup.category,
                    'field': startup.field,
                    'score': candidate_scores.get(startup_id, 0.0),
                    'match_reasons': [
                        f"ALS score: {candidate_scores.get(startup_id, 0.0):.3f}",
                        "Users with similar preferences engaged with this startup"
                    ]
                })
            
            return {
                'startups': recommendations,
                'total': len(recommendations),
                'scores': {r['id']: r['score'] for r in recommendations},
                'method_used': 'als'
            }
            
        except Exception as e:
            logger.error(f"Error in ALS inference: {e}", exc_info=True)
            return self._fallback_popular(limit, filters, db)
        finally:
            db.close()
    
    def _fallback_popular(self, limit: int, filters: Optional[Dict], db) -> Dict:
        """Fallback to popular items when ALS cannot be used"""
        logger.info("Using popular item fallback")
        
        try:
            query = db.query(Startup).filter(Startup.status == 'active')
            
            # Apply filters
            if filters:
                if filters.get('type'):
                    query = query.filter(Startup.type == filters['type'])
                if filters.get('category'):
                    categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                    query = query.filter(Startup.category.in_(categories))
            
            startups = query.order_by(Startup.views.desc()).limit(limit).all()
            
            recommendations = []
            for startup in startups:
                recommendations.append({
                    'id': str(startup.id),
                    'title': startup.title,
                    'description': startup.description,
                    'type': startup.type,
                    'category': startup.category,
                    'field': startup.field,
                    'score': float(startup.views),
                    'match_reasons': ["Popular startup (ALS fallback)"]
                })
            
            return {
                'startups': recommendations,
                'total': len(recommendations),
                'scores': {r['id']: r['score'] for r in recommendations},
                'method_used': 'popular_fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback: {e}")
            return {
                'startups': [],
                'total': 0,
                'scores': {},
                'method_used': 'error'
            }


