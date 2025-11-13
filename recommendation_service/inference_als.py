"""
ALS Inference Module
Standalone module for loading and running ALS recommendations
Similar to inference_two_tower.py but for collaborative filtering
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from database.connection import SessionLocal
from database.models import Startup
from utils.logger import get_logger

logger = get_logger(__name__)


class ALSInference:
    """
    Standalone ALS inference for recommendation
    Loads model once and provides fast recommendations
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ALS inference
        
        Args:
            model_path: Path to trained ALS model (e.g., models/als_v1.pkl)
        """
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_reverse = None
        self.item_reverse = None
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained model and mappings"""
        try:
            logger.info(f"Loading ALS model from {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Get model directory and name
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('.pkl', '')
            
            # Load user and item factors
            user_factors_path = os.path.join(model_dir, f"{model_name}_user_factors.npy")
            item_factors_path = os.path.join(model_dir, f"{model_name}_item_factors.npy")
            
            if not os.path.exists(user_factors_path):
                raise FileNotFoundError(f"User factors not found: {user_factors_path}")
            if not os.path.exists(item_factors_path):
                raise FileNotFoundError(f"Item factors not found: {item_factors_path}")
            
            self.user_factors = np.load(user_factors_path)
            self.item_factors = np.load(item_factors_path)
            
            # Load mappings
            user_mapping_path = os.path.join(model_dir, f"{model_name}_user_mapping.json")
            item_mapping_path = os.path.join(model_dir, f"{model_name}_item_mapping.json")
            
            with open(user_mapping_path, 'r') as f:
                self.user_mapping = json.load(f)
            
            with open(item_mapping_path, 'r') as f:
                self.item_mapping = json.load(f)
            
            # Create reverse mappings
            self.user_reverse = {int(v): k for k, v in self.user_mapping.items()}
            self.item_reverse = {int(v): k for k, v in self.item_mapping.items()}
            
            logger.info(f"ALS model loaded successfully:")
            logger.info(f"  Users: {len(self.user_mapping)}")
            logger.info(f"  Items: {len(self.item_mapping)}")
            logger.info(f"  Factors: {self.user_factors.shape[1]}")
            
        except Exception as e:
            logger.error(f"Failed to load ALS model: {e}")
            raise
    
    def recommend(self, user_id: str, limit: int = 10, filters: Optional[Dict] = None) -> Dict:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID (UUID string)
            limit: Number of recommendations to return
            filters: Optional filters (type, category, field, etc.)
        
        Returns:
            dict with:
                - startups: List of startup recommendations
                - total: Number of recommendations
                - scores: Dict of startup_id -> score
                - method_used: 'als'
        """
        db = SessionLocal()
        try:
            logger.info(f"ALS inference for user {user_id}, limit {limit}")
            
            # Check if user exists in model
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not in ALS model, using fallback")
                return self._fallback_popular(limit, filters, db)
            
            # Get user embedding
            user_idx = self.user_mapping[user_id]
            user_embedding = self.user_factors[user_idx]
            
            # Compute scores for all items
            scores = self.item_factors.dot(user_embedding)
            
            # Get top candidates (more than limit for filtering)
            top_k_indices = np.argsort(scores)[::-1][:limit * 5]
            
            # Convert to startup IDs and scores
            candidate_ids = []
            candidate_scores = {}
            for idx in top_k_indices:
                if idx in self.item_reverse:
                    startup_id = self.item_reverse[idx]
                    candidate_ids.append(startup_id)
                    candidate_scores[startup_id] = float(scores[idx])
            
            # Query startups from database
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
                if len(sorted_startups) >= limit:
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


