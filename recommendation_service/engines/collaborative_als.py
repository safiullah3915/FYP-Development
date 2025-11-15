"""
ALS Collaborative Filtering Recommender
Uses Alternating Least Squares for implicit feedback recommendation
"""
import os
import json
import pickle
import numpy as np
from .base_recommender import BaseRecommender
from utils.logger import get_logger
from database.models import Startup, Position

logger = get_logger(__name__)


class ALSRecommender(BaseRecommender):
    """
    ALS-based collaborative filtering recommender
    
    Uses implicit feedback signals (views, clicks, likes) to learn
    user and item latent factors via matrix factorization
    """
    
    def __init__(self, db_session=None, model_path=None):
        """
        Initialize ALS recommender
        
        Args:
            db_session: Database session for querying startup data
            model_path: Path to trained ALS model file
        """
        self.db = db_session
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_reverse = None
        self.item_reverse = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained ALS model and mappings
        
        Args:
            filepath: Path to model file (e.g., models/als_v1.pkl)
        """
        try:
            logger.info(f"Loading ALS model from {model_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load factors
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('.pkl', '')
            
            user_factors_path = os.path.join(model_dir, f"{model_name}_user_factors.npy")
            item_factors_path = os.path.join(model_dir, f"{model_name}_item_factors.npy")
            
            self.user_factors = np.load(user_factors_path)
            self.item_factors = np.load(item_factors_path)
            
            # Load mappings
            user_mapping_path = os.path.join(model_dir, f"{model_name}_user_mapping.json")
            item_mapping_path = os.path.join(model_dir, f"{model_name}_item_mapping.json")
            
            with open(user_mapping_path, 'r') as f:
                self.user_mapping = json.load(f)
            
            with open(item_mapping_path, 'r') as f:
                self.item_mapping = json.load(f)
            
            # Create reverse mappings (index -> UUID)
            self.user_reverse = {int(v): k for k, v in self.user_mapping.items()}
            self.item_reverse = {int(v): k for k, v in self.item_mapping.items()}
            
            logger.info(f"ALS model loaded successfully")
            logger.info(f"  Users: {len(self.user_mapping)}")
            logger.info(f"  Items: {len(self.item_mapping)}")
            logger.info(f"  Factors: {self.user_factors.shape[1]}")
            
        except Exception as e:
            logger.error(f"Error loading ALS model: {e}")
            raise
    
    def recommend(self, user_id, use_case, limit, filters=None):
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID (UUID string)
            use_case: Type of recommendation (e.g., 'developer_startup')
            limit: Maximum number of recommendations
            filters: Optional filters dict (type, category, etc.)
            
        Returns:
            dict with:
                - item_ids: list of recommended startup IDs
                - scores: dict mapping startup ID to score
                - match_reasons: dict mapping startup ID to reasons
        """
        try:
            logger.info(f"ALS recommend for user {user_id}, use_case {use_case}, limit {limit}")
            
            if not self.model or not self.user_factors:
                logger.warning("ALS model not loaded")
                return self._fallback_recommendations(limit, filters)
            
            # Check if user exists in mapping
            if user_id not in self.user_mapping:
                logger.warning(f"User {user_id} not found in ALS model, using fallback")
                return self._fallback_recommendations(limit, filters)
            
            # Get user index and embedding
            user_idx = self.user_mapping[user_id]
            user_embedding = self.user_factors[user_idx]
            
            # Compute scores for all items
            scores = self.item_factors.dot(user_embedding)
            
            # Get top-k items
            top_k_indices = np.argsort(scores)[::-1][:limit * 3]  # Get more for filtering
            
            # Convert indices to item IDs
            candidate_items = []
            candidate_scores = {}
            for idx in top_k_indices:
                if idx in self.item_reverse:
                    item_id = self.item_reverse[idx]
                    candidate_items.append(item_id)
                    candidate_scores[item_id] = float(scores[idx])
            
            # Apply filters
            filtered_items = self._apply_filters(candidate_items, use_case, filters)
            
            # Take top limit
            final_items = filtered_items[:limit]
            
            # Generate match reasons
            match_reasons = {}
            for item_id in final_items:
                match_reasons[item_id] = self.explain(user_id, item_id, use_case)
            
            return {
                'item_ids': final_items,
                'scores': {item_id: candidate_scores[item_id] for item_id in final_items},
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in ALS recommend: {e}", exc_info=True)
            return self._fallback_recommendations(limit, filters)
    
    def _apply_filters(self, item_ids, use_case, filters):
        """Apply filters to candidate items"""
        if not filters or not self.db:
            return item_ids
        
        try:
            # Query startups
            query = self.db.query(Startup).filter(
                Startup.id.in_(item_ids),
                Startup.status == 'active'
            )
            
            # Apply type filter
            if filters.get('type'):
                query = query.filter(Startup.type == filters['type'])
            
            # Apply category filter
            if filters.get('category'):
                categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                query = query.filter(Startup.category.in_(categories))
            
            # Apply field filter
            if filters.get('field'):
                fields = filters['field'] if isinstance(filters['field'], list) else [filters['field']]
                query = query.filter(Startup.field.in_(fields))
            
            # Require at least one active position if requested
            if filters.get('require_open_positions'):
                query = query.join(Position, Position.startup_id == Startup.id).filter(
                    Position.is_active == True
                )
                query = query.distinct()
            
            # Get filtered startup IDs
            filtered_startups = query.all()
            filtered_ids = [str(s.id) for s in filtered_startups]
            
            # Maintain original order
            result = [item_id for item_id in item_ids if item_id in filtered_ids]
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return item_ids
    
    def _fallback_recommendations(self, limit, filters):
        """Fallback to popular items when ALS can't be used"""
        logger.info("Using fallback recommendations (popular items)")
        
        if not self.db:
            return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
        
        try:
            # Query popular startups
            query = self.db.query(Startup).filter(Startup.status == 'active')
            
            # Apply filters if provided
            if filters:
                if filters.get('type'):
                    query = query.filter(Startup.type == filters['type'])
                if filters.get('category'):
                    categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                    query = query.filter(Startup.category.in_(categories))
            
            # Order by views and get top limit
            startups = query.order_by(Startup.views.desc()).limit(limit).all()
            
            item_ids = [str(s.id) for s in startups]
            scores = {str(s.id): float(s.views) for s in startups}
            match_reasons = {str(s.id): ["Popular startup"] for s in startups}
            
            return {
                'item_ids': item_ids,
                'scores': scores,
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
    
    def explain(self, user_id, item_id, use_case):
        """
        Generate explanation for why item was recommended
        
        Args:
            user_id: User ID
            item_id: Item ID (startup)
            use_case: Use case type
            
        Returns:
            list: Match reasons
        """
        reasons = []
        
        try:
            if not self.model or user_id not in self.user_mapping or item_id not in self.item_mapping:
                return ["Recommended based on collaborative filtering"]
            
            # Get user and item indices
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            
            # Get embeddings
            user_emb = self.user_factors[user_idx]
            item_emb = self.item_factors[item_idx]
            
            # Compute similarity score
            score = np.dot(user_emb, item_emb)
            
            reasons.append(f"Collaborative filtering score: {score:.3f}")
            reasons.append("Users with similar tastes also engaged with this startup")
            
            # Find similar items (top 3)
            item_scores = self.item_factors.dot(item_emb)
            top_similar_indices = np.argsort(item_scores)[::-1][1:4]  # Exclude self
            
            similar_count = 0
            for idx in top_similar_indices:
                if idx in self.item_reverse:
                    similar_count += 1
            
            if similar_count > 0:
                reasons.append(f"Similar to {similar_count} other startups you've engaged with")
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            reasons = ["Recommended based on collaborative filtering patterns"]
        
        return reasons
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.model:
            logger.warning("No model to save")
            return
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"ALS model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ALS model: {e}")
            raise
    
    def train(self, interactions_df):
        """
        Train ALS model (placeholder for compatibility)
        
        Note: Use train_als.py script for actual training
        
        Args:
            interactions_df: DataFrame with user_id, startup_id, weight columns
        """
        logger.warning("ALS training should be done using train_als.py script")
        logger.info("This method is a placeholder for BaseRecommender interface compatibility")
