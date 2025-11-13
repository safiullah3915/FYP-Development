"""
ALS Collaborative Filtering (Placeholder)
TODO: Implement ALS using implicit library
"""
from .base_recommender import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class ALSRecommender(BaseRecommender):
    """
    TODO: Implement ALS using implicit library
    
    Will train on UserInteraction data:
    - user_id x startup_id matrix
    - weights from interaction_type
    - Implicit feedback (no explicit ratings)
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self.model = None
    
    def train(self, interactions_df):
        """
        TODO: Implement training
        
        Args:
            interactions_df: DataFrame with user_id, startup_id, weight columns
        """
        logger.warning("ALS training not yet implemented")
        pass
    
    def recommend(self, user_id, use_case, limit, filters=None):
        """
        TODO: Implement recommendations
        
        Args:
            user_id: User ID
            use_case: Use case type
            limit: Max recommendations
            filters: Optional filters
            
        Returns:
            dict with item_ids, scores, match_reasons
        """
        logger.warning("ALS recommendations not yet implemented")
        return {
            'item_ids': [],
            'scores': {},
            'match_reasons': {}
        }
    
    def explain(self, user_id, item_id, use_case):
        """TODO: Generate explanations"""
        return ["ALS explanations not yet implemented"]
    
    def save_model(self, filepath):
        """
        TODO: Save to recommendation_service/models/
        
        Args:
            filepath: Path to save model
        """
        logger.warning("ALS model saving not yet implemented")
        pass
    
    def load_model(self, filepath):
        """
        TODO: Load from disk
        
        Args:
            filepath: Path to load model from
        """
        logger.warning("ALS model loading not yet implemented")
        pass

