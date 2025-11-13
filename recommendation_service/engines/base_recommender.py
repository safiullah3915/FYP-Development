"""
Base recommender interface
All recommendation engines must implement this interface
"""
from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Abstract base class for all recommendation engines"""
    
    @abstractmethod
    def recommend(self, user_id, use_case, limit, filters=None):
        """
        Generate recommendations
        
        Args:
            user_id: User ID requesting recommendations
            use_case: Type of recommendation (developer_startup, founder_developer, etc.)
            limit: Maximum number of recommendations
            filters: Optional filters dict
            
        Returns:
            dict with:
                - item_ids: list of recommended item IDs (startup_ids or user_ids)
                - scores: dict mapping item_id to score
                - match_reasons: dict mapping item_id to list of reasons
        """
        pass
    
    @abstractmethod
    def explain(self, user_id, item_id, use_case):
        """
        Generate explanation for a specific recommendation
        
        Args:
            user_id: User ID
            item_id: Item ID (startup or user)
            use_case: Type of recommendation
            
        Returns:
            list of match reasons
        """
        pass

