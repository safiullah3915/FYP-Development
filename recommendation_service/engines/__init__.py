"""
Recommendation Engines Package
Contains all recommendation algorithms and routing logic
"""

from .router import RecommendationRouter
from .base_recommender import BaseRecommender
__all__ = [
    'RecommendationRouter',
    'BaseRecommender',
]

