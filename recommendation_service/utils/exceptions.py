"""
Custom exceptions for recommendation system
"""


class RecommendationError(Exception):
    """Base exception for recommendation errors"""
    pass


class InsufficientDataError(RecommendationError):
    """User/startup has no usable data"""
    pass


class EmbeddingNotFoundError(RecommendationError):
    """Missing embeddings"""
    pass


class InvalidInputError(RecommendationError):
    """Bad parameters"""
    pass

