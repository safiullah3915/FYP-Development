from django.core.cache import cache
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class RecommendationCacheService:
    """
    Abstraction layer for recommendation caching
    Currently uses Django cache (LocMem), but can switch to Redis without code changes
    """
    
    CACHE_PREFIX = 'rec'
    DEFAULT_TTL = 3600  # 1 hour
    
    @classmethod
    def _make_key(cls, *parts: str) -> str:
        """Create cache key from parts"""
        return f"{cls.CACHE_PREFIX}:{':'.join(str(p) for p in parts)}"
    
    @classmethod
    def cache_recommendations(cls, user_id: str, use_case: str, recommendations: List[Dict], ttl: int = None) -> bool:
        """Cache recommendations for a user"""
        try:
            key = cls._make_key('user', use_case, user_id)
            cache.set(key, recommendations, timeout=ttl or cls.DEFAULT_TTL)
            return True
        except Exception as e:
            logger.error(f"Failed to cache recommendations: {e}")
            return False
    
    @classmethod
    def get_cached_recommendations(cls, user_id: str, use_case: str) -> Optional[List[Dict]]:
        """Get cached recommendations"""
        try:
            key = cls._make_key('user', use_case, user_id)
            return cache.get(key)
        except Exception as e:
            logger.error(f"Failed to get cached recommendations: {e}")
            return None
    
    @classmethod
    def invalidate_user_recommendations(cls, user_id: str) -> bool:
        """Invalidate all cached recommendations for a user"""
        try:
            # For LocMem: clear all (simple)
            # For Redis: would use pattern matching
            # This abstraction allows switching implementations
            use_cases = ['developer_startup', 'founder_developer', 'founder_startup', 'investor_startup']
            keys = [cls._make_key('user', use_case, user_id) for use_case in use_cases]
            cache.delete_many(keys)
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    @classmethod
    def cache_session_metrics(cls, session_id: str, metrics: Dict, ttl: int = 86400) -> bool:
        """Cache session metrics"""
        try:
            key = cls._make_key('session_metrics', session_id)
            cache.set(key, metrics, timeout=ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to cache session metrics: {e}")
            return False

# Future: Redis implementation (drop-in replacement)
# class RedisRecommendationCacheService(RecommendationCacheService):
#     # Override methods to use Redis directly
#     # No other code needs to change

