from django.core.cache import cache
from django.utils import timezone
from typing import Optional, Dict
import logging
import uuid

logger = logging.getLogger(__name__)


class RecommendationCacheService:
    """
    Abstraction layer for recommendation caching
    Currently uses Django cache (LocMem), but can switch to Redis without code changes
    """
    
    CACHE_PREFIX = 'rec'
    RECOMMENDATION_TTL = 60  # seconds
    TRENDING_TTL = 30  # seconds
    INDICATOR_TTL = 300  # seconds to remember last bump
    @classmethod
    def _make_version_key(cls, *parts: str) -> str:
        return cls._make_key('version', *parts)

    @classmethod
    def _generate_version(cls) -> str:
        return f"{int(timezone.now().timestamp())}-{uuid.uuid4().hex[:6]}"

    @classmethod
    def _get_or_create_version(cls, *parts: str) -> str:
        key = cls._make_version_key(*parts)
        version = cache.get(key)
        if not version:
            version = cls._generate_version()
            cache.set(key, version, timeout=cls.INDICATOR_TTL)
        return version

    @classmethod
    def bump_version(cls, *parts: str) -> str:
        version = cls._generate_version()
        key = cls._make_version_key(*parts)
        cache.set(key, version, timeout=cls.INDICATOR_TTL)
        logger.debug("Cache version bump", extra={"key": key, "version": version})
        return version

    @classmethod
    def _set_indicator(cls, indicator_type: str, indicator_id: str, payload: Dict) -> None:
        key = cls._make_key('indicator', indicator_type, indicator_id)
        cache.set(key, {
            'data': payload,
            'timestamp': timezone.now().isoformat()
        }, timeout=cls.INDICATOR_TTL)

    @classmethod
    def record_interaction_indicator(cls, user_id: str, startup_id: str, interaction_type: str) -> None:
        cls._set_indicator('user', user_id, {
            'startup_id': startup_id,
            'interaction_type': interaction_type
        })

    @classmethod
    def record_startup_indicator(cls, startup_id: str, reason: str) -> None:
        cls._set_indicator('startup', startup_id, {'reason': reason})

    @classmethod
    def record_user_profile_indicator(cls, user_id: str, reason: str) -> None:
        cls._set_indicator('user_profile', user_id, {'reason': reason})

    @classmethod
    def _wrap_payload(cls, data, ttl: int, version: str, meta: Optional[Dict] = None) -> Dict:
        payload = {
            'data': data,
            'cached_at': timezone.now().isoformat(),
            'ttl_seconds': ttl,
            'version': version,
        }
        if meta:
            payload['meta'] = meta
        return payload

    @classmethod
    def _is_payload_valid(cls, payload: Optional[Dict], expected_version: str) -> bool:
        if not payload:
            return False
        if payload.get('version') != expected_version:
            return False
        return True

    
    @classmethod
    def _make_key(cls, *parts: str) -> str:
        """Create cache key from parts"""
        return f"{cls.CACHE_PREFIX}:{':'.join(str(p) for p in parts)}"
    
    @classmethod
    def cache_recommendations(cls, user_id: str, use_case: str, data: Dict, ttl: int = None) -> bool:
        """Cache recommendations for a user"""
        try:
            key = cls._make_key('user', use_case, user_id)
            version = cls._get_or_create_version('user', user_id)
            payload = cls._wrap_payload(
                data,
                ttl or cls.RECOMMENDATION_TTL,
                version,
                meta={'use_case': use_case}
            )
            cache.set(key, payload, timeout=ttl or cls.RECOMMENDATION_TTL)
            return True
        except Exception as e:
            logger.error(f"Failed to cache recommendations: {e}")
            return False
    
    @classmethod
    def get_cached_recommendations(cls, user_id: str, use_case: str) -> Optional[Dict]:
        """Get cached recommendations"""
        try:
            key = cls._make_key('user', use_case, user_id)
            payload = cache.get(key)
            expected_version = cls._get_or_create_version('user', user_id)
            if cls._is_payload_valid(payload, expected_version):
                return payload.get('data')
            return None
        except Exception as e:
            logger.error(f"Failed to get cached recommendations: {e}")
            return None
    
    @classmethod
    def invalidate_user_recommendations(cls, user_id: str) -> bool:
        """Invalidate all cached recommendations for a user"""
        try:
            use_cases = ['developer_startup', 'founder_developer', 'founder_startup', 'investor_startup', 'founder_investor']
            keys = [cls._make_key('user', use_case, user_id) for use_case in use_cases]
            cache.delete_many(keys)
            cls.bump_version('user', user_id)
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

    # Trending cache helpers -------------------------------------------------
    @classmethod
    def cache_trending(cls, params_fingerprint: str, data: Dict, ttl: int = None) -> bool:
        try:
            key = cls._make_key('trending', params_fingerprint)
            version = cls._get_or_create_version('trending', 'global')
            payload = cls._wrap_payload(data, ttl or cls.TRENDING_TTL, version, meta={'params': params_fingerprint})
            cache.set(key, payload, timeout=ttl or cls.TRENDING_TTL)
            return True
        except Exception as e:
            logger.error(f"Failed to cache trending startups: {e}")
            return False

    @classmethod
    def get_cached_trending(cls, params_fingerprint: str) -> Optional[Dict]:
        try:
            key = cls._make_key('trending', params_fingerprint)
            payload = cache.get(key)
            expected_version = cls._get_or_create_version('trending', 'global')
            if cls._is_payload_valid(payload, expected_version):
                return payload.get('data')
            return None
        except Exception as e:
            logger.error(f"Failed to get cached trending data: {e}")
            return None

    @classmethod
    def invalidate_trending(cls) -> None:
        cls.bump_version('trending', 'global')

    @classmethod
    def get_user_indicator(cls, user_id: str) -> str:
        return cls._get_or_create_version('user', user_id)

# Future: Redis implementation (drop-in replacement)
# class RedisRecommendationCacheService(RecommendationCacheService):
#     # Override methods to use Redis directly
#     # No other code needs to change

