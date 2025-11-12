from api.recommendation_models import RecommendationSession
from django.utils import timezone
from datetime import timedelta
from typing import List, Dict, Optional
import uuid


class SessionService:
    """Service for managing recommendation sessions"""
    
    DEFAULT_SESSION_TTL_HOURS = 24
    
    @staticmethod
    def create_session(
        user_id: str,
        use_case: str,
        method: str,
        recommendations: List[Dict],
        model_version: Optional[str] = None,
        ttl_hours: int = None
    ) -> RecommendationSession:
        """Create a new recommendation session"""
        session_id = str(uuid.uuid4())
        expires_at = timezone.now() + timedelta(hours=ttl_hours or SessionService.DEFAULT_SESSION_TTL_HOURS)
        
        # Format recommendations with rank if not present
        formatted_recs = []
        for idx, rec in enumerate(recommendations):
            if isinstance(rec, dict):
                if 'rank' not in rec:
                    rec['rank'] = idx + 1
                formatted_recs.append(rec)
            else:
                # If rec is just startup_id
                formatted_recs.append({
                    'startup_id': str(rec),
                    'rank': idx + 1,
                    'score': 0.0
                })
        
        session = RecommendationSession.objects.create(
            id=session_id,
            user_id_id=user_id,
            use_case=use_case,
            recommendation_method=method,
            model_version=model_version or '',
            recommendations_shown=formatted_recs,
            expires_at=expires_at
        )
        
        return session
    
    @staticmethod
    def get_valid_session(session_id: str, user_id: str) -> Optional[RecommendationSession]:
        """Get session if it exists and is valid"""
        try:
            session = RecommendationSession.objects.get(id=session_id, user_id_id=user_id)
            if session.expires_at and session.expires_at < timezone.now():
                return None  # Expired
            return session
        except RecommendationSession.DoesNotExist:
            return None

