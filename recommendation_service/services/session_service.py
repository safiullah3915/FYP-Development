"""
Session Tracking Service
Create and format recommendation sessions for storage
"""
import uuid
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionService:
    """Service for creating and managing recommendation sessions"""
    
    def __init__(self):
        self.session_ttl_hours = 24
    
    def create_session_data(self, user_id, use_case, method, recommendations, model_version='content_based_v1.0'):
        """
        Format session for Django backend storage
        
        Args:
            user_id: User ID (UUID string)
            use_case: Type of recommendation (developer_startup, etc.)
            method: Method used (content_based, collaborative, etc.)
            recommendations: Dict with item_ids, scores, match_reasons
            model_version: Model version string
            
        Returns:
            dict: Formatted session data ready for storage
        """
        try:
            session_id = str(uuid.uuid4())
            created_at = datetime.now()
            expires_at = created_at + timedelta(hours=self.session_ttl_hours)
            
            # Format recommendations with rank
            recommendations_with_rank = []
            item_ids = recommendations.get('item_ids', [])
            scores = recommendations.get('scores', {})
            match_reasons = recommendations.get('match_reasons', {})
            
            for idx, item_id in enumerate(item_ids):
                rec_data = {
                    'item_id': str(item_id),  # Can be startup_id or user_id
                    'rank': idx + 1,
                    'score': float(scores.get(item_id, 0.0)),
                    'match_reasons': match_reasons.get(item_id, [])
                }
                recommendations_with_rank.append(rec_data)
            
            session_data = {
                'recommendation_session_id': session_id,
                'user_id': str(user_id),
                'use_case': use_case,
                'method': method,
                'model_version': model_version,
                'recommendations': recommendations_with_rank,
                'created_at': created_at.isoformat(),
                'expires_at': expires_at.isoformat()
            }
            
            logger.info(f"Created session {session_id} for user {user_id}, use_case {use_case}")
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error creating session data: {e}")
            # Return minimal session data
            return {
                'recommendation_session_id': str(uuid.uuid4()),
                'user_id': str(user_id),
                'use_case': use_case,
                'method': method,
                'model_version': model_version,
                'recommendations': [],
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=self.session_ttl_hours)).isoformat()
            }
    
    def format_for_api_response(self, session_data, recommendations):
        """
        Format session data for API response
        
        Args:
            session_data: Session data dict
            recommendations: Recommendations dict
            
        Returns:
            dict: Formatted for Flask API response
        """
        try:
            # Determine if we're returning startup_ids or user_ids
            item_ids = recommendations.get('item_ids', [])
            use_case = session_data.get('use_case', '')
            
            # For founder→developer and founder→investor, return user_ids
            if 'founder_developer' in use_case or 'founder_investor' in use_case:
                id_key = 'user_ids'
            else:
                id_key = 'startup_ids'
            
            response = {
                'recommendation_session_id': session_data['recommendation_session_id'],
                'use_case': session_data['use_case'],
                'method': session_data['method'],
                'model_version': session_data['model_version'],
                id_key: [str(id) for id in item_ids],
                'scores': {str(k): float(v) for k, v in recommendations.get('scores', {}).items()},
                'match_reasons': {str(k): v for k, v in recommendations.get('match_reasons', {}).items()},
                'total': len(item_ids),
                'interaction_count': recommendations.get('interaction_count', 0),
                'created_at': session_data['created_at'],
                'expires_at': session_data['expires_at']
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting API response: {e}")
            return {
                'recommendation_session_id': session_data.get('recommendation_session_id'),
                'use_case': session_data.get('use_case'),
                'method': session_data.get('method'),
                'startup_ids': [],
                'scores': {},
                'match_reasons': {},
                'total': 0,
                'interaction_count': 0,
                'created_at': session_data.get('created_at'),
                'expires_at': session_data.get('expires_at')
            }

