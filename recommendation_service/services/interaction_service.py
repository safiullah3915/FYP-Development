"""
Interaction Service
Query and analyze user interaction history
"""
from database.models import UserInteraction, Application
from utils.logger import get_logger

logger = get_logger(__name__)


class InteractionService:
    """Service for querying user interaction history"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def get_user_interaction_count(self, user_id):
        """
        Count total interactions for routing
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            int: Total number of interactions
        """
        try:
            count = self.db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).count()
            return count
        except Exception as e:
            logger.error(f"Error getting interaction count for user {user_id}: {e}")
            return 0
    
    def get_user_interactions(self, user_id, limit=100):
        """
        Get recent interactions with weights
        
        Args:
            user_id: User ID (UUID string)
            limit: Maximum number of interactions to return
            
        Returns:
            list: List of interaction dicts with startup_id, type, weight
        """
        try:
            interactions = self.db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).order_by(UserInteraction.created_at.desc()).limit(limit).all()
            
            return [{
                'startup_id': str(i.startup_id),
                'interaction_type': i.interaction_type,
                'weight': i.weight,
                'created_at': i.created_at,
            } for i in interactions]
            
        except Exception as e:
            logger.error(f"Error getting interactions for user {user_id}: {e}")
            return []
    
    def has_applied_to_startup(self, user_id, startup_id):
        """
        Check if user already applied to startup
        
        Args:
            user_id: User ID (UUID string)
            startup_id: Startup ID (UUID string)
            
        Returns:
            bool: True if user has applied
        """
        try:
            exists = self.db.query(Application).filter(
                Application.applicant_id == user_id,
                Application.startup_id == startup_id
            ).first() is not None
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking application status: {e}")
            return False
    
    def get_negative_interactions(self, user_id):
        """
        Get startups user disliked/rejected
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            set: Set of startup IDs user has negative interactions with
        """
        try:
            interactions = self.db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id,
                UserInteraction.interaction_type == 'dislike'
            ).all()
            
            return {str(i.startup_id) for i in interactions}
            
        except Exception as e:
            logger.error(f"Error getting negative interactions for user {user_id}: {e}")
            return set()
    
    def get_applied_startups(self, user_id):
        """
        Get list of startups user has applied to
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            set: Set of startup IDs user has applied to
        """
        try:
            applications = self.db.query(Application).filter(
                Application.applicant_id == user_id
            ).all()
            
            return {str(app.startup_id) for app in applications}
            
        except Exception as e:
            logger.error(f"Error getting applied startups for user {user_id}: {e}")
            return set()
    
    def get_favorited_startups(self, user_id):
        """
        Get list of startups user has favorited
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            set: Set of startup IDs user has favorited
        """
        try:
            from database.models import Favorite
            favorites = self.db.query(Favorite).filter(
                Favorite.user_id == user_id
            ).all()
            
            return {str(fav.startup_id) for fav in favorites}
            
        except Exception as e:
            logger.error(f"Error getting favorited startups for user {user_id}: {e}")
            return set()

