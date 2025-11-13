"""
Recommendation Router
Routes requests to appropriate recommendation engine based on user interaction history
"""
from database.connection import SessionLocal
from database.models import UserInteraction
from utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationRouter:
    """Routes recommendation requests to appropriate engine"""
    
    COLD_START_THRESHOLD = 5  # interactions
    WARM_USER_THRESHOLD = 20  # interactions for hot users
    
    def __init__(self, enable_two_tower: bool = False):
        """
        Initialize router
        
        Args:
            enable_two_tower: Enable two-tower model routing (default: False)
        """
        self.cold_start_threshold = self.COLD_START_THRESHOLD
        self.warm_user_threshold = self.WARM_USER_THRESHOLD
        self.enable_two_tower = enable_two_tower
    
    def route(self, user_id, use_case):
        """
        Determine which engine to use based on interaction count
        
        Routing logic:
        - Cold start (< 5 interactions): content_based
        - Warm users (5-20 interactions): two_tower if enabled, else content_based
        - Hot users (> 20 interactions): two_tower if enabled, else content_based
        
        Args:
            user_id: User ID (UUID string)
            use_case: Type of recommendation
            
        Returns:
            tuple: (method_name, interaction_count)
                - method_name: 'content_based', 'two_tower', or 'ensemble'
                - interaction_count: number of interactions user has
        """
        try:
            interaction_count = self._get_interaction_count(user_id)
            
            logger.info(f"User {user_id} has {interaction_count} interactions")
            
            # Cold start users: always use content-based
            if interaction_count < self.cold_start_threshold:
                logger.info(f"Routing to content_based (cold start)")
                return 'content_based', interaction_count
            
            # Warm/Hot users: use two-tower if enabled
            elif self.enable_two_tower:
                if interaction_count < self.warm_user_threshold:
                    # Warm users: could use ensemble in future
                    logger.info(f"Routing to two_tower (warm user)")
                    return 'two_tower', interaction_count
                else:
                    # Hot users: two-tower should perform best
                    logger.info(f"Routing to two_tower (hot user)")
                    return 'two_tower', interaction_count
            
            # Fallback to content-based
            else:
                logger.info(f"Routing to content_based (two-tower disabled)")
                return 'content_based', interaction_count
                
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Fallback to content_based on error
            return 'content_based', 0
    
    def _get_interaction_count(self, user_id):
        """
        Get total interaction count for user
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            int: Number of interactions
        """
        db = SessionLocal()
        try:
            count = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).count()
            return count
        except Exception as e:
            logger.error(f"Error getting interaction count: {e}")
            return 0
        finally:
            db.close()

