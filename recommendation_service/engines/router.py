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
    
    def __init__(self, enable_two_tower: bool = False, enable_als: bool = False, enable_ensemble: bool = False):
        """
        Initialize router
        
        Args:
            enable_two_tower: Enable two-tower model routing (default: False)
            enable_als: Enable ALS collaborative filtering (default: False)
            enable_ensemble: Enable ensemble (ALS + Two-Tower) (default: False)
        """
        self.cold_start_threshold = self.COLD_START_THRESHOLD
        self.warm_user_threshold = self.WARM_USER_THRESHOLD
        self.enable_two_tower = enable_two_tower
        self.enable_als = enable_als
        self.enable_ensemble = enable_ensemble
    
    def route(self, user_id, use_case):
        """
        Determine which engine to use based on interaction count
        
        Routing logic:
        - Cold start (< 5 interactions): content_based
        - Warm users (5-19 interactions): ALS if enabled, else two_tower, else content_based
        - Hot users (>= 20 interactions): ensemble if enabled, else ALS, else two_tower, else content_based
        
        Args:
            user_id: User ID (UUID string)
            use_case: Type of recommendation
            
        Returns:
            tuple: (method_name, interaction_count)
                - method_name: 'content_based', 'als', 'two_tower', or 'ensemble'
                - interaction_count: number of interactions user has
        """
        try:
            interaction_count = self._get_interaction_count(user_id)
            
            logger.info(f"User {user_id} has {interaction_count} interactions")
            
            # Cold start users: always use content-based
            if interaction_count < self.cold_start_threshold:
                logger.info(f"Routing to content_based (cold start)")
                return 'content_based', interaction_count
            
            # Warm users (5-19 interactions): prefer ALS
            elif interaction_count < self.warm_user_threshold:
                if self.enable_als:
                    logger.info(f"Routing to ALS (warm user: {interaction_count} interactions)")
                    return 'als', interaction_count
                elif self.enable_two_tower:
                    logger.info(f"Routing to two_tower (warm user, ALS disabled)")
                    return 'two_tower', interaction_count
                else:
                    logger.info(f"Routing to content_based (warm user, models disabled)")
                    return 'content_based', interaction_count
            
            # Hot users (>= 20 interactions): prefer ensemble
            else:
                if self.enable_ensemble:
                    logger.info(f"Routing to ensemble (hot user: {interaction_count} interactions)")
                    return 'ensemble', interaction_count
                elif self.enable_als:
                    logger.info(f"Routing to ALS (hot user, ensemble disabled)")
                    return 'als', interaction_count
                elif self.enable_two_tower:
                    logger.info(f"Routing to two_tower (hot user, ALS/ensemble disabled)")
                    return 'two_tower', interaction_count
                else:
                    logger.info(f"Routing to content_based (hot user, all models disabled)")
                    return 'content_based', interaction_count
                
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Fallback to content_based on error
            return 'content_based', 0
    
    def route_reverse(self, startup_id, use_case):
        """
        Determine which engine to use for reverse recommendations (Startup → User)
        Based on how many unique users have interacted with the startup
        
        Routing logic:
        - Cold start (< 5 unique users): content_based
        - Warm/Hot startups (>= 5 unique users): als_reverse
        
        Args:
            startup_id: Startup ID (UUID string)
            use_case: Type of recommendation ('founder_developer' or 'founder_investor')
            
        Returns:
            tuple: (method_name, interaction_count)
                - method_name: 'content_based' or 'als_reverse'
                - interaction_count: number of unique users who interacted with startup
        """
        try:
            interaction_count = self._get_startup_interaction_count(startup_id)
            
            logger.info(f"Startup {startup_id} has {interaction_count} unique user interactions")
            
            # Cold start startups: use content-based
            if interaction_count < self.cold_start_threshold:
                logger.info(f"Routing to content_based (cold start startup)")
                return 'content_based', interaction_count
            
            # Warm/Hot startups: use ALS reverse
            else:
                logger.info(f"Routing to ALS Reverse (warm/hot startup: {interaction_count} interactions)")
                return 'als_reverse', interaction_count
                
        except Exception as e:
            logger.error(f"Error in reverse routing: {e}")
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
    
    def _get_startup_interaction_count(self, startup_id):
        """
        Get count of unique users who have interacted with a startup
        
        Args:
            startup_id: Startup ID (UUID string)
            
        Returns:
            int: Number of unique users who interacted
        """
        db = SessionLocal()
        try:
            # Count distinct users who interacted with this startup
            count = db.query(UserInteraction).filter(
                UserInteraction.startup_id == startup_id
            ).distinct(UserInteraction.user_id).count()
            return count
        except Exception as e:
            logger.error(f"Error getting startup interaction count: {e}")
            return 0
        finally:
            db.close()

