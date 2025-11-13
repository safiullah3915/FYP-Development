"""
High-Level Recommendation Service
Orchestrates the entire recommendation flow
"""
from engines.router import RecommendationRouter
from engines.content_based import ContentBasedRecommender
from engines.two_tower import TwoTowerRecommender
from engines.model_registry import get_registry
from services.session_service import SessionService
from utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """Main service for generating recommendations"""
    
    def __init__(self, db_session, enable_two_tower: bool = False):
        """
        Initialize recommendation service
        
        Args:
            db_session: Database session
            enable_two_tower: Enable two-tower model (default: False)
        """
        self.db = db_session
        self.router = RecommendationRouter(enable_two_tower=enable_two_tower)
        self.content_based = ContentBasedRecommender(db_session)
        self.session_service = SessionService()
        
        # Initialize two-tower if enabled
        self.two_tower = None
        if enable_two_tower:
            self._initialize_two_tower()
    
    def _initialize_two_tower(self):
        """Initialize two-tower model"""
        try:
            logger.info("Initializing Two-Tower model...")
            registry = get_registry()
            
            # Get active model paths
            model_paths = registry.get_active_model(
                use_case='developer_startup',
                model_type='two_tower'
            )
            
            if model_paths:
                self.two_tower = TwoTowerRecommender(
                    db_session=self.db,
                    model_path=model_paths['model_path'],
                    encoder_path=model_paths['encoder_path']
                )
                logger.info("Two-Tower model loaded successfully")
            else:
                logger.warning("No Two-Tower model found, will fallback to content-based")
                
        except Exception as e:
            logger.error(f"Error initializing Two-Tower model: {e}")
            self.two_tower = None
    
    def get_recommendations(self, user_id, use_case, limit=10, filters=None):
        """
        Main entry point for recommendations
        
        Args:
            user_id: User ID requesting recommendations
            use_case: Type of recommendation
                - 'developer_startup': Student/professional → Startup
                - 'investor_startup': Investor → Startup
                - 'founder_developer': Founder → Developer
                - 'founder_investor': Founder → Investor
                - 'founder_startup': Founder → Startup
            limit: Maximum number of recommendations
            filters: Optional filters dict
                - type: 'marketplace' or 'collaboration'
                - category: string or list
                - field: string or list
                - fresh_only: bool
        
        Returns:
            dict with:
                - item_ids: list of recommended IDs
                - scores: dict mapping ID to score
                - match_reasons: dict mapping ID to reasons
                - method_used: string
                - interaction_count: int
        """
        try:
            logger.info(f"Getting recommendations for user {user_id}, use_case {use_case}, limit {limit}")
            
            # 1. Route to appropriate engine
            method, interaction_count = self.router.route(user_id, use_case)
            
            logger.info(f"Routed to {method} (interaction_count: {interaction_count})")
            
            # 2. Get recommendations from engine
            if method == 'content_based':
                results = self.content_based.recommend(user_id, use_case, limit, filters)
            elif method == 'two_tower':
                # Use two-tower if available, otherwise fallback
                if self.two_tower:
                    results = self.two_tower.recommend(user_id, use_case, limit, filters)
                else:
                    logger.warning("Two-Tower not available, falling back to content_based")
                    results = self.content_based.recommend(user_id, use_case, limit, filters)
                    method = 'content_based'
            else:
                # Future: other methods (collaborative, ensemble)
                logger.warning(f"Method {method} not implemented, falling back to content_based")
                results = self.content_based.recommend(user_id, use_case, limit, filters)
                method = 'content_based'
            
            # 3. Add metadata
            results['method_used'] = method
            results['interaction_count'] = interaction_count
            
            logger.info(f"Generated {len(results.get('item_ids', []))} recommendations")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in get_recommendations: {e}", exc_info=True)
            return {
                'item_ids': [],
                'scores': {},
                'match_reasons': {},
                'method_used': 'error',
                'interaction_count': 0
            }
    
    def explain_recommendation(self, user_id, item_id, use_case):
        """
        Explain why an item was recommended
        
        Args:
            user_id: User ID
            item_id: Item ID (startup or user)
            use_case: Type of recommendation
            
        Returns:
            list: Match reasons
        """
        try:
            return self.content_based.explain(user_id, item_id, use_case)
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return ["Unable to generate explanation"]

