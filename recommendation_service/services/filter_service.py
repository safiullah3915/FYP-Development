"""
Filter Service
Multi-layered filtering for high-quality recommendations
"""
from datetime import datetime, timedelta
from sqlalchemy import func
from database.models import Startup, Application, Favorite, UserInteraction, Position
from utils.logger import get_logger

logger = get_logger(__name__)


class FilterService:
    """Ensures users only see relevant, high-quality, actionable startups"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.min_description_length = 50
        self.max_startup_age_days = 365
        self.min_profile_completeness = 0.6
    
    # === QUALITY FILTERS (Always Applied) ===
    
    def filter_active_only(self, query):
        """Only active startups - no sold/paused/inactive"""
        return query.filter(Startup.status == 'active')
    
    def filter_quality_content(self, query):
        """
        Ensure startups have meaningful content
        - Description > 50 chars
        - Has valid title
        - Has category/field set
        """
        return query.filter(
            func.length(Startup.description) >= self.min_description_length,
            Startup.title.isnot(None),
            Startup.category.isnot(None),
            Startup.field.isnot(None)
        )
    
    def filter_fresh_startups(self, query, max_age_days=365):
        """
        Prioritize recent startups (created within last year)
        Prevents showing stale/abandoned listings
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        return query.filter(Startup.created_at >= cutoff_date)
    
    def filter_has_embeddings(self, query):
        """Only recommend startups with valid embeddings"""
        return query.filter(Startup.profile_embedding.isnot(None))
    
    # === USER-SPECIFIC FILTERS ===
    
    def filter_already_applied(self, query, user_id):
        """
        Exclude startups user already applied to
        Prevents wasting user's time on duplicate applications
        """
        try:
            applied_startup_ids = self.db.query(Application.startup_id).filter(
                Application.applicant_id == user_id
            ).subquery()
            
            return query.filter(~Startup.id.in_(applied_startup_ids))
        except Exception as e:
            logger.error(f"Error filtering applied startups: {e}")
            return query
    
    def filter_already_favorited(self, query, user_id, exclude=False):
        """
        Option to exclude/include favorited startups
        exclude=True: Show new discoveries
        exclude=False: Can include favorites for re-engagement
        """
        try:
            favorited_ids = self.db.query(Favorite.startup_id).filter(
                Favorite.user_id == user_id
            ).subquery()
            
            if exclude:
                return query.filter(~Startup.id.in_(favorited_ids))
            return query
        except Exception as e:
            logger.error(f"Error filtering favorited startups: {e}")
            return query
    
    def filter_own_startups(self, query, user_id):
        """Never recommend user's own startups"""
        return query.filter(Startup.owner_id != user_id)
    
    def filter_negative_interactions(self, query, user_id):
        """
        Exclude startups user explicitly disliked
        Respects user preferences and improves satisfaction
        """
        try:
            disliked_ids = self.db.query(UserInteraction.startup_id).filter(
                UserInteraction.user_id == user_id,
                UserInteraction.interaction_type == 'dislike'
            ).subquery()
            
            return query.filter(~Startup.id.in_(disliked_ids))
        except Exception as e:
            logger.error(f"Error filtering negative interactions: {e}")
            return query
    
    # === ROLE-BASED FILTERS ===
    
    def filter_by_type(self, query, startup_type):
        """marketplace or collaboration"""
        if startup_type:
            return query.filter(Startup.type == startup_type)
        return query
    
    def filter_by_category(self, query, categories):
        """Filter by categories (SaaS, ecommerce, web3, etc.)"""
        if categories:
            if isinstance(categories, str):
                categories = [categories]
            return query.filter(Startup.category.in_(categories))
        return query
    
    def filter_by_field(self, query, fields):
        """Filter by startup fields"""
        if fields:
            if isinstance(fields, str):
                fields = [fields]
            return query.filter(Startup.field.in_(fields))
        return query
    
    def filter_has_open_positions(self, query):
        """
        For developers: Only show startups with active positions
        Ensures actionable recommendations
        """
        try:
            startups_with_positions = self.db.query(Position.startup_id).filter(
                Position.is_active == True
            ).distinct().subquery()
            
            return query.filter(Startup.id.in_(startups_with_positions))
        except Exception as e:
            logger.error(f"Error filtering by open positions: {e}")
            return query
    
    def filter_for_investors(self, query):
        """
        For investors: Only marketplace startups with financial data
        Ensures investment-ready opportunities
        """
        return query.filter(
            Startup.type == 'marketplace',
            Startup.revenue.isnot(None),
            Startup.revenue != '',
            Startup.asking_price.isnot(None),
            Startup.asking_price != ''
        )
    
    # === MASTER FILTER ORCHESTRATOR ===
    
    def apply_base_quality_filters(self, query):
        """
        Apply all quality filters that should ALWAYS be applied
        Ensures baseline quality for all recommendations
        """
        query = self.filter_active_only(query)
        query = self.filter_quality_content(query)
        query = self.filter_has_embeddings(query)
        return query

    def apply_min_quality_filters(self, query):
        """
        Relaxed quality baseline (used as a fallback)
        Skips embedding requirement to avoid empty results on cold datasets
        """
        query = self.filter_active_only(query)
        query = self.filter_quality_content(query)
        return query
    
    def apply_user_filters(self, query, user_id, user_role):
        """
        Apply user-specific filters
        Personalizes based on user history and preferences
        """
        query = self.filter_own_startups(query, user_id)
        query = self.filter_already_applied(query, user_id)
        query = self.filter_negative_interactions(query, user_id)
        
        # Role-specific filters
        if user_role == 'student':
            query = self.filter_has_open_positions(query)
        elif user_role == 'investor':
            query = self.filter_for_investors(query)
        
        return query
    
    def apply_all_filters(self, base_query, user_id, user_role, filters_dict=None):
        """
        Master method: Apply all relevant filters
        
        Order matters for SQL optimization:
        1. Quality filters (reduces dataset significantly)
        2. User filters (personalization)
        3. Optional filters (from API params)
        
        Args:
            base_query: Base SQLAlchemy query
            user_id: User ID (UUID string)
            user_role: User role ('student', 'investor', 'entrepreneur')
            filters_dict: Optional filters from API
                - type: 'marketplace' or 'collaboration'
                - category: string or list
                - field: string or list
                - fresh_only: bool
        
        Returns:
            Filtered SQLAlchemy query
        """
        try:
            # Step 1: Quality baseline
            query = self.apply_base_quality_filters(base_query)
            
            # Step 2: User personalization
            query = self.apply_user_filters(query, user_id, user_role)
            
            # Step 3: Optional filters from API
            if filters_dict:
                if 'type' in filters_dict and filters_dict['type']:
                    query = self.filter_by_type(query, filters_dict['type'])
                if 'category' in filters_dict and filters_dict['category']:
                    query = self.filter_by_category(query, filters_dict['category'])
                if 'field' in filters_dict and filters_dict['field']:
                    query = self.filter_by_field(query, filters_dict['field'])
                if filters_dict.get('fresh_only', False):
                    query = self.filter_fresh_startups(query, max_age_days=90)
                if filters_dict.get('require_open_positions'):
                    query = self.filter_has_open_positions(query)
            
            return query
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            # Return base query with minimal filtering
            return self.apply_base_quality_filters(base_query)
    
    def get_filtered_startup_ids(self, user_id, user_role, filters_dict=None):
        """
        Get list of filtered startup IDs
        
        Args:
            user_id: User ID
            user_role: User role
            filters_dict: Optional filters
            
        Returns:
            list: List of startup ID strings
        """
        try:
            base_query = self.db.query(Startup.id)
            filtered_query = self.apply_all_filters(base_query, user_id, user_role, filters_dict)

            # First pass
            startup_ids = [str(row.id) for row in filtered_query.all()]
            if startup_ids:
                logger.info(f"Filtered to {len(startup_ids)} startups for user {user_id}")
                return startup_ids

            # Fallback 1: relax embedding requirement
            logger.info("No candidates after strict filters; retrying without embedding requirement")
            relaxed_query = self.apply_min_quality_filters(base_query)
            # Re-apply user filters without embedding constraint
            relaxed_query = self.filter_own_startups(relaxed_query, user_id)
            relaxed_query = self.filter_already_applied(relaxed_query, user_id)
            relaxed_query = self.filter_negative_interactions(relaxed_query, user_id)

            if user_role == 'student':
                relaxed_query = self.filter_has_open_positions(relaxed_query)
            elif user_role == 'investor':
                relaxed_query = self.filter_for_investors(relaxed_query)

            # Optional filters
            if filters_dict:
                if 'type' in filters_dict and filters_dict['type']:
                    relaxed_query = self.filter_by_type(relaxed_query, filters_dict['type'])
                if 'category' in filters_dict and filters_dict['category']:
                    relaxed_query = self.filter_by_category(relaxed_query, filters_dict['category'])
                if 'field' in filters_dict and filters_dict['field']:
                    relaxed_query = self.filter_by_field(relaxed_query, filters_dict['field'])

            startup_ids = [str(row.id) for row in relaxed_query.all()]
            if startup_ids:
                logger.info(f"Relaxed filtering returned {len(startup_ids)} startups for user {user_id}")
                return startup_ids

            # Fallback 2: if explicitly requiring open positions, try once without it
            if filters_dict and filters_dict.get('require_open_positions'):
                logger.info("Still empty; final retry without open-positions constraint")
                filters_clone = dict(filters_dict)
                filters_clone.pop('require_open_positions', None)
                final_query = self.apply_all_filters(base_query, user_id, user_role, filters_clone)
                startup_ids = [str(row.id) for row in final_query.all()]
                logger.info(f"Final relaxed filtering returned {len(startup_ids)} startups for user {user_id}")
                return startup_ids

            logger.info(f"No startups found for user {user_id} after all fallbacks")
            return []
            
        except Exception as e:
            logger.error(f"Error getting filtered startup IDs: {e}")
            return []

