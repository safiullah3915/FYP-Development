"""
Business Rules Service
Apply domain-specific rules for optimal user experience
"""
import numpy as np
from datetime import datetime, timedelta
from database.models import UserInteraction
from utils.logger import get_logger

logger = get_logger(__name__)


class BusinessRules:
    """Apply intelligent business rules that improve recommendation quality"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.engagement_boost_factor = 1.25
        self.negative_penalty_factor = 0.5
        self.cold_start_boost_factor = 1.15
        self.freshness_weight = 0.15
        self.recency_boost_days = 30
        self.recency_boost_factor = 1.2
    
    def apply_engagement_boost(self, scores, startup_data):
        """
        Boost startups with high engagement (views, applications, favorites)
        Social proof: Popular startups are likely higher quality
        """
        boosted_scores = scores.copy()
        
        for startup_id, data in startup_data.items():
            try:
                # Calculate engagement score
                views_7d = data.get('view_count_7d', 0)
                applications_7d = data.get('application_count_7d', 0)
                favorites_7d = data.get('favorite_count_7d', 0)
                
                # Weighted engagement score
                engagement_score = (
                    views_7d * 0.1 +
                    applications_7d * 2.0 +
                    favorites_7d * 1.5
                )
                
                # Normalize and apply boost (cap at 25%)
                normalized_engagement = min(engagement_score / 50, 1.0)
                boost = normalized_engagement * (self.engagement_boost_factor - 1)
                
                boosted_scores[startup_id] = scores[startup_id] * (1 + boost)
            except Exception as e:
                logger.warning(f"Error applying engagement boost for {startup_id}: {e}")
                continue
        
        return boosted_scores
    
    def apply_freshness_boost(self, scores, startup_data):
        """
        Boost recently created/updated startups
        Exponential decay over 90 days
        """
        boosted_scores = scores.copy()
        current_time = datetime.now()
        
        for startup_id, data in startup_data.items():
            try:
                created_at = data.get('created_at')
                if not created_at:
                    continue
                
                # Calculate age in days
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                
                age_days = (current_time - created_at).days
                
                # Exponential decay: full boost at 0 days, ~0 boost at 90 days
                freshness_boost = self.freshness_weight * np.exp(-age_days / 30)
                
                boosted_scores[startup_id] = scores[startup_id] * (1 + freshness_boost)
            except Exception as e:
                logger.warning(f"Error applying freshness boost for {startup_id}: {e}")
                continue
        
        return boosted_scores
    
    def penalize_negative_interactions(self, scores, user_id, startup_ids):
        """
        Heavily penalize startups user disliked or rejected
        Respects user preferences and improves satisfaction
        """
        try:
            # Get negative interactions
            negative_interactions = self.db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id,
                UserInteraction.startup_id.in_(startup_ids),
                UserInteraction.interaction_type == 'dislike'
            ).all()
            
            penalized_scores = scores.copy()
            
            for interaction in negative_interactions:
                startup_id = str(interaction.startup_id)
                if startup_id in penalized_scores:
                    # Heavy penalty: reduce score by 50%
                    penalized_scores[startup_id] *= self.negative_penalty_factor
            
            return penalized_scores
        except Exception as e:
            logger.error(f"Error applying negative penalties: {e}")
            return scores
    
    def apply_position_availability_boost(self, scores, startup_data):
        """Boost startups with active open positions"""
        boosted_scores = scores.copy()
        
        for startup_id, data in startup_data.items():
            try:
                positions = data.get('positions', [])
                active_positions = [p for p in positions if p.get('is_active', False)]
                
                if len(active_positions) > 0:
                    boost = min(len(active_positions) * 0.05, 0.15)  # Max 15% boost
                    boosted_scores[startup_id] = scores[startup_id] * (1 + boost)
            except Exception as e:
                logger.warning(f"Error applying position boost for {startup_id}: {e}")
                continue
        
        return boosted_scores
    
    def apply_all_business_rules(self, scores, startup_data, user_id, user_role):
        """
        Master method: Apply all relevant business rules
        
        Args:
            scores: Dict of {startup_id: score}
            startup_data: Dict of {startup_id: features}
            user_id: User ID
            user_role: User role
            
        Returns:
            dict: Boosted scores
        """
        try:
            # Apply general rules
            scores = self.apply_freshness_boost(scores, startup_data)
            scores = self.apply_position_availability_boost(scores, startup_data)
            scores = self.penalize_negative_interactions(scores, user_id, list(startup_data.keys()))
            
            # Note: Engagement boost would require trending metrics
            # scores = self.apply_engagement_boost(scores, startup_data)
            
            return scores
        except Exception as e:
            logger.error(f"Error applying business rules: {e}")
            return scores

