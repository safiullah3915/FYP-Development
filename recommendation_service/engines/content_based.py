"""
Content-Based Recommender
Main recommendation engine using embeddings, preferences, and profile matching
"""
from .base_recommender import BaseRecommender
from .feature_extractor import FeatureExtractor
from .similarity import (
    cosine_similarity_embeddings,
    preference_similarity,
    profile_similarity,
    combine_scores
)
from .match_reasons import (
    generate_match_reasons,
    generate_developer_match_reasons,
    generate_investor_match_reasons
)
from services.filter_service import FilterService
from services.business_rules import BusinessRules
from services.diversity_service import DiversityService
from utils.embedding_utils import batch_load_embeddings
from database.models import Startup, User
from utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Content-based filtering using embeddings, preferences, and profile data"""
    
    # Component weights (33-33-34 split)
    EMBEDDING_WEIGHT = 0.33
    PREFERENCE_WEIGHT = 0.33
    PROFILE_WEIGHT = 0.34
    
    def __init__(self, db_session):
        self.db = db_session
        self.feature_extractor = FeatureExtractor(db_session)
        self.filter_service = FilterService(db_session)
        self.business_rules = BusinessRules(db_session)
        self.diversity_service = DiversityService()
    
    def recommend(self, user_id, use_case, limit, filters=None):
        """
        Generate recommendations
        
        Args:
            user_id: User ID requesting recommendations
            use_case: Type of recommendation
            limit: Maximum number of recommendations
            filters: Optional filters dict
            
        Returns:
            dict with item_ids, scores, match_reasons
        """
        try:
            logger.info(f"Generating recommendations for user {user_id}, use_case {use_case}")
            
            if 'developer_startup' in use_case or 'investor_startup' in use_case or 'founder_startup' in use_case:
                return self.recommend_startups_for_user(user_id, limit, filters, use_case)
            elif 'founder_developer' in use_case:
                return self.recommend_users_for_startup(user_id, limit, filters, 'student')
            elif 'founder_investor' in use_case:
                return self.recommend_users_for_startup(user_id, limit, filters, 'investor')
            else:
                logger.warning(f"Unknown use case: {use_case}")
                return self._empty_result()
                
        except Exception as e:
            logger.error(f"Error in recommend: {e}", exc_info=True)
            return self._empty_result()
    
    def recommend_startups_for_user(self, user_id, limit, filters, use_case):
        """
        Developer/Investor → Startup recommendations
        
        Args:
            user_id: User ID
            limit: Max recommendations
            filters: Optional filters
            use_case: Use case type
            
        Returns:
            dict with startup_ids, scores, match_reasons
        """
        try:
            # 1. Extract user features
            user_features = self.feature_extractor.extract_user_features(user_id)
            user_role = user_features.get('role', 'unknown')
            
            logger.info(f"User role: {user_role}")
            
            # 2. Get filtered candidate startups
            candidate_startup_ids = self.filter_service.get_filtered_startup_ids(
                user_id, user_role, filters
            )
            
            if not candidate_startup_ids:
                logger.warning(f"No candidate startups found for user {user_id}")
                return self._empty_result()
            
            logger.info(f"Found {len(candidate_startup_ids)} candidate startups")
            
            # 3. Extract startup features
            startup_features = self.feature_extractor.extract_startup_features(candidate_startup_ids)
            
            if not startup_features:
                logger.warning(f"No startup features extracted")
                return self._empty_result()
            
            # 4. Calculate similarity scores
            scores = self._calculate_similarity_scores(user_features, startup_features)
            
            if not scores:
                logger.warning(f"No similarity scores calculated")
                return self._empty_result()
            
            # 5. Apply business rules
            scores = self.business_rules.apply_all_business_rules(
                scores, startup_features, user_id, user_role
            )
            
            # 6. Get embeddings for diversity
            user_emb = user_features.get('embedding')
            startup_embeddings = batch_load_embeddings(
                self.db, list(scores.keys()), Startup, 'id'
            )
            
            # 7. Apply diversity
            candidates = list(scores.keys())
            selected_ids = self.diversity_service.apply_all_diversity_strategies(
                candidates, scores, startup_embeddings, startup_features, limit
            )
            
            # 8. Generate match reasons
            match_reasons = {}
            for startup_id in selected_ids:
                if user_role == 'student':
                    reasons = generate_developer_match_reasons(
                        user_features,
                        startup_features[startup_id],
                        {'embedding': scores.get(startup_id, 0)}
                    )
                elif user_role == 'investor':
                    reasons = generate_investor_match_reasons(
                        user_features,
                        startup_features[startup_id],
                        {'embedding': scores.get(startup_id, 0)}
                    )
                else:
                    reasons = generate_match_reasons(
                        user_features,
                        startup_features[startup_id],
                        {'embedding': scores.get(startup_id, 0)}
                    )
                match_reasons[startup_id] = reasons
            
            logger.info(f"Generated {len(selected_ids)} recommendations")
            
            return {
                'item_ids': selected_ids,
                'scores': {sid: scores[sid] for sid in selected_ids},
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in recommend_startups_for_user: {e}", exc_info=True)
            return self._empty_result()
    
    def recommend_users_for_startup(self, startup_id, limit, filters, target_role):
        """
        Founder → Developer/Investor recommendations
        
        Args:
            startup_id: Startup ID (or founder user_id)
            limit: Max recommendations
            filters: Optional filters
            target_role: 'student' or 'investor'
            
        Returns:
            dict with user_ids, scores, match_reasons
        """
        try:
            # For now, return empty - this would require reverse matching
            # Implementation would be similar but matching users to startup
            logger.info(f"Founder→{target_role} matching not yet fully implemented")
            return self._empty_result()
            
        except Exception as e:
            logger.error(f"Error in recommend_users_for_startup: {e}", exc_info=True)
            return self._empty_result()
    
    def explain(self, user_id, item_id, use_case):
        """
        Generate explanation for a specific recommendation
        
        Args:
            user_id: User ID
            item_id: Item ID (startup or user)
            use_case: Type of recommendation
            
        Returns:
            list of match reasons
        """
        try:
            user_features = self.feature_extractor.extract_user_features(user_id)
            startup_features = self.feature_extractor.extract_startup_features([item_id])
            
            if item_id not in startup_features:
                return ["Unable to generate explanation"]
            
            scores = self._calculate_similarity_scores(user_features, startup_features)
            
            reasons = generate_match_reasons(
                user_features,
                startup_features[item_id],
                {'embedding': scores.get(item_id, 0)}
            )
            
            return reasons
            
        except Exception as e:
            logger.error(f"Error in explain: {e}")
            return ["Unable to generate explanation"]
    
    def _calculate_similarity_scores(self, user_features, startup_features):
        """
        Calculate combined similarity scores
        
        Args:
            user_features: User features dict
            startup_features: Dict of {startup_id: features}
            
        Returns:
            dict: {startup_id: combined_score}
        """
        try:
            all_scores = {}
            
            # Get user embedding
            user_embedding = user_features.get('embedding')
            
            # Get startup embeddings
            startup_embeddings = {
                sid: features.get('embedding')
                for sid, features in startup_features.items()
                if features.get('embedding') is not None
            }
            
            for startup_id, startup_data in startup_features.items():
                # Component 1: Embedding similarity (33%)
                embedding_score = 0.0
                if user_embedding is not None and startup_id in startup_embeddings:
                    emb_scores = cosine_similarity_embeddings(user_embedding, {startup_id: startup_embeddings[startup_id]})
                    embedding_score = emb_scores.get(startup_id, 0.0)
                
                # Component 2: Preference similarity (33%)
                preference_score = preference_similarity(
                    user_features.get('preferences', {}),
                    {
                        'category': startup_data.get('category'),
                        'field': startup_data.get('field'),
                        'tags': startup_data.get('tags', []),
                        'stages': startup_data.get('stages', []),
                        'position_requirements': startup_data.get('position_requirements', [])
                    }
                )
                
                # Component 3: Profile similarity (34%)
                profile_score = profile_similarity(
                    user_features.get('profile', {}),
                    {
                        'position_requirements': startup_data.get('position_requirements', []),
                        'phase': startup_data.get('phase'),
                        'stages': startup_data.get('stages', []),
                    }
                )
                
                # Combine scores
                combined_score = combine_scores(
                    {
                        'embedding': embedding_score,
                        'preference': preference_score,
                        'profile': profile_score
                    },
                    {
                        'embedding': self.EMBEDDING_WEIGHT,
                        'preference': self.PREFERENCE_WEIGHT,
                        'profile': self.PROFILE_WEIGHT
                    }
                )
                
                all_scores[startup_id] = combined_score
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}", exc_info=True)
            return {}
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'item_ids': [],
            'scores': {},
            'match_reasons': {}
        }

