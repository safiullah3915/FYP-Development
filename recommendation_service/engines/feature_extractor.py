"""
Feature extraction for users and startups
Efficiently loads and prepares data for recommendation algorithms
"""
import json
from utils.data_loader import load_user_with_relations, load_startup_with_relations, load_active_startups_batch
from utils.embedding_utils import load_embedding_from_json
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extracts and prepares features for recommendation"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def extract_user_features(self, user_id):
        """
        Extract comprehensive user features
        
        Args:
            user_id: User ID (UUID string)
            
        Returns:
            dict with:
                - embedding: numpy array or None
                - preferences: dict from UserOnboardingPreferences
                - profile: dict from UserProfile
                - role: string
                - user_id: string
        """
        try:
            user = load_user_with_relations(self.db, user_id)
            
            if not user:
                logger.warning(f"User {user_id} not found")
                return self._empty_user_features(user_id)
            
            # Extract embedding
            embedding = load_embedding_from_json(user.profile_embedding)
            
            # Extract preferences
            preferences = {}
            try:
                if hasattr(user, 'onboarding_preferences') and user.onboarding_preferences:
                    prefs = user.onboarding_preferences
                    preferences = {
                        'selected_categories': self._parse_json_field(prefs.selected_categories),
                        'selected_fields': self._parse_json_field(prefs.selected_fields),
                        'selected_tags': self._parse_json_field(prefs.selected_tags),
                        'preferred_startup_stages': self._parse_json_field(prefs.preferred_startup_stages),
                        'preferred_engagement_types': self._parse_json_field(prefs.preferred_engagement_types),
                        'preferred_skills': self._parse_json_field(prefs.preferred_skills),
                    }
            except Exception as e:
                logger.warning(f"Error extracting preferences for user {user_id}: {e}")
            
            # Extract profile
            profile = {}
            try:
                if hasattr(user, 'profile') and user.profile:
                    prof = user.profile
                    profile = {
                        'bio': prof.bio or '',
                        'skills': self._parse_json_field(prof.skills),
                        'experience': self._parse_json_field(prof.experience),
                        'location': prof.location or '',
                    }
            except Exception as e:
                logger.warning(f"Error extracting profile for user {user_id}: {e}")
            
            return {
                'embedding': embedding,
                'preferences': preferences,
                'profile': profile,
                'role': user.role,
                'user_id': str(user.id),
            }
            
        except Exception as e:
            logger.error(f"Error extracting user features for {user_id}: {e}")
            return self._empty_user_features(user_id)
    
    def extract_startup_features(self, startup_ids):
        """
        Batch load startup features
        
        Args:
            startup_ids: List of startup IDs (UUID strings)
            
        Returns:
            dict: {startup_id: features_dict}
        """
        if not startup_ids:
            return {}
        
        features = {}
        
        for startup_id in startup_ids:
            try:
                startup = load_startup_with_relations(self.db, startup_id)
                
                if not startup:
                    continue
                
                # Extract embedding
                embedding = load_embedding_from_json(startup.profile_embedding)
                
                # Extract tags
                tags = []
                try:
                    if hasattr(startup, 'tags'):
                        tags = [tag.tag for tag in startup.tags.all()]
                except Exception as e:
                    logger.warning(f"Error extracting tags for startup {startup_id}: {e}")
                
                # Extract positions
                positions = []
                position_requirements = []
                try:
                    if hasattr(startup, 'positions'):
                        for pos in startup.positions.all():
                            if pos.is_active:
                                positions.append({
                                    'id': str(pos.id),
                                    'title': pos.title,
                                    'description': pos.description or '',
                                    'requirements': pos.requirements or '',
                                })
                                # Extract skills from requirements
                                if pos.requirements:
                                    position_requirements.append(pos.requirements)
                except Exception as e:
                    logger.warning(f"Error extracting positions for startup {startup_id}: {e}")
                
                # Extract stages
                stages = self._parse_json_field(startup.stages)
                
                features[str(startup_id)] = {
                    'embedding': embedding,
                    'category': startup.category,
                    'field': startup.field,
                    'tags': tags,
                    'stages': stages,
                    'positions': positions,
                    'position_requirements': position_requirements,
                    'type': startup.type,
                    'phase': startup.phase or '',
                    'team_size': startup.team_size or '',
                    'earn_through': startup.earn_through or '',
                    'revenue': startup.revenue or '',
                    'profit': startup.profit or '',
                    'asking_price': startup.asking_price or '',
                    'description': startup.description or '',
                    'title': startup.title,
                    'created_at': startup.created_at,
                    'startup_id': str(startup.id),
                }
                
            except Exception as e:
                logger.error(f"Error extracting features for startup {startup_id}: {e}")
                continue
        
        return features
    
    def extract_all_active_startups(self, filters=None):
        """
        Get all active startup candidates with filters
        
        Args:
            filters: Optional dict with filter criteria
            
        Returns:
            dict: {startup_id: features_dict}
        """
        try:
            startups = load_active_startups_batch(self.db, filters)
            startup_ids = [str(s.id) for s in startups]
            
            return self.extract_startup_features(startup_ids)
            
        except Exception as e:
            logger.error(f"Error extracting active startups: {e}")
            return {}
    
    def _parse_json_field(self, field):
        """Parse JSON field that might be string or list"""
        if field is None:
            return []
        if isinstance(field, list):
            return field
        if isinstance(field, str):
            try:
                parsed = json.loads(field)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    def _empty_user_features(self, user_id):
        """Return empty user features structure"""
        return {
            'embedding': None,
            'preferences': {},
            'profile': {},
            'role': 'unknown',
            'user_id': str(user_id),
        }

