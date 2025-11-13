"""
Similarity calculation functions for content-based filtering
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import get_logger

logger = get_logger(__name__)


def cosine_similarity_embeddings(user_emb, startup_embs):
    """
    Vectorized cosine similarity using sklearn
    
    Args:
        user_emb: User embedding vector (numpy array)
        startup_embs: Dict of {startup_id: embedding_vector}
        
    Returns:
        dict: {startup_id: similarity_score} normalized to [0, 1]
    """
    if user_emb is None or len(startup_embs) == 0:
        return {}
    
    try:
        # Prepare startup embeddings matrix
        startup_ids = list(startup_embs.keys())
        startup_matrix = np.array([startup_embs[sid] for sid in startup_ids])
        
        # Calculate cosine similarity
        similarities = cosine_similarity([user_emb], startup_matrix)[0]
        
        # Normalize to [0, 1] from [-1, 1]
        similarities_normalized = (similarities + 1) / 2
        
        # Create result dict
        result = {startup_ids[i]: float(similarities_normalized[i]) 
                  for i in range(len(startup_ids))}
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return {}


def jaccard_similarity(set1, set2):
    """
    Jaccard index for categorical overlap
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        float: Jaccard similarity [0, 1]
    """
    if not set1 or not set2:
        return 0.0
    
    try:
        set1 = set(set1) if not isinstance(set1, set) else set1
        set2 = set(set2) if not isinstance(set2, set) else set2
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating Jaccard similarity: {e}")
        return 0.0


def preference_similarity(user_prefs, startup_attrs):
    """
    Multi-dimensional preference matching
    
    Args:
        user_prefs: Dict with user preferences
            - selected_categories: list
            - selected_fields: list
            - selected_tags: list
            - preferred_startup_stages: list
            - preferred_skills: list
        startup_attrs: Dict with startup attributes
            - category: string
            - field: string
            - tags: list
            - stages: list
            - position_requirements: list (skills needed)
            
    Returns:
        float: Preference similarity score [0, 1]
    """
    try:
        scores = []
        weights = []
        
        # Category match
        if user_prefs.get('selected_categories'):
            category_match = 1.0 if startup_attrs.get('category') in user_prefs['selected_categories'] else 0.0
            scores.append(category_match)
            weights.append(0.3)
        
        # Field match
        if user_prefs.get('selected_fields'):
            field_match = 1.0 if startup_attrs.get('field') in user_prefs['selected_fields'] else 0.0
            scores.append(field_match)
            weights.append(0.25)
        
        # Tags match (Jaccard)
        if user_prefs.get('selected_tags') and startup_attrs.get('tags'):
            tag_similarity = jaccard_similarity(user_prefs['selected_tags'], startup_attrs['tags'])
            scores.append(tag_similarity)
            weights.append(0.25)
        
        # Stages match (Jaccard)
        if user_prefs.get('preferred_startup_stages') and startup_attrs.get('stages'):
            stage_similarity = jaccard_similarity(user_prefs['preferred_startup_stages'], startup_attrs['stages'])
            scores.append(stage_similarity)
            weights.append(0.15)
        
        # Skills match (Jaccard)
        if user_prefs.get('preferred_skills') and startup_attrs.get('position_requirements'):
            skill_similarity = jaccard_similarity(user_prefs['preferred_skills'], startup_attrs['position_requirements'])
            scores.append(skill_similarity)
            weights.append(0.05)
        
        # Weighted average
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
        
    except Exception as e:
        logger.error(f"Error calculating preference similarity: {e}")
        return 0.0


def profile_similarity(user_profile, startup_data):
    """
    Skills, experience, location matching
    
    Args:
        user_profile: Dict with user profile data
            - skills: list
            - experience: list/dict
            - location: string
        startup_data: Dict with startup data
            - position_requirements: list (skills needed)
            - phase: string
            - stages: list
            - location: string (if available)
            
    Returns:
        float: Profile similarity score [0, 1]
    """
    try:
        scores = []
        weights = []
        
        # Skills match (most important)
        if user_profile.get('skills') and startup_data.get('position_requirements'):
            skill_similarity = jaccard_similarity(user_profile['skills'], startup_data['position_requirements'])
            scores.append(skill_similarity)
            weights.append(0.6)
        
        # Experience/phase match
        if user_profile.get('experience') and startup_data.get('phase'):
            # Simple heuristic: check if phase mentioned in experience
            experience_str = str(user_profile['experience']).lower()
            phase_str = str(startup_data['phase']).lower()
            phase_match = 1.0 if phase_str in experience_str else 0.3
            scores.append(phase_match)
            weights.append(0.2)
        
        # Location match
        if user_profile.get('location') and startup_data.get('location'):
            location_match = 1.0 if user_profile['location'].lower() == startup_data['location'].lower() else 0.0
            scores.append(location_match)
            weights.append(0.2)
        
        # Weighted average
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
        
    except Exception as e:
        logger.error(f"Error calculating profile similarity: {e}")
        return 0.0


def combine_scores(scores_dict, weights):
    """
    Weighted combination with normalization
    
    Args:
        scores_dict: Dict of {component_name: score}
        weights: Dict of {component_name: weight}
        
    Returns:
        float: Combined score [0, 1]
    """
    try:
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        combined = sum(scores_dict.get(k, 0.0) * weights.get(k, 0.0) 
                      for k in weights.keys())
        
        return combined / total_weight
        
    except Exception as e:
        logger.error(f"Error combining scores: {e}")
        return 0.0

