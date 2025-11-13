"""
Match reason generation for explainable recommendations
"""
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_match_reasons(user_features, startup_features, scores):
    """
    Generate human-readable match reasons
    
    Args:
        user_features: Dict with user data
            - embedding: numpy array
            - preferences: dict
            - profile: dict
            - role: string
        startup_features: Dict with startup data
            - embedding: numpy array
            - category: string
            - field: string
            - tags: list
            - stages: list
            - positions: list
        scores: Dict with component scores
            - embedding: float
            - preference: float
            - profile: float
            
    Returns:
        list: Top 3-5 match reasons
    """
    reasons = []
    
    try:
        # 1. Embedding/Semantic match
        if scores.get('embedding', 0) > 0.7:
            percentage = int(scores['embedding'] * 100)
            reasons.append(f"{percentage}% semantic match based on your profile")
        
        # 2. Category match
        user_categories = user_features.get('preferences', {}).get('selected_categories', [])
        startup_category = startup_features.get('category')
        if startup_category and startup_category in user_categories:
            reasons.append(f"Category match: You're interested in {startup_category}")
        
        # 3. Field match
        user_fields = user_features.get('preferences', {}).get('selected_fields', [])
        startup_field = startup_features.get('field')
        if startup_field and startup_field in user_fields:
            reasons.append(f"Field match: {startup_field} aligns with your interests")
        
        # 4. Skills match
        user_skills = user_features.get('profile', {}).get('skills', [])
        startup_positions = startup_features.get('positions', [])
        if user_skills and startup_positions:
            # Extract required skills from positions
            required_skills = set()
            for pos in startup_positions:
                if pos.get('requirements'):
                    # Simple extraction: look for common tech keywords
                    req_lower = pos['requirements'].lower()
                    for skill in user_skills:
                        if skill.lower() in req_lower:
                            required_skills.add(skill)
            
            if required_skills:
                skills_str = ', '.join(list(required_skills)[:3])
                reasons.append(f"Skills match: Your {skills_str} skills align with their needs")
        
        # 5. Stage alignment
        user_stages = user_features.get('preferences', {}).get('preferred_startup_stages', [])
        startup_stages = startup_features.get('stages', [])
        if user_stages and startup_stages:
            matching_stages = set(user_stages) & set(startup_stages)
            if matching_stages:
                stage_str = ', '.join(matching_stages)
                reasons.append(f"Stage alignment: {stage_str} matches your preferences")
        
        # 6. Tags match
        user_tags = user_features.get('preferences', {}).get('selected_tags', [])
        startup_tags = startup_features.get('tags', [])
        if user_tags and startup_tags:
            matching_tags = set(user_tags) & set(startup_tags)
            if matching_tags:
                tag_str = ', '.join(list(matching_tags)[:3])
                reasons.append(f"Technology match: {tag_str}")
        
        # 7. High overall match
        if scores.get('preference', 0) > 0.8:
            reasons.append("Strong preference alignment with your interests")
        
        # 8. Profile completeness (if startup has many details)
        if len(startup_features.get('tags', [])) > 3 and startup_features.get('description'):
            reasons.append("Well-documented opportunity with detailed information")
        
        # Return top 5 reasons
        return reasons[:5] if reasons else ["Recommended based on your profile"]
        
    except Exception as e:
        logger.error(f"Error generating match reasons: {e}")
        return ["Recommended for you"]


def generate_developer_match_reasons(developer_features, startup_features, scores):
    """
    Generate match reasons specifically for developer→startup recommendations
    
    Args:
        developer_features: Dict with developer data
        startup_features: Dict with startup data
        scores: Dict with component scores
        
    Returns:
        list: Match reasons tailored for developers
    """
    reasons = []
    
    try:
        # Skills alignment
        dev_skills = developer_features.get('profile', {}).get('skills', [])
        positions = startup_features.get('positions', [])
        
        if dev_skills and positions:
            for pos in positions[:2]:  # Check first 2 positions
                if pos.get('requirements'):
                    req_lower = pos['requirements'].lower()
                    matching = [s for s in dev_skills if s.lower() in req_lower]
                    if matching:
                        reasons.append(f"Your {', '.join(matching[:2])} skills match {pos.get('title', 'position')}")
        
        # Work arrangement
        startup_earn = startup_features.get('earn_through')
        if startup_earn:
            reasons.append(f"Compensation: {startup_earn}")
        
        # Team size
        team_size = startup_features.get('team_size')
        if team_size:
            reasons.append(f"Team size: {team_size} - good for collaboration")
        
        # Phase
        phase = startup_features.get('phase')
        if phase:
            reasons.append(f"Startup phase: {phase}")
        
        # Category interest
        category = startup_features.get('category')
        user_categories = developer_features.get('preferences', {}).get('selected_categories', [])
        if category and category in user_categories:
            reasons.append(f"Working in {category} - your area of interest")
        
        return reasons[:5] if reasons else generate_match_reasons(developer_features, startup_features, scores)
        
    except Exception as e:
        logger.error(f"Error generating developer match reasons: {e}")
        return generate_match_reasons(developer_features, startup_features, scores)


def generate_investor_match_reasons(investor_features, startup_features, scores):
    """
    Generate match reasons specifically for investor→startup recommendations
    
    Args:
        investor_features: Dict with investor data
        startup_features: Dict with startup data
        scores: Dict with component scores
        
    Returns:
        list: Match reasons tailored for investors
    """
    reasons = []
    
    try:
        # Category/sector match
        category = startup_features.get('category')
        investor_categories = investor_features.get('preferences', {}).get('selected_categories', [])
        if category and category in investor_categories:
            reasons.append(f"Sector match: {category} aligns with your investment focus")
        
        # Financial metrics
        revenue = startup_features.get('revenue')
        profit = startup_features.get('profit')
        if revenue and revenue != '$0':
            reasons.append(f"Revenue: {revenue}")
        if profit and profit != '$0':
            reasons.append(f"Profit: {profit}")
        
        # Stage match
        stages = startup_features.get('stages', [])
        investor_stages = investor_features.get('preferences', {}).get('preferred_startup_stages', [])
        if stages and investor_stages:
            matching = set(stages) & set(investor_stages)
            if matching:
                reasons.append(f"Investment stage: {', '.join(matching)}")
        
        # Asking price
        asking_price = startup_features.get('asking_price')
        if asking_price and asking_price != '$0':
            reasons.append(f"Asking price: {asking_price}")
        
        # Growth indicators
        if scores.get('embedding', 0) > 0.75:
            reasons.append("Strong profile match based on investment criteria")
        
        return reasons[:5] if reasons else generate_match_reasons(investor_features, startup_features, scores)
        
    except Exception as e:
        logger.error(f"Error generating investor match reasons: {e}")
        return generate_match_reasons(investor_features, startup_features, scores)

