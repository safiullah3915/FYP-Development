"""
Data loading utilities for efficient database queries
"""
from sqlalchemy.orm import joinedload
from database.models import User, Startup, UserProfile, UserOnboardingPreferences, StartupTag, Position
from utils.logger import get_logger

logger = get_logger(__name__)


def load_user_with_relations(db_session, user_id):
    """
    Load user + profile + preferences in one query
    
    Args:
        db_session: SQLAlchemy database session
        user_id: User ID (UUID) - can be with or without dashes
        
    Returns:
        User object with relations loaded or None
    """
    try:
        # Be robust to UUID representation (hyphenated vs 32-hex)
        # Django stores UUIDs WITH dashes, but some code normalizes them
        uid = str(user_id)
        candidates = [uid]
        try:
            import uuid as _uuid
            if '-' in uid:
                # If UUID has dashes, also try without dashes
                candidates.append(uid.replace('-', ''))
            elif len(uid) == 32:
                # If UUID has no dashes (32 hex chars), also try with dashes
                try:
                    candidates.append(str(_uuid.UUID(uid)))
                except Exception:
                    pass
        except Exception:
            pass
        
        user = db_session.query(User).options(
            joinedload(User.profile),
            joinedload(User.onboarding_preferences)
        ).filter(User.id.in_(candidates)).first()
        
        return user
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None


def load_startup_with_relations(db_session, startup_id):
    """
    Load startup + tags + positions in one query
    
    Args:
        db_session: SQLAlchemy database session
        startup_id: Startup ID (UUID)
        
    Returns:
        Startup object with relations loaded or None
    """
    try:
        # Be robust to UUID representation (hyphenated vs 32-hex)
        sid = str(startup_id)
        candidates = [sid]
        try:
            import uuid as _uuid
            if '-' in sid:
                candidates.append(sid.replace('-', ''))
            elif len(sid) == 32:
                try:
                    candidates.append(str(_uuid.UUID(sid)))
                except Exception:
                    pass
        except Exception:
            pass
        
        startup = db_session.query(Startup).options(
            joinedload(Startup.tags),
            joinedload(Startup.positions)
        ).filter(Startup.id.in_(candidates)).first()
        
        return startup
    except Exception as e:
        logger.error(f"Error loading startup {startup_id}: {e}")
        return None


def load_active_startups_batch(db_session, filters=None):
    """
    Optimized query for candidate startups
    
    Args:
        db_session: SQLAlchemy database session
        filters: Optional dict with filter criteria
            - type: 'marketplace' or 'collaboration'
            - category: list of categories
            - field: list of fields
            
    Returns:
        List of Startup objects
    """
    try:
        query = db_session.query(Startup).filter(
            Startup.status == 'active',
            Startup.profile_embedding.isnot(None)
        )
        
        if filters:
            if 'type' in filters and filters['type']:
                query = query.filter(Startup.type == filters['type'])
            if 'category' in filters and filters['category']:
                categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                query = query.filter(Startup.category.in_(categories))
            if 'field' in filters and filters['field']:
                fields = filters['field'] if isinstance(filters['field'], list) else [filters['field']]
                query = query.filter(Startup.field.in_(fields))
        
        return query.all()
    except Exception as e:
        logger.error(f"Error loading startups: {e}")
        return []


def load_users_by_role(db_session, role, limit=None):
    """
    Load users by role (for founder→developer, founder→investor matching)
    
    Args:
        db_session: SQLAlchemy database session
        role: User role ('student', 'investor', 'entrepreneur')
        limit: Optional limit on number of users
        
    Returns:
        List of User objects
    """
    try:
        query = db_session.query(User).filter(
            User.role == role,
            User.is_active == True,
            User.profile_embedding.isnot(None)
        )
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    except Exception as e:
        logger.error(f"Error loading users by role {role}: {e}")
        return []

