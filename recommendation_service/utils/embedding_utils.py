"""
Embedding utilities for loading and validating embeddings
"""
import json
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def load_embedding_from_json(json_string):
    """
    Parse JSON string to numpy array
    
    Args:
        json_string: JSON string representation of embedding vector
        
    Returns:
        numpy array or None if parsing fails
    """
    if not json_string:
        return None
    try:
        embedding_list = json.loads(json_string)
        return np.array(embedding_list, dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse embedding: {e}")
        return None


def validate_embedding(embedding, expected_dim=384):
    """
    Check if embedding is valid
    
    Args:
        embedding: numpy array
        expected_dim: expected dimension (default 384 for all-MiniLM-L6-v2)
        
    Returns:
        bool: True if valid, False otherwise
    """
    if embedding is None:
        return False
    if len(embedding) != expected_dim:
        logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected {expected_dim}")
        return False
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        logger.warning("Embedding contains NaN or Inf values")
        return False
    return True


def batch_load_embeddings(db_session, ids, model_class, id_field='id'):
    """
    Efficiently load embeddings for multiple entities
    
    Args:
        db_session: SQLAlchemy database session
        ids: List of entity IDs
        model_class: SQLAlchemy model class (User or Startup)
        id_field: Name of ID field (default 'id')
        
    Returns:
        dict: {entity_id: embedding_vector}
    """
    entities = db_session.query(model_class).filter(
        getattr(model_class, id_field).in_(ids)
    ).all()
    
    embeddings = {}
    for entity in entities:
        emb = load_embedding_from_json(entity.profile_embedding)
        if validate_embedding(emb):
            embeddings[str(entity.id)] = emb
        else:
            logger.debug(f"Skipping invalid embedding for {model_class.__name__} {entity.id}")
    
    return embeddings

