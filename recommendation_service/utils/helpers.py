"""
Helper utility functions for Flask Recommendation Service
"""
import json
from typing import Optional, List, Dict, Any


def parse_json_field(value: Optional[str]) -> Any:
    """
    Parse JSON string field to Python object
    
    Args:
        value: JSON string or None
    
    Returns:
        Parsed Python object (dict, list, etc.) or None
    """
    if value is None or value == '':
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_embedding(embedding_str: Optional[str]) -> Optional[List[float]]:
    """
    Parse embedding JSON string to list of floats
    
    Args:
        embedding_str: JSON string representation of embedding vector
    
    Returns:
        List of floats or None
    """
    parsed = parse_json_field(embedding_str)
    if parsed is None:
        return None
    
    try:
        # Ensure it's a list of numbers
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return None
    except (ValueError, TypeError):
        return None


def safe_get(dictionary: Dict, *keys, default=None) -> Any:
    """
    Safely get nested dictionary values
    
    Args:
        dictionary: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        else:
            return default
    return result if result is not None else default


