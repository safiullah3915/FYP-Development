"""
Feature Extraction for Ranker
Extracts ranking signals from candidate items:
- Recency: How new/fresh the item is
- Popularity: Views, interactions
- Diversity: Avoid clustering similar items
"""
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
import math


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize score to [0, 1] range
    
    Args:
        score: Raw score
        min_val: Minimum expected value
        max_val: Maximum expected value
    
    Returns:
        Normalized score in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def calculate_recency_score(created_at: Optional[str], updated_at: Optional[str], decay_days: float = 30.0) -> float:
    """
    Calculate recency score using exponential decay
    Newer items get higher scores
    
    Args:
        created_at: ISO timestamp of creation
        updated_at: ISO timestamp of last update
        decay_days: Half-life in days (default: 30 days)
    
    Returns:
        Recency score in [0, 1], where 1 = very recent
    """
    try:
        # Use most recent timestamp (updated_at or created_at)
        if updated_at:
            timestamp_str = updated_at
        elif created_at:
            timestamp_str = created_at
        else:
            # No timestamp: assume old
            return 0.3
        
        # Parse timestamp
        if isinstance(timestamp_str, str):
            # Handle various ISO formats
            if 'T' in timestamp_str:
                if '+' in timestamp_str or timestamp_str.endswith('Z'):
                    # Has timezone
                    item_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    # No timezone, assume UTC
                    item_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            else:
                # Date only format
                item_time = datetime.strptime(timestamp_str[:10], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            # Already datetime object
            item_time = timestamp_str
            if item_time.tzinfo is None:
                item_time = item_time.replace(tzinfo=timezone.utc)
        
        # Calculate age in days
        now = datetime.now(timezone.utc)
        age_days = (now - item_time).total_seconds() / 86400.0  # seconds to days
        
        # Exponential decay: score = exp(-age / decay_constant)
        # After decay_days, score is ~0.37 (exp(-1))
        score = math.exp(-age_days / decay_days)
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        # Parsing error: return neutral score
        return 0.5


def calculate_popularity_score(views: int, interaction_count: int, max_views: int = 10000) -> float:
    """
    Calculate popularity score from views and interactions
    Uses log scaling to handle wide range of values
    
    Args:
        views: Number of views
        interaction_count: Number of interactions (clicks, likes, etc.)
        max_views: Expected maximum views for normalization
    
    Returns:
        Popularity score in [0, 1]
    """
    try:
        # Weight interactions higher than views
        # Weighted sum: views * 1.0 + interactions * 3.0
        weighted_popularity = views + (interaction_count * 3.0)
        
        # Log scaling to compress range
        # log(1 + x) ensures log(1) = 0 for x=0
        log_popularity = math.log1p(weighted_popularity)
        
        # Normalize using expected max
        max_log = math.log1p(max_views)
        
        if max_log == 0:
            return 0.0
        
        score = log_popularity / max_log
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        return 0.0


def calculate_diversity_penalty(candidate: Dict, already_ranked: List[Dict], threshold: float = 0.8) -> float:
    """
    Calculate diversity penalty to avoid clustering similar items
    Lower penalty = more diverse
    
    Similarity is based on:
    - Category/field overlap
    - Type match
    
    Args:
        candidate: Current candidate item
        already_ranked: List of items already ranked
        threshold: Similarity threshold for penalty
    
    Returns:
        Diversity score in [0, 1], where 1 = very diverse, 0 = not diverse
    """
    if not already_ranked:
        # First item: maximum diversity
        return 1.0
    
    try:
        # Extract candidate features
        candidate_category = candidate.get('category', '').lower()
        candidate_field = candidate.get('field', '').lower()
        candidate_type = candidate.get('type', '').lower()
        
        # Calculate similarity to already ranked items
        similarities = []
        
        for ranked_item in already_ranked:
            similarity = 0.0
            matches = 0
            total_features = 0
            
            # Category match
            if candidate_category and ranked_item.get('category'):
                total_features += 1
                if candidate_category == ranked_item.get('category', '').lower():
                    matches += 1
            
            # Field match
            if candidate_field and ranked_item.get('field'):
                total_features += 1
                if candidate_field == ranked_item.get('field', '').lower():
                    matches += 1
            
            # Type match
            if candidate_type and ranked_item.get('type'):
                total_features += 1
                if candidate_type == ranked_item.get('type', '').lower():
                    matches += 1
            
            # Calculate similarity ratio
            if total_features > 0:
                similarity = matches / total_features
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Max similarity to any already ranked item
        max_similarity = max(similarities)
        
        # Convert similarity to diversity score
        # High similarity → low diversity score
        # Apply threshold: if similarity > threshold, penalize heavily
        if max_similarity >= threshold:
            # Strong penalty for very similar items
            diversity_score = 1.0 - max_similarity
        else:
            # Mild penalty for somewhat similar items
            diversity_score = 1.0 - (max_similarity * 0.5)
        
        return max(0.0, min(1.0, diversity_score))
        
    except Exception as e:
        # Error: assume neutral diversity
        return 0.7


def extract_all_features(candidate: Dict, already_ranked: List[Dict]) -> Dict[str, float]:
    """
    Extract all ranking features for a candidate
    
    Args:
        candidate: Candidate item dict
        already_ranked: Previously ranked items
    
    Returns:
        Dict with all features
    """
    return {
        'model_score': normalize_score(candidate.get('score', 0.0)),
        'recency': calculate_recency_score(
            candidate.get('created_at'),
            candidate.get('updated_at')
        ),
        'popularity': calculate_popularity_score(
            candidate.get('views', 0),
            candidate.get('interaction_count', 0)
        ),
        'diversity': calculate_diversity_penalty(
            candidate,
            already_ranked
        )
    }

