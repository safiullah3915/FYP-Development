"""
Ensemble Recommender
Combines ALS and Two-Tower model predictions using weighted average
"""
from typing import Dict, List, Optional
from .base_recommender import BaseRecommender
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class EnsembleRecommender(BaseRecommender):
    """
    Ensemble recommender that combines multiple models
    
    Current implementation: Weighted average of ALS and Two-Tower
    - ALS weight: 0.6 (captures collaborative patterns)
    - Two-Tower weight: 0.4 (captures semantic patterns)
    """
    
    def __init__(self, als_recommender, two_tower_recommender, als_weight=0.6):
        """
        Initialize ensemble with component recommenders
        
        Args:
            als_recommender: ALS recommender instance
            two_tower_recommender: Two-Tower recommender instance
            als_weight: Weight for ALS predictions (default: 0.6)
        """
        self.als = als_recommender
        self.two_tower = two_tower_recommender
        self.als_weight = als_weight
        self.two_tower_weight = 1.0 - als_weight
        
        logger.info(f"Ensemble initialized with weights: ALS={self.als_weight:.2f}, Two-Tower={self.two_tower_weight:.2f}")
    
    def recommend(self, user_id, use_case, limit, filters=None):
        """
        Generate ensemble recommendations
        
        Args:
            user_id: User ID
            use_case: Use case type
            limit: Max recommendations
            filters: Optional filters
            
        Returns:
            dict with item_ids, scores, match_reasons
        """
        try:
            logger.info(f"Ensemble recommend for user {user_id}, use_case {use_case}, limit {limit}")
            
            # Get predictions from both models
            als_results = self._get_als_predictions(user_id, use_case, limit, filters)
            two_tower_results = self._get_two_tower_predictions(user_id, use_case, limit, filters)
            
            # Check if we got results from both
            if not als_results['item_ids'] and not two_tower_results['item_ids']:
                logger.warning("Both models returned empty results")
                return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
            
            # If only one model has results, use that
            if not als_results['item_ids']:
                logger.info("Using only Two-Tower (ALS had no results)")
                return two_tower_results
            if not two_tower_results['item_ids']:
                logger.info("Using only ALS (Two-Tower had no results)")
                return als_results
            
            # Normalize scores
            als_scores_norm = self.normalize_scores(als_results['scores'])
            two_tower_scores_norm = self.normalize_scores(two_tower_results['scores'])
            
            # Combine scores using weighted average
            combined_scores = self.weighted_average(
                als_scores_norm, 
                two_tower_scores_norm,
                self.als_weight
            )
            
            # Sort by combined score
            sorted_items = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            # Extract results
            item_ids = [item_id for item_id, _ in sorted_items]
            final_scores = {item_id: score for item_id, score in sorted_items}
            
            # Combine match reasons
            match_reasons = self._combine_match_reasons(
                item_ids,
                als_results['match_reasons'],
                two_tower_results['match_reasons'],
                als_scores_norm,
                two_tower_scores_norm
            )
            
            logger.info(f"Ensemble generated {len(item_ids)} recommendations")
            
            return {
                'item_ids': item_ids,
                'scores': final_scores,
                'match_reasons': match_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble recommend: {e}", exc_info=True)
            # Fallback to ALS or Two-Tower
            return self._fallback_single_model(user_id, use_case, limit, filters)
    
    def _get_als_predictions(self, user_id, use_case, limit, filters):
        """Get predictions from ALS model"""
        try:
            if self.als:
                # Request more candidates for ensemble diversity
                results = self.als.recommend(user_id, use_case, limit * 2, filters)
                return results
            else:
                logger.warning("ALS model not available")
                return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
        except Exception as e:
            logger.error(f"Error getting ALS predictions: {e}")
            return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
    
    def _get_two_tower_predictions(self, user_id, use_case, limit, filters):
        """Get predictions from Two-Tower model"""
        try:
            if self.two_tower:
                # Request more candidates for ensemble diversity
                results = self.two_tower.recommend(user_id, use_case, limit * 2, filters)
                return results
            else:
                logger.warning("Two-Tower model not available")
                return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
        except Exception as e:
            logger.error(f"Error getting Two-Tower predictions: {e}")
            return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] using min-max scaling
        
        Args:
            scores: Dict mapping item_id to score
        
        Returns:
            Dict with normalized scores
        """
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        
        if max_score == min_score:
            # All scores are the same, return uniform distribution
            return {k: 1.0 for k in scores.keys()}
        
        normalized = {}
        for item_id, score in scores.items():
            normalized[item_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def weighted_average(self, als_scores: Dict[str, float], two_tower_scores: Dict[str, float], alpha: float) -> Dict[str, float]:
        """
        Combine scores using weighted average
        
        Args:
            als_scores: Normalized ALS scores
            two_tower_scores: Normalized Two-Tower scores
            alpha: Weight for ALS (1-alpha for Two-Tower)
        
        Returns:
            Dict with combined scores
        """
        combined = {}
        
        # Get all unique items
        all_items = set(als_scores.keys()) | set(two_tower_scores.keys())
        
        for item_id in all_items:
            als_score = als_scores.get(item_id, 0.0)
            tt_score = two_tower_scores.get(item_id, 0.0)
            
            # Weighted average
            combined[item_id] = alpha * als_score + (1 - alpha) * tt_score
        
        return combined
    
    def rank_fusion(self, als_rankings: List[str], two_tower_rankings: List[str], k=60) -> List[str]:
        """
        Reciprocal Rank Fusion (RRF) for combining rankings
        
        Args:
            als_rankings: Ranked list from ALS
            two_tower_rankings: Ranked list from Two-Tower
            k: Constant for RRF (default: 60)
        
        Returns:
            Combined ranking
        """
        scores = {}
        
        # Calculate RRF scores for ALS
        for rank, item_id in enumerate(als_rankings, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
        
        # Add RRF scores for Two-Tower
        for rank, item_id in enumerate(two_tower_rankings, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
        
        # Sort by combined RRF score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items]
    
    def _combine_match_reasons(self, item_ids, als_reasons, two_tower_reasons, 
                                als_scores, two_tower_scores):
        """Combine match reasons from both models"""
        combined_reasons = {}
        
        for item_id in item_ids:
            reasons = []
            
            # Add ALS contribution
            if item_id in als_scores and als_scores[item_id] > 0:
                reasons.append(f"Collaborative filtering (ALS): {als_scores[item_id]:.3f}")
                if item_id in als_reasons:
                    reasons.extend(als_reasons[item_id][:2])  # Take top 2 reasons
            
            # Add Two-Tower contribution
            if item_id in two_tower_scores and two_tower_scores[item_id] > 0:
                reasons.append(f"Deep learning (Two-Tower): {two_tower_scores[item_id]:.3f}")
                if item_id in two_tower_reasons:
                    reasons.extend(two_tower_reasons[item_id][:2])  # Take top 2 reasons
            
            if not reasons:
                reasons.append("Ensemble recommendation")
            
            combined_reasons[item_id] = reasons
        
        return combined_reasons
    
    def _fallback_single_model(self, user_id, use_case, limit, filters):
        """Fallback to single model if ensemble fails"""
        logger.info("Ensemble failed, falling back to single model")
        
        # Try ALS first
        if self.als:
            try:
                results = self.als.recommend(user_id, use_case, limit, filters)
                if results['item_ids']:
                    logger.info("Using ALS as fallback")
                    return results
            except:
                pass
        
        # Try Two-Tower
        if self.two_tower:
            try:
                results = self.two_tower.recommend(user_id, use_case, limit, filters)
                if results['item_ids']:
                    logger.info("Using Two-Tower as fallback")
                    return results
            except:
                pass
        
        # Both failed
        logger.error("All models failed")
        return {'item_ids': [], 'scores': {}, 'match_reasons': {}}
    
    def explain(self, user_id, item_id, use_case):
        """Generate explanation combining both models"""
        reasons = ["Ensemble recommendation combining:"]
        
        # Get ALS explanation
        if self.als:
            try:
                als_reasons = self.als.explain(user_id, item_id, use_case)
                reasons.append(f"  - Collaborative filtering: {als_reasons[0] if als_reasons else 'N/A'}")
            except:
                pass
        
        # Get Two-Tower explanation
        if self.two_tower:
            try:
                tt_reasons = self.two_tower.explain(user_id, item_id, use_case)
                reasons.append(f"  - Deep learning: {tt_reasons[0] if tt_reasons else 'N/A'}")
            except:
                pass
        
        return reasons
