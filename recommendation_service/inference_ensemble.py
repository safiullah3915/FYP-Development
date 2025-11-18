"""
Ensemble Inference Module
Combines ALS and Two-Tower models for improved recommendations
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from database.connection import SessionLocal
from database.models import Startup
from utils.logger import get_logger
from inference_als import ALSInference
from inference_two_tower import TwoTowerInference

logger = get_logger(__name__)


class EnsembleInference:
    """
    Ensemble inference combining ALS and Two-Tower models
    Uses weighted average: 0.6 ALS + 0.4 Two-Tower
    """
    
    def __init__(self, als_model_path: str, two_tower_model_path: str, 
                 als_weight: float = 0.6):
        """
        Initialize ensemble inference
        
        Args:
            als_model_path: Path to ALS config/prefix (e.g., models/als_v1_config.json)
            two_tower_model_path: Path to Two-Tower model (e.g., models/two_tower_v1_best.pth)
            als_weight: Weight for ALS (default: 0.6)
        """
        self.als_weight = als_weight
        self.two_tower_weight = 1.0 - als_weight
        
        # Initialize models
        self.als_model = None
        self.two_tower_model = None
        
        self._load_models(als_model_path, two_tower_model_path)
    
    def _load_models(self, als_path: str, two_tower_path: str):
        """Load both ALS and Two-Tower models"""
        logger.info("Loading ensemble models...")
        
        # Load ALS
        try:
            if os.path.exists(als_path):
                self.als_model = ALSInference(als_path)
                logger.info("  ALS model loaded")
            else:
                logger.warning(f"  ALS model not found: {als_path}")
        except Exception as e:
            logger.error(f"  Failed to load ALS model: {e}")
        
        # Load Two-Tower
        try:
            if os.path.exists(two_tower_path):
                self.two_tower_model = TwoTowerInference(two_tower_path)
                logger.info("  Two-Tower model loaded")
            else:
                logger.warning(f"  Two-Tower model not found: {two_tower_path}")
        except Exception as e:
            logger.error(f"  Failed to load Two-Tower model: {e}")
        
        if not self.als_model and not self.two_tower_model:
            raise RuntimeError("Failed to load any models for ensemble")
        
        logger.info(f"Ensemble initialized with weights: ALS={self.als_weight:.2f}, Two-Tower={self.two_tower_weight:.2f}")
    
    def recommend(self, user_id: str, limit: int = 10, filters: Optional[Dict] = None, fetch_multiplier: int = 1) -> Dict:
        """
        Generate ensemble recommendations
        
        Args:
            user_id: User ID (UUID string)
            limit: Number of recommendations
            filters: Optional filters
            fetch_multiplier: Multiplier for fetching more candidates (for reranking)
        
        Returns:
            dict with startups, total, scores, method_used
        """
        try:
            actual_limit = limit * fetch_multiplier
            logger.info(f"Ensemble inference for user {user_id}, limit {limit} (fetching {actual_limit} candidates)")
            
            # Get predictions from both models with fetch_multiplier
            als_results = self._get_als_results(user_id, actual_limit, filters)
            two_tower_results = self._get_two_tower_results(user_id, actual_limit, filters)
            
            # If both models failed, return empty
            if not als_results['startups'] and not two_tower_results['startups']:
                logger.warning("Both models returned empty results")
                return {
                    'startups': [],
                    'total': 0,
                    'scores': {},
                    'method_used': 'ensemble_failed'
                }
            
            # If only one model has results, use that one
            if not als_results['startups']:
                logger.info("Using only Two-Tower (ALS unavailable)")
                two_tower_results['method_used'] = 'ensemble_two_tower_only'
                return two_tower_results
            
            if not two_tower_results['startups']:
                logger.info("Using only ALS (Two-Tower unavailable)")
                als_results['method_used'] = 'ensemble_als_only'
                return als_results
            
            # Both models have results - combine them
            combined_results = self._combine_results(
                als_results, two_tower_results, actual_limit
            )
            
            logger.info(f"Ensemble generated {combined_results['total']} recommendations")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in ensemble inference: {e}", exc_info=True)
            return {
                'startups': [],
                'total': 0,
                'scores': {},
                'method_used': 'ensemble_error'
            }
    
    def _get_als_results(self, user_id: str, limit: int, filters: Optional[Dict]) -> Dict:
        """Get results from ALS model"""
        if not self.als_model:
            return {'startups': [], 'total': 0, 'scores': {}}
        
        try:
            # Note: limit is already multiplied by caller if needed
            results = self.als_model.recommend(user_id, limit, filters)
            return results
        except Exception as e:
            logger.error(f"Error getting ALS results: {e}")
            return {'startups': [], 'total': 0, 'scores': {}}
    
    def _get_two_tower_results(self, user_id: str, limit: int, filters: Optional[Dict]) -> Dict:
        """Get results from Two-Tower model"""
        if not self.two_tower_model:
            return {'startups': [], 'total': 0, 'scores': {}}
        
        try:
            # Note: limit is already multiplied by caller if needed
            results = self.two_tower_model.recommend(user_id, limit, filters)
            
            # Two-Tower returns {'item_ids': [...], 'scores': {...}, 'match_reasons': {...}}
            # Convert to same format as ALS: {'startups': [...], 'scores': {...}}
            if 'item_ids' in results:
                # Need to fetch startup details from database
                from database.connection import SessionLocal
                from database.models import Startup
                
                db = SessionLocal()
                try:
                    startup_ids = results['item_ids']
                    # Startup IDs from two-tower are already normalized (from database)
                    # Query database directly with these IDs
                    startups = db.query(Startup).filter(
                        Startup.id.in_(startup_ids)
                    ).all()
                    
                    # Create mapping: normalized_id -> startup object
                    startups_dict = {str(s.id): s for s in startups}
                    match_reasons = results.get('match_reasons', {})
                    
                    recommendations = []
                    for startup_id in startup_ids:
                        # IDs are already normalized, use directly
                        if startup_id in startups_dict:
                            startup = startups_dict[startup_id]
                            recommendations.append({
                                'id': startup_id,
                                'title': startup.title,
                                'description': startup.description,
                                'type': startup.type,
                                'category': startup.category,
                                'field': startup.field,
                                'score': results['scores'].get(startup_id, 0.0),
                                'match_reasons': match_reasons.get(startup_id, ["Two-Tower model prediction"])
                            })
                    
                    return {
                        'startups': recommendations,
                        'total': len(recommendations),
                        'scores': results.get('scores', {}),
                        'method_used': 'two_tower'
                    }
                finally:
                    db.close()
            else:
                # Already in correct format
                return results
        except Exception as e:
            logger.error(f"Error getting Two-Tower results: {e}", exc_info=True)
            return {'startups': [], 'total': 0, 'scores': {}}
    
    def _combine_results(self, als_results: Dict, two_tower_results: Dict, limit: int) -> Dict:
        """Combine results from both models using weighted average"""
        # Normalize scores
        als_scores_norm = self._normalize_scores(als_results['scores'])
        two_tower_scores_norm = self._normalize_scores(two_tower_results['scores'])
        
        # Combine scores
        combined_scores = {}
        all_startup_ids = set(als_scores_norm.keys()) | set(two_tower_scores_norm.keys())
        
        for startup_id in all_startup_ids:
            als_score = als_scores_norm.get(startup_id, 0.0)
            tt_score = two_tower_scores_norm.get(startup_id, 0.0)
            combined_scores[startup_id] = (
                self.als_weight * als_score + 
                self.two_tower_weight * tt_score
            )
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        top_ids = [startup_id for startup_id, _ in sorted_ids]
        
        # Query database for startup details
        db = SessionLocal()
        try:
            startups = db.query(Startup).filter(
                Startup.id.in_(top_ids)
            ).all()
            
            startups_dict = {str(s.id): s for s in startups}
            
            # Build response in score order
            recommendations = []
            for startup_id, combined_score in sorted_ids:
                if startup_id in startups_dict:
                    startup = startups_dict[startup_id]
                    
                    # Combine match reasons
                    match_reasons = []
                    if startup_id in als_scores_norm:
                        match_reasons.append(f"Collaborative filtering (ALS): {als_scores_norm[startup_id]:.3f}")
                    if startup_id in two_tower_scores_norm:
                        match_reasons.append(f"Deep learning (Two-Tower): {two_tower_scores_norm[startup_id]:.3f}")
                    match_reasons.append(f"Ensemble score: {combined_score:.3f}")
                    
                    recommendations.append({
                        'id': startup_id,
                        'title': startup.title,
                        'description': startup.description,
                        'type': startup.type,
                        'category': startup.category,
                        'field': startup.field,
                        'score': combined_score,
                        'match_reasons': match_reasons
                    })
            
            return {
                'startups': recommendations,
                'total': len(recommendations),
                'scores': {r['id']: r['score'] for r in recommendations},
                'method_used': 'ensemble'
            }
            
        finally:
            db.close()
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] using min-max scaling"""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        
        if max_score == min_score:
            return {k: 1.0 for k in scores.keys()}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }


