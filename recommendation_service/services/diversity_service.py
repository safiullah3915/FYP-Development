"""
Diversity Service
Balance relevance, diversity, and freshness for optimal UX
"""
import numpy as np
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class DiversityService:
    """
    Ensures recommendations are diverse, fresh, and engaging
    Prevents filter bubble and recommendation fatigue
    """
    
    def __init__(self):
        self.diversity_lambda = 0.7  # 70% relevance, 30% diversity
        self.max_same_category = 3  # Max 3 startups from same category in top 10
    
    def apply_diversity_mmr(self, candidates, scores, embeddings, limit):
        """
        MMR (Maximal Marginal Relevance) algorithm
        Balances relevance with diversity to prevent monotony
        
        Args:
            candidates: List of candidate IDs
            scores: Dict of {id: score}
            embeddings: Dict of {id: embedding_vector}
            limit: Number of items to select
            
        Returns:
            list: Selected IDs in order
        """
        if not candidates or limit <= 0:
            return []
        
        try:
            selected = []
            remaining = list(candidates)
            
            while len(selected) < limit and remaining:
                if not selected:
                    # First item: highest relevance score
                    best = max(remaining, key=lambda x: scores.get(x, 0))
                else:
                    # Balance relevance and diversity
                    mmr_scores = []
                    for candidate in remaining:
                        relevance = scores.get(candidate, 0)
                        
                        # Calculate max similarity to already selected items
                        if candidate in embeddings and any(s in embeddings for s in selected):
                            max_sim = max([
                                self._cosine_similarity(
                                    embeddings[candidate],
                                    embeddings[selected_item]
                                )
                                for selected_item in selected
                                if selected_item in embeddings
                            ])
                        else:
                            max_sim = 0
                        
                        # MMR score: high relevance, low similarity to selected
                        mmr_score = (
                            self.diversity_lambda * relevance - 
                            (1 - self.diversity_lambda) * max_sim
                        )
                        mmr_scores.append(mmr_score)
                    
                    best_idx = np.argmax(mmr_scores)
                    best = remaining[best_idx]
                
                selected.append(best)
                remaining.remove(best)
            
            return selected
        except Exception as e:
            logger.error(f"Error in MMR diversity: {e}")
            # Fallback: return by score
            return sorted(candidates, key=lambda x: scores.get(x, 0), reverse=True)[:limit]
    
    def apply_category_diversity(self, recommendations, startup_data):
        """
        Ensure category diversity in recommendations
        Prevents all recommendations from same category
        
        Args:
            recommendations: List of startup IDs
            startup_data: Dict of {startup_id: features}
            
        Returns:
            list: Reordered recommendations with category diversity
        """
        if len(recommendations) <= 3:
            return recommendations
        
        try:
            diversified = []
            category_count = {}
            remaining = list(recommendations)
            
            # First pass: add items respecting category limit
            for rec in remaining[:]:
                category = startup_data.get(rec, {}).get('category', 'other')
                
                if category_count.get(category, 0) < self.max_same_category:
                    diversified.append(rec)
                    category_count[category] = category_count.get(category, 0) + 1
                    remaining.remove(rec)
            
            # Second pass: add remaining items
            diversified.extend(remaining)
            
            return diversified[:len(recommendations)]
        except Exception as e:
            logger.error(f"Error in category diversity: {e}")
            return recommendations
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return dot_product / norm_product if norm_product > 0 else 0
        except:
            return 0
    
    def apply_all_diversity_strategies(self, candidates, scores, embeddings, 
                                      startup_data, limit):
        """
        Master method: Apply all diversity and freshness strategies
        
        Args:
            candidates: List of candidate IDs
            scores: Dict of {id: score}
            embeddings: Dict of {id: embedding}
            startup_data: Dict of {id: features}
            limit: Number of recommendations
            
        Returns:
            list: Diverse, high-quality recommendations
        """
        try:
            # Step 1: Select diverse items using MMR
            selected = self.apply_diversity_mmr(candidates, scores, embeddings, limit)
            
            # Step 2: Ensure category diversity
            selected = self.apply_category_diversity(selected, startup_data)
            
            return selected
        except Exception as e:
            logger.error(f"Error applying diversity strategies: {e}")
            # Fallback: return top by score
            return sorted(candidates, key=lambda x: scores.get(x, 0), reverse=True)[:limit]

