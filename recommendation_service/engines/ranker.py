"""
Neural Ranker Module
Reranks recommendation candidates using learned signals:
- Model score (from base recommender)
- Recency (how new the startup is)
- Popularity (views, interactions)
- Diversity (avoid clustering similar items)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class RankerMLP(nn.Module):
    """
    Simple 2-layer MLP for ranking
    Input: [model_score, recency_score, popularity_score, diversity_penalty, original_score]
    Output: ranking score
    """
    def __init__(self, input_dim=5, hidden_dim1=32, hidden_dim2=16):
        super(RankerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


class NeuralRanker:
    """
    Neural ranker that reorders recommendation candidates
    Uses learned weights to combine multiple ranking signals
    """
    
    def __init__(self, model_path: Optional[str] = None, use_rule_based: bool = False):
        """
        Initialize ranker
        
        Args:
            model_path: Path to trained model weights (if None, uses rule-based)
            use_rule_based: Force rule-based ranking even if model exists
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_rule_based = use_rule_based
        
        # Rule-based weights (used as fallback or default)
        self.rule_weights = {
            'model_score': 0.5,
            'recency': 0.2,
            'popularity': 0.2,
            'diversity': 0.1
        }
        
        if model_path and not use_rule_based:
            self._load_model(model_path)
        else:
            logger.info("Using rule-based ranker (no model loaded)")
    
    def _load_model(self, model_path: str):
        """Load trained neural ranker"""
        try:
            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}, using rule-based ranker")
                return
            
            self.model = RankerMLP().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Neural ranker loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading ranker model: {e}")
            logger.info("Falling back to rule-based ranker")
            self.model = None
    
    def rank(self, candidates: List[Dict], user_id: str, already_ranked: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Rerank candidates using neural or rule-based scoring
        
        Args:
            candidates: List of candidate items with 'id', 'score', etc.
            user_id: User ID for context
            already_ranked: Previously ranked items (for diversity calculation)
            
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            return []
        
        try:
            # Extract features for all candidates
            features_list = []
            for candidate in candidates:
                features = self._extract_features(candidate, already_ranked or [])
                features_list.append(features)
            
            # Get ranking scores
            if self.model is not None and not self.use_rule_based:
                # Neural scoring
                ranking_scores = self._neural_score(features_list)
            else:
                # Rule-based scoring
                ranking_scores = self._rule_based_score(features_list)
            
            # Attach scores and sort
            for i, candidate in enumerate(candidates):
                candidate['ranking_score'] = float(ranking_scores[i])
            
            # Sort by ranking score (descending)
            ranked_candidates = sorted(
                candidates,
                key=lambda x: x['ranking_score'],
                reverse=True
            )
            
            return ranked_candidates
            
        except Exception as e:
            logger.error(f"Error in ranking: {e}")
            # Fallback: return original order
            return candidates
    
    def _extract_features(self, candidate: Dict, already_ranked: List[Dict]) -> Dict:
        """
        Extract ranking features from candidate
        
        Features:
        1. model_score: Score from recommendation model (normalized)
        2. recency_score: How recent the item is
        3. popularity_score: Views, interactions (normalized)
        4. diversity_penalty: Similarity to already ranked items
        """
        from engines.ranker_features import (
            normalize_score,
            calculate_recency_score,
            calculate_popularity_score,
            calculate_diversity_penalty
        )
        
        # 1. Model score (already provided by recommender)
        model_score = normalize_score(candidate.get('score', 0.0))
        
        # 2. Recency score
        recency_score = calculate_recency_score(
            created_at=candidate.get('created_at'),
            updated_at=candidate.get('updated_at')
        )
        
        # 3. Popularity score
        popularity_score = calculate_popularity_score(
            views=candidate.get('views', 0),
            interaction_count=candidate.get('interaction_count', 0)
        )
        
        # 4. Diversity penalty
        diversity_penalty = calculate_diversity_penalty(
            candidate=candidate,
            already_ranked=already_ranked
        )
        
        return {
            'model_score': model_score,
            'recency': recency_score,
            'popularity': popularity_score,
            'diversity': diversity_penalty,
            'original_score': candidate.get('score', 0.0)
        }
    
    def _neural_score(self, features_list: List[Dict]) -> np.ndarray:
        """Score using neural network"""
        # Convert features to tensor
        features_array = np.array([
            [f['model_score'], f['recency'], f['popularity'], f['diversity'], f['original_score']]
            for f in features_list
        ], dtype=np.float32)
        
        with torch.no_grad():
            features_tensor = torch.from_numpy(features_array).to(self.device)
            scores = self.model(features_tensor)
            return scores.cpu().numpy()
    
    def _rule_based_score(self, features_list: List[Dict]) -> np.ndarray:
        """Score using weighted rule-based formula"""
        scores = []
        for features in features_list:
            score = (
                self.rule_weights['model_score'] * features['model_score'] +
                self.rule_weights['recency'] * features['recency'] +
                self.rule_weights['popularity'] * features['popularity'] +
                self.rule_weights['diversity'] * features['diversity']
            )
            scores.append(score)
        
        return np.array(scores)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for training
    Ensures positive items ranked higher than negative items
    """
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Scores for positive (relevant) items
            neg_scores: Scores for negative (irrelevant) items
        
        Returns:
            Loss value
        """
        # Want: pos_scores > neg_scores + margin
        # Loss: max(0, margin - (pos_scores - neg_scores))
        loss = torch.clamp(self.margin - (pos_scores - neg_scores), min=0)
        return loss

