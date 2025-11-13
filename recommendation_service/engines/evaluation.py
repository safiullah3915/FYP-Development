"""
Evaluation Metrics for Recommendation Systems
Implements ranking metrics: Precision@K, Recall@K, NDCG@K, MAP, Hit Rate, Coverage
"""
import numpy as np
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Precision@K: Proportion of recommended items in top-K that are relevant
    
    Args:
        y_true: Binary relevance labels (1 = relevant, 0 = not relevant)
        y_pred: Predicted scores
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    # Get top-k indices
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    # Count relevant items in top-k
    relevant_in_top_k = np.sum(y_true[top_k_idx] > 0.5)  # threshold at 0.5
    
    return relevant_in_top_k / k


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Recall@K: Proportion of relevant items that appear in top-K
    
    Args:
        y_true: Binary relevance labels
        y_pred: Predicted scores
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    total_relevant = np.sum(y_true > 0.5)
    if total_relevant == 0:
        return 0.0
    
    # Get top-k indices
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    # Count relevant items in top-k
    relevant_in_top_k = np.sum(y_true[top_k_idx] > 0.5)
    
    return relevant_in_top_k / total_relevant


def f1_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K
    
    Args:
        y_true: Binary relevance labels
        y_pred: Predicted scores
        k: Number of top recommendations to consider
        
    Returns:
        F1@K score
    """
    prec = precision_at_k(y_true, y_pred, k)
    rec = recall_at_k(y_true, y_pred, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K
    
    Args:
        y_true: Relevance labels (can be graded, e.g., 0.0 to 1.0)
        y_pred: Predicted scores
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    # Get top-k indices based on predictions
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, idx in enumerate(top_k_idx):
        # Gain is the relevance score, discounted by position
        gain = y_true[idx]
        discount = np.log2(i + 2)  # +2 because positions start at 1
        dcg += gain / discount
    
    # IDCG: Ideal DCG (using perfect ranking)
    ideal_idx = np.argsort(y_true)[::-1][:k]
    idcg = 0.0
    for i, idx in enumerate(ideal_idx):
        gain = y_true[idx]
        discount = np.log2(i + 2)
        idcg += gain / discount
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average Precision: Average of precision values at each relevant item position
    
    Args:
        y_true: Binary relevance labels
        y_pred: Predicted scores
        
    Returns:
        Average Precision score
    """
    total_relevant = np.sum(y_true > 0.5)
    if total_relevant == 0:
        return 0.0
    
    # Sort by predicted scores
    sorted_idx = np.argsort(y_pred)[::-1]
    
    ap = 0.0
    relevant_count = 0
    
    for i, idx in enumerate(sorted_idx):
        if y_true[idx] > 0.5:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            ap += precision
    
    return ap / total_relevant


def mean_average_precision(y_true_list: List[np.ndarray], y_pred_list: List[np.ndarray]) -> float:
    """
    MAP: Mean Average Precision across multiple queries
    
    Args:
        y_true_list: List of relevance arrays (one per query/user)
        y_pred_list: List of prediction arrays (one per query/user)
        
    Returns:
        MAP score
    """
    if len(y_true_list) == 0:
        return 0.0
    
    ap_scores = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        if len(y_true) > 0:
            ap = average_precision(y_true, y_pred)
            ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def hit_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Hit Rate@K: Whether at least one relevant item appears in top-K
    
    Args:
        y_true: Binary relevance labels
        y_pred: Predicted scores
        k: Number of top recommendations to consider
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if len(y_true) == 0:
        return 0.0
    
    # Get top-k indices
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    
    # Check if any relevant items in top-k
    has_relevant = np.any(y_true[top_k_idx] > 0.5)
    
    return 1.0 if has_relevant else 0.0


def coverage(recommended_items: List[List[int]], total_items: int) -> float:
    """
    Coverage: Proportion of all items that have been recommended at least once
    
    Args:
        recommended_items: List of recommendation lists (one per user)
        total_items: Total number of items in the catalog
        
    Returns:
        Coverage score (0.0 to 1.0)
    """
    if total_items == 0:
        return 0.0
    
    unique_recommended = set()
    for rec_list in recommended_items:
        unique_recommended.update(rec_list)
    
    return len(unique_recommended) / total_items


class RankingEvaluator:
    """Evaluator for ranking-based recommendation metrics"""
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator
        
        Args:
            k_values: List of K values to evaluate (e.g., [10, 20, 50])
        """
        self.k_values = k_values if k_values else [10, 20, 50]
    
    def evaluate_batch(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a single batch/query
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted scores (N,)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        for k in self.k_values:
            if len(y_true) >= k:
                metrics[f'precision@{k}'] = precision_at_k(y_true, y_pred, k)
                metrics[f'recall@{k}'] = recall_at_k(y_true, y_pred, k)
                metrics[f'f1@{k}'] = f1_at_k(y_true, y_pred, k)
                metrics[f'ndcg@{k}'] = ndcg_at_k(y_true, y_pred, k)
                metrics[f'hit_rate@{k}'] = hit_rate_at_k(y_true, y_pred, k)
        
        metrics['map'] = average_precision(y_true, y_pred)
        
        return metrics
    
    def evaluate_dataset(
        self,
        y_true_list: List[np.ndarray],
        y_pred_list: List[np.ndarray],
        user_ids: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset (multiple queries/users)
        
        Args:
            y_true_list: List of true label arrays
            y_pred_list: List of predicted score arrays
            user_ids: Optional list of user IDs for debugging
            
        Returns:
            Dictionary of averaged metric scores
        """
        if len(y_true_list) != len(y_pred_list):
            raise ValueError("y_true_list and y_pred_list must have same length")
        
        all_metrics = []
        
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            if len(y_true) == 0:
                continue
            
            batch_metrics = self.evaluate_batch(y_true, y_pred)
            all_metrics.append(batch_metrics)
        
        # Average metrics
        if not all_metrics:
            return {f'{metric}@{k}': 0.0 for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate'] for k in self.k_values}
        
        averaged_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            averaged_metrics[key] = np.mean(values) if values else 0.0
        
        logger.info("Evaluation Results:")
        for metric_name, value in sorted(averaged_metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return averaged_metrics


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate model predictions
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted scores (N,)
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary of metric scores
    """
    evaluator = RankingEvaluator(k_values)
    return evaluator.evaluate_batch(y_true, y_pred)

