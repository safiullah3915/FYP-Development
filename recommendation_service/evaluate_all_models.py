#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates all recommendation models and calculates real metrics
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from engines.evaluation import RankingEvaluator
from database.connection import SessionLocal
from database.models import User, Startup, UserInteraction
from utils.logger import get_logger

logger = get_logger(__name__)

# Import model inference classes
from inference_two_tower import TwoTowerInference
from inference_als import ALSInference
from inference_als_reverse import ALSReverseInference
from engines.ranker import NeuralRanker


def evaluate_two_tower_model(model_path, model_name, use_case, is_reverse=False):
    """Evaluate Two-Tower model on real user interactions"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Two-Tower Model: {model_name}")
    logger.info(f"Use Case: {use_case}")
    logger.info(f"{'='*60}")
    
    try:
        # Load model
        model = TwoTowerInference(str(model_path))
        
        db = SessionLocal()
        metrics = {
            'model_name': model_name,
            'model_type': 'two_tower',
            'use_case': use_case,
            'is_reverse': is_reverse,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Get test users (users with interactions)
        if is_reverse:
            # For reverse: startups that have been interacted with (from UserInteraction)
            test_entities = db.query(UserInteraction.startup_id).distinct().limit(50).all()
            test_entity_ids = [str(row[0]) for row in test_entities]
        else:
            # For forward: users with interactions
            test_entities = db.query(UserInteraction.user_id).distinct().limit(50).all()
            test_entity_ids = [str(row[0]) for row in test_entities]
        
        if not test_entity_ids:
            logger.warning(f"No test entities found for {model_name}")
            return None
        
        logger.info(f"Evaluating on {len(test_entity_ids)} test entities")
        
        # Collect predictions and ground truth
        all_predictions = []
        all_labels = []
        all_recommended_items = []
        
        evaluator = RankingEvaluator(k_values=[10, 20, 50])
        
        for entity_id in tqdm(test_entity_ids[:20], desc="Evaluating"):  # Limit to 20 for speed
            try:
                # Get recommendations
                if is_reverse:
                    # For reverse models, use startup_id parameter
                    result = model.recommend(
                        startup_id=entity_id,
                        limit=50
                    )
                    # Convert to standard format
                    if 'users' in result:
                        recommended_ids = [r['id'] for r in result['users']]
                        scores = result.get('scores', {})
                    else:
                        recommended_ids = result.get('item_ids', [])
                        scores = result.get('scores', {})
                else:
                    result = model.recommend(
                        user_id=entity_id,
                        limit=50
                    )
                    # Handle different response formats
                    if 'startups' in result:
                        recommended_ids = [r['id'] for r in result['startups']]
                        scores = result.get('scores', {})
                    else:
                        recommended_ids = result.get('item_ids', [])
                        scores = result.get('scores', {})
                
                recommended_ids = result.get('item_ids', [])
                scores = result.get('scores', {})
                
                if not recommended_ids:
                    continue
                
                # Get ground truth (actual interactions)
                if is_reverse:
                    # For reverse: get users who interacted with this startup (positive signals)
                    interactions = db.query(UserInteraction).filter(
                        UserInteraction.startup_id == entity_id,
                        UserInteraction.interaction_type.in_(['like', 'favorite', 'apply', 'interest'])
                    ).all()
                    positive_items = {str(inter.user_id) for inter in interactions}
                else:
                    # For forward: get startups user interacted with
                    interactions = db.query(UserInteraction).filter(
                        UserInteraction.user_id == entity_id,
                        UserInteraction.interaction_type.in_(['like', 'favorite', 'apply', 'interest'])
                    ).all()
                    positive_items = {str(inter.startup_id) for inter in interactions}
                
                if not positive_items:
                    continue
                
                # Create binary labels for recommended items
                labels = np.array([1.0 if item_id in positive_items else 0.0 for item_id in recommended_ids])
                pred_scores = np.array([scores.get(item_id, 0.0) for item_id in recommended_ids])
                
                # Evaluate this user
                user_metrics = evaluator.evaluate_batch(labels, pred_scores)
                
                all_predictions.append(pred_scores)
                all_labels.append(labels)
                all_recommended_items.extend(recommended_ids)
                
            except Exception as e:
                logger.warning(f"Error evaluating entity {entity_id}: {e}")
                continue
        
        db.close()
        
        if not all_predictions:
            logger.warning(f"No valid predictions for {model_name}")
            return None
        
        # Aggregate metrics across all users
        # Calculate per-user metrics and average
        per_user_metrics = []
        for labels, preds in zip(all_labels, all_predictions):
            if len(labels) > 0:
                user_metrics = evaluator.evaluate_batch(labels, preds)
                per_user_metrics.append(user_metrics)
        
        if per_user_metrics:
            # Average across users
            avg_metrics = {}
            for key in per_user_metrics[0].keys():
                values = [m[key] for m in per_user_metrics if key in m]
                avg_metrics[key] = float(np.mean(values)) if values else 0.0
            
            metrics['metrics'] = avg_metrics
            metrics['num_test_entities'] = len(all_predictions)
            metrics['total_recommendations'] = len(all_recommended_items)
            
            # Calculate coverage
            unique_items = len(set(all_recommended_items))
            if is_reverse:
                total_items = db.query(User).count()
            else:
                total_items = db.query(Startup).filter(Startup.status == 'active').count()
            
            metrics['metrics']['coverage'] = unique_items / total_items if total_items > 0 else 0.0
            metrics['coverage_details'] = {
                'unique_recommended': unique_items,
                'total_items': total_items
            }
            
            logger.info(f"\nMetrics for {model_name}:")
            for key, value in sorted(avg_metrics.items()):
                logger.info(f"  {key}: {value:.4f}")
            
            return metrics
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def evaluate_als_model(model_path, model_name, use_case, is_reverse=False):
    """Evaluate ALS model on real user interactions"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating ALS Model: {model_name}")
    logger.info(f"Use Case: {use_case}")
    logger.info(f"{'='*60}")
    
    try:
        # Load model
        if is_reverse:
            model = ALSReverseInference(str(model_path))
        else:
            model = ALSInference(str(model_path))
        
        db = SessionLocal()
        metrics = {
            'model_name': model_name,
            'model_type': 'als',
            'use_case': use_case,
            'is_reverse': is_reverse,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Get test users
        if is_reverse:
            # For reverse: startups that have been interacted with
            test_entities = db.query(UserInteraction.startup_id).distinct().limit(50).all()
            test_entity_ids = [str(row[0]) for row in test_entities]
        else:
            test_entities = db.query(UserInteraction.user_id).distinct().limit(50).all()
            test_entity_ids = [str(row[0]) for row in test_entities]
        
        if not test_entity_ids:
            logger.warning(f"No test entities found for {model_name}")
            return None
        
        logger.info(f"Evaluating on {len(test_entity_ids)} test entities")
        
        all_predictions = []
        all_labels = []
        all_recommended_items = []
        
        evaluator = RankingEvaluator(k_values=[10, 20, 50])
        
        for entity_id in tqdm(test_entity_ids[:20], desc="Evaluating"):
            try:
                # Get recommendations
                if is_reverse:
                    result = model.recommend(
                        startup_id=entity_id,
                        limit=50
                    )
                    if 'users' in result:
                        recommended_ids = [r['id'] for r in result['users']]
                        scores = result.get('scores', {})
                    else:
                        recommended_ids = result.get('item_ids', [])
                        scores = result.get('scores', {})
                else:
                    result = model.recommend(
                        user_id=entity_id,
                        limit=50
                    )
                    if 'startups' in result:
                        recommended_ids = [r['id'] for r in result['startups']]
                        scores = result.get('scores', {})
                    else:
                        recommended_ids = result.get('item_ids', [])
                        scores = result.get('scores', {})
                
                if not recommended_ids:
                    continue
                
                # Get ground truth
                if is_reverse:
                    # For reverse: get users who interacted with this startup
                    interactions = db.query(UserInteraction).filter(
                        UserInteraction.startup_id == entity_id,
                        UserInteraction.interaction_type.in_(['like', 'favorite', 'apply', 'interest'])
                    ).all()
                    positive_items = {str(inter.user_id) for inter in interactions}
                else:
                    interactions = db.query(UserInteraction).filter(
                        UserInteraction.user_id == entity_id,
                        UserInteraction.interaction_type.in_(['like', 'favorite', 'apply', 'interest'])
                    ).all()
                    positive_items = {str(inter.startup_id) for inter in interactions}
                
                if not positive_items:
                    continue
                
                labels = np.array([1.0 if item_id in positive_items else 0.0 for item_id in recommended_ids])
                pred_scores = np.array([scores.get(item_id, 0.0) for item_id in recommended_ids])
                
                all_predictions.append(pred_scores)
                all_labels.append(labels)
                all_recommended_items.extend(recommended_ids)
                
            except Exception as e:
                logger.warning(f"Error evaluating entity {entity_id}: {e}")
                continue
        
        db.close()
        
        if not all_predictions:
            logger.warning(f"No valid predictions for {model_name}")
            return None
        
        # Aggregate metrics
        per_user_metrics = []
        for labels, preds in zip(all_labels, all_predictions):
            if len(labels) > 0:
                user_metrics = evaluator.evaluate_batch(labels, preds)
                per_user_metrics.append(user_metrics)
        
        if per_user_metrics:
            avg_metrics = {}
            for key in per_user_metrics[0].keys():
                values = [m[key] for m in per_user_metrics if key in m]
                avg_metrics[key] = float(np.mean(values)) if values else 0.0
            
            metrics['metrics'] = avg_metrics
            metrics['num_test_entities'] = len(all_predictions)
            metrics['total_recommendations'] = len(all_recommended_items)
            
            # Coverage
            unique_items = len(set(all_recommended_items))
            db = SessionLocal()
            if is_reverse:
                total_items = db.query(User).count()
            else:
                total_items = db.query(Startup).filter(Startup.status == 'active').count()
            db.close()
            
            metrics['metrics']['coverage'] = unique_items / total_items if total_items > 0 else 0.0
            metrics['coverage_details'] = {
                'unique_recommended': unique_items,
                'total_items': total_items
            }
            
            logger.info(f"\nMetrics for {model_name}:")
            for key, value in sorted(avg_metrics.items()):
                logger.info(f"  {key}: {value:.4f}")
            
            return metrics
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_existing_metrics():
    """Load existing metrics from saved files"""
    models_dir = Path(__file__).parent / "models"
    metrics_file = models_dir / "all_models_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def save_all_metrics(all_metrics):
    """Save all metrics to JSON file"""
    models_dir = Path(__file__).parent / "models"
    metrics_file = models_dir / "all_models_metrics.json"
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"\nAll metrics saved to {metrics_file}")


def main():
    """Main evaluation function"""
    logger.info("="*60)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*60)
    
    models_dir = Path(__file__).parent / "models"
    all_metrics = {}
    
    # Evaluate Two-Tower Forward
    two_tower_path = models_dir / "two_tower_v1_best.pth"
    if two_tower_path.exists():
        metrics = evaluate_two_tower_model(
            two_tower_path,
            "two_tower_v1",
            "developer_startup",
            is_reverse=False
        )
        if metrics:
            all_metrics['two_tower_v1'] = metrics
    
    # Evaluate Two-Tower Reverse
    two_tower_reverse_path = models_dir / "two_tower_reverse_v1_best.pth"
    if two_tower_reverse_path.exists():
        # For reverse, TwoTowerInference uses user_id parameter but treats it as startup_id
        # The model itself is trained in reverse mode
        try:
            from inference_two_tower import TwoTowerInference
            model = TwoTowerInference(str(two_tower_reverse_path))
            
            db = SessionLocal()
            metrics = {
                'model_name': "two_tower_reverse_v1",
                'model_type': 'two_tower',
                'use_case': "startup_developer",
                'is_reverse': True,
                'evaluation_date': datetime.now().isoformat(),
                'metrics': {}
            }
            
            # Get test startups
            test_entities = db.query(UserInteraction.startup_id).distinct().limit(50).all()
            test_entity_ids = [str(row[0]) for row in test_entities]
            
            if test_entity_ids:
                logger.info(f"Evaluating on {len(test_entity_ids)} test entities")
                
                all_predictions = []
                all_labels = []
                all_recommended_items = []
                
                evaluator = RankingEvaluator(k_values=[10, 20, 50])
                
                for entity_id in tqdm(test_entity_ids[:20], desc="Evaluating"):
                    try:
                        # For reverse, pass startup_id as user_id to the model
                        result = model.recommend(
                            user_id=entity_id,  # Actually a startup_id in reverse mode
                            limit=50
                        )
                        
                        if 'startups' in result:
                            recommended_ids = [r['id'] for r in result['startups']]
                            scores = result.get('scores', {})
                        else:
                            recommended_ids = result.get('item_ids', [])
                            scores = result.get('scores', {})
                        
                        if not recommended_ids:
                            continue
                        
                        # Get ground truth: users who interacted with this startup
                        interactions = db.query(UserInteraction).filter(
                            UserInteraction.startup_id == entity_id,
                            UserInteraction.interaction_type.in_(['like', 'favorite', 'apply', 'interest'])
                        ).all()
                        positive_items = {str(inter.user_id) for inter in interactions}
                        
                        if not positive_items:
                            continue
                        
                        labels = np.array([1.0 if item_id in positive_items else 0.0 for item_id in recommended_ids])
                        pred_scores = np.array([scores.get(item_id, 0.0) for item_id in recommended_ids])
                        
                        all_predictions.append(pred_scores)
                        all_labels.append(labels)
                        all_recommended_items.extend(recommended_ids)
                        
                    except Exception as e:
                        logger.warning(f"Error evaluating entity {entity_id}: {e}")
                        continue
                
                if all_predictions:
                    per_user_metrics = []
                    for labels, preds in zip(all_labels, all_predictions):
                        if len(labels) > 0:
                            user_metrics = evaluator.evaluate_batch(labels, preds)
                            per_user_metrics.append(user_metrics)
                    
                    if per_user_metrics:
                        avg_metrics = {}
                        for key in per_user_metrics[0].keys():
                            values = [m[key] for m in per_user_metrics if key in m]
                            avg_metrics[key] = float(np.mean(values)) if values else 0.0
                        
                        metrics['metrics'] = avg_metrics
                        metrics['num_test_entities'] = len(all_predictions)
                        metrics['total_recommendations'] = len(all_recommended_items)
                        
                        # Coverage
                        unique_items = len(set(all_recommended_items))
                        total_items = db.query(User).count()
                        
                        metrics['metrics']['coverage'] = unique_items / total_items if total_items > 0 else 0.0
                        metrics['coverage_details'] = {
                            'unique_recommended': unique_items,
                            'total_items': total_items
                        }
                        
                        logger.info(f"\nMetrics for two_tower_reverse_v1:")
                        for key, value in sorted(avg_metrics.items()):
                            logger.info(f"  {key}: {value:.4f}")
                        
                        all_metrics['two_tower_reverse_v1'] = metrics
            
            db.close()
        except Exception as e:
            logger.error(f"Error evaluating two_tower_reverse_v1: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Evaluate ALS Forward
    als_config = models_dir / "als_v1_config.json"
    if als_config.exists():
        metrics = evaluate_als_model(
            str(als_config),
            "als_v1",
            "developer_startup",
            is_reverse=False
        )
        if metrics:
            all_metrics['als_v1'] = metrics
    
    # Evaluate ALS Reverse
    als_reverse_config = models_dir / "als_reverse_v1_config.json"
    if als_reverse_config.exists():
        metrics = evaluate_als_model(
            str(als_reverse_config),
            "als_reverse_v1",
            "startup_developer",
            is_reverse=True
        )
        if metrics:
            all_metrics['als_reverse_v1'] = metrics
    
    # Load training history for additional metrics
    two_tower_history = models_dir / "two_tower_v1_history.json"
    if two_tower_history.exists():
        with open(two_tower_history, 'r') as f:
            history = json.load(f)
            if 'two_tower_v1' not in all_metrics:
                all_metrics['two_tower_v1'] = {}
            all_metrics['two_tower_v1']['training_history'] = history
    
    two_tower_reverse_history = models_dir / "two_tower_reverse_v1_history.json"
    if two_tower_reverse_history.exists():
        with open(two_tower_reverse_history, 'r') as f:
            history = json.load(f)
            if 'two_tower_reverse_v1' not in all_metrics:
                all_metrics['two_tower_reverse_v1'] = {}
            all_metrics['two_tower_reverse_v1']['training_history'] = history
    
    # Load ALS configs for additional info
    if als_config.exists():
        with open(als_config, 'r') as f:
            config = json.load(f)
            if 'als_v1' not in all_metrics:
                all_metrics['als_v1'] = {}
            all_metrics['als_v1']['config'] = config
    
    if als_reverse_config.exists():
        with open(als_reverse_config, 'r') as f:
            config = json.load(f)
            if 'als_reverse_v1' not in all_metrics:
                all_metrics['als_reverse_v1'] = {}
            all_metrics['als_reverse_v1']['config'] = config
    
    # Save all metrics
    save_all_metrics(all_metrics)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Evaluated {len(all_metrics)} models")
    logger.info(f"Metrics saved to: {models_dir / 'all_models_metrics.json'}")


if __name__ == '__main__':
    main()

