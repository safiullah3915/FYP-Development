"""
Flask Recommendation Service Application
Minimal Flask app for testing database connection and models
"""
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import hashlib
from copy import deepcopy
from datetime import datetime
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, CORS_ORIGINS
from database.connection import check_db_connection, SessionLocal
from database.models import User, Startup, UserInteraction
# Note: StartupInteraction is in Django backend, accessed via API if needed
from utils.logger import get_logger

logger = get_logger(__name__)

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, origins=CORS_ORIGINS)

# ============================================================================
# Initialize ML Models
# ============================================================================
two_tower_model = None
als_model = None
ensemble_model = None
MODELS_DIR = Path(__file__).parent / "models"
ALS_MODEL_NAME = "als_v1"
ALS_REVERSE_MODEL_NAME = "als_reverse_v1"
TWO_TOWER_MODEL_PATH = MODELS_DIR / "two_tower_v1_best.pth"
TWO_TOWER_REVERSE_MODEL_PATH = MODELS_DIR / "two_tower_reverse_v1_best.pth"
RANKER_REVERSE_MODEL_PATH = MODELS_DIR / "ranker_reverse_v1.pth"


def resolve_model_artifact(base_name: str, prefer_config: bool = True) -> Path:
    """
    Locate the best available artifact pointer for a model prefix.
    Preference order:
      1. <base_name>_config.json
      2. <base_name>.pkl (legacy)
    """
    candidates = []
    if prefer_config:
        candidates.append(MODELS_DIR / f"{base_name}_config.json")
    candidates.append(MODELS_DIR / f"{base_name}.pkl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

# Load recommendation models
logger.info("Loading recommendation models...")

# Load Two-Tower Model (Forward: User -> Startup)
try:
    from inference_two_tower import TwoTowerInference
    model_path = TWO_TOWER_MODEL_PATH
    
    if model_path.exists():
        two_tower_model = TwoTowerInference(str(model_path))
    else:
        logger.warning(f"Two-Tower Forward model not found at {model_path}")
except Exception as e:
    logger.error(f"Failed to load Two-Tower Forward model: {e}")

# Load Two-Tower Reverse Model (Reverse: Startup -> User)
two_tower_reverse_model = None
try:
    from inference_two_tower import TwoTowerInference
    model_path = TWO_TOWER_REVERSE_MODEL_PATH
    
    if model_path.exists():
        two_tower_reverse_model = TwoTowerInference(str(model_path))
    else:
        logger.warning(f"Two-Tower Reverse model not found at {TWO_TOWER_REVERSE_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load Two-Tower Reverse model: {e}")

# Load ALS Model (Forward: User -> Startup)
try:
    from inference_als import ALSInference
    als_pointer = resolve_model_artifact(ALS_MODEL_NAME)
    
    if als_pointer.exists():
        als_model = ALSInference(str(als_pointer))
    else:
        logger.warning(f"ALS Forward artifacts not found (expected {als_pointer})")
except Exception as e:
    logger.error(f"Failed to load ALS Forward model: {e}")

# Load ALS Reverse Model (Reverse: Startup -> User)
als_reverse_model = None
try:
    from inference_als_reverse import ALSReverseInference
    als_reverse_pointer = resolve_model_artifact(ALS_REVERSE_MODEL_NAME)
    
    if als_reverse_pointer.exists():
        als_reverse_model = ALSReverseInference(str(als_reverse_pointer))
    else:
        logger.warning(f"ALS Reverse artifacts not found (expected {als_reverse_pointer})")
except Exception as e:
    logger.error(f"Failed to load ALS Reverse model: {e}")

# Load Ensemble Model (if both base models available)
try:
    if two_tower_model and als_model:
        from inference_ensemble import EnsembleInference
        ensemble_model = EnsembleInference(
            als_model_path=str(resolve_model_artifact(ALS_MODEL_NAME)),
            two_tower_model_path=str(TWO_TOWER_MODEL_PATH),
            als_weight=0.6
        )
except Exception as e:
    logger.error(f"Failed to initialize ensemble: {e}")

# Load Ranker Model (reranks forward recommendations for better quality)
ranker_model = None
try:
    ranker_path = Path(__file__).parent / "models" / "ranker_v1.pth"
    
    # Import directly from module to avoid circular import through __init__
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("ranker_module", 
                                                   Path(__file__).parent / "engines" / "ranker.py")
    ranker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranker_module)
    NeuralRanker = ranker_module.NeuralRanker
    
    if ranker_path.exists():
        ranker_model = NeuralRanker(str(ranker_path))
    else:
        ranker_model = NeuralRanker(use_rule_based=True)
except Exception as e:
    logger.warning(f"Could not load ranker forward: {e}")
    ranker_model = None

# Load Ranker Reverse Model (reranks reverse recommendations for better quality)
ranker_reverse_model = None
try:
    ranker_reverse_path = RANKER_REVERSE_MODEL_PATH
    
    # Import directly from module to avoid circular import through __init__
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("ranker_module", 
                                                   Path(__file__).parent / "engines" / "ranker.py")
    ranker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranker_module)
    NeuralRanker = ranker_module.NeuralRanker
    
    if ranker_reverse_path.exists():
        ranker_reverse_model = NeuralRanker(str(ranker_reverse_path))
    else:
        ranker_reverse_model = NeuralRanker(use_rule_based=True)
except Exception as e:
    logger.warning(f"Could not load ranker reverse: {e}")
    ranker_reverse_model = None

# Log model loading summary
loaded_models = []
if 'two_tower_model' in locals() and two_tower_model:
    loaded_models.append("Two-Tower Forward")
if 'two_tower_reverse_model' in locals() and two_tower_reverse_model:
    loaded_models.append("Two-Tower Reverse")
if 'als_model' in locals() and als_model:
    loaded_models.append("ALS Forward")
if 'als_reverse_model' in locals() and als_reverse_model:
    loaded_models.append("ALS Reverse")
if 'ensemble_model' in locals() and ensemble_model:
    loaded_models.append("Ensemble")
if 'ranker_model' in locals() and ranker_model:
    loaded_models.append("Ranker Forward")
if 'ranker_reverse_model' in locals() and ranker_reverse_model:
    loaded_models.append("Ranker Reverse")

if loaded_models:
    logger.info(f"Models loaded successfully: {', '.join(loaded_models)}")
else:
    logger.warning("No models loaded - using content-based recommendations only")


def apply_ranker(results, user_id, limit, method_used, use_reverse_ranker=False):
    """
    Apply ranker to reorder recommendations
    
    Args:
        results: Dict with 'startups' or similar key containing recommendations
        user_id: User ID or Startup ID for context
        limit: Final number of items to return
        method_used: Which recommendation method was used
        use_reverse_ranker: If True, use reverse ranker model
    
    Returns:
        Updated results dict with reranked items
    """
    # Select appropriate ranker model
    ranker = ranker_reverse_model if use_reverse_ranker else ranker_model
    
    if not ranker:
        return results
    
    # Only rank personalized recommendations (not trending/popular)
    if method_used in ['trending', 'popular']:
        return results
    
    # Get candidates key (could be 'startups', 'developers', 'investors')
    candidates_key = None
    for key in ['startups', 'developers', 'investors']:
        if key in results and results[key]:
            candidates_key = key
            break
    
    if not candidates_key:
        return results
    
    candidates = results[candidates_key]
    
    if len(candidates) == 0:
        return results
    
    try:
        # Rerank candidates
        reranked = ranker.rank(
            candidates=candidates,
            user_id=user_id,
            already_ranked=[]
        )
        
        # Update results
        results[candidates_key] = reranked[:limit]
        results['reranked'] = True
        
        ranker_type = "reverse" if use_reverse_ranker else "forward"
        logger.info(f"Reranked {len(candidates)} candidates to {len(results[candidates_key])} items using {ranker_type} ranker")
        
    except Exception as e:
        logger.error(f"Error applying ranker: {e}")
        # Keep original order on error
    
    return results


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Tests database connection and returns service status
    """
    try:
        db_connected = check_db_connection()
        
        return jsonify({
            'status': 'healthy' if db_connected else 'unhealthy',
            'service': 'recommendation-service',
            'version': '0.1.0',
            'database_connected': db_connected
        }), 200 if db_connected else 503
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/test/users', methods=['GET'])
def test_users():
    """
    Test endpoint to query users table
    Returns first 5 users to verify database connection works
    """
    db = SessionLocal()
    try:
        users = db.query(User).limit(5).all()
        
        return jsonify({
            'count': len(users),
            'users': [{
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'role': user.role
            } for user in users]
        }), 200
    
    except Exception as e:
        logger.error(f"Error querying users: {e}")
        return jsonify({
            'error': str(e)
        }), 500
    
    finally:
        db.close()


@app.route('/test/startups', methods=['GET'])
def test_startups():
    """
    Test endpoint to query startups table
    Returns first 5 startups to verify database connection works
    """
    db = SessionLocal()
    try:
        startups = db.query(Startup).limit(5).all()
        
        return jsonify({
            'count': len(startups),
            'startups': [{
                'id': startup.id,
                'title': startup.title,
                'category': startup.category,
                'type': startup.type,
                'status': startup.status
            } for startup in startups]
        }), 200
    
    except Exception as e:
        logger.error(f"Error querying startups: {e}")
        return jsonify({
            'error': str(e)
        }), 500
    
    finally:
        db.close()


# ============================================================================
# RECOMMENDATION API ENDPOINTS
# ============================================================================

# NOTE: Session ID generation removed - frontend generates session_id
# This function is deprecated and not used. Session data is formatted by SessionService.
# Frontend generates session_id and sends it to Django for storage.


@app.route('/api/recommendations/startups/for-developer/<user_id>', methods=['GET'])
def get_startups_for_developer(user_id):
    """
    Get startup recommendations for a developer (student/professional)
    
    Query params:
        limit: Number of results (default: 10)
        type: Filter by 'marketplace' or 'collaboration' (optional)
    
    Returns:
        List of recommended startups with scores and match reasons
    """
    db = SessionLocal()
    try:
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            return jsonify({'error': 'Invalid user_id'}), 400
        
        cache_indicator = request.args.get('cache_indicator', 'default')
        force_refresh = _should_bust_cache()
        
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 10))
            if limit < 1:
                limit = 10
            # Hard cap at 10 for frontend expectations
            if limit > 10:
                limit = 10
        except (ValueError, TypeError):
            limit = 10
        
        startup_type = request.args.get('type', None)
        if startup_type and startup_type not in ['marketplace', 'collaboration']:
            startup_type = None
        
        # Build filters
        filters = {}
        if startup_type:
            filters['type'] = startup_type
        
        # Allow backend (Django) to enforce open positions; only respect explicit param
        require_open_positions_param = request.args.get('require_open_positions')
        if require_open_positions_param is not None and require_open_positions_param.lower() in ['1', 'true', 'yes']:
            filters['require_open_positions'] = True
        
        cache_params = {
            'limit': limit,
            'type': startup_type or '',
            'filters': json.dumps(filters, sort_keys=True),
            'offset': request.args.get('offset')
        }
        params_hash = _make_params_fingerprint(cache_params)
        namespace = 'developer_startup'
        if not force_refresh:
            cached_entry = _get_personalized_cache_entry(namespace, user_id, params_hash, cache_indicator)
            if cached_entry:
                cached_data, cache_meta = cached_entry
                logger.info(f"📦 Flask: Cache hit for {namespace} user={user_id} params={params_hash}")
                cached_data['cache_meta'] = cache_meta
                return jsonify(cached_data), 200
        
        # Check interaction count for routing
        # Normalize UUID format (remove dashes) since SQLite stores UUIDs without dashes
        normalized_user_id = str(user_id).replace('-', '')
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.user_id == normalized_user_id
        ).count()
        
        logger.info(f"[for-developer] user_id={user_id} interaction_count={interaction_count} type={startup_type} filters={filters}")
        
        # Smart routing based on interaction count
        if interaction_count < 5:
            # Cold start: content-based
            logger.info(f"[for-developer] routing=content_based (cold start: {interaction_count})")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
            results = rec_service.get_recommendations(
                user_id=user_id,
                use_case='developer_startup',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count < 20:
            # Warm users: ALS
            logger.info(f"[for-developer] routing=als (warm user: {interaction_count})")
            if als_model:
                results = als_model.recommend(user_id, limit, filters)
                method_used = 'als'
                model_version = 'als_v1.0'
            elif two_tower_model:
                logger.info("  ALS unavailable, falling back to Two-Tower")
                results = two_tower_model.recommend(user_id, limit, filters)
                method_used = 'two_tower'
                model_version = 'two_tower_v1.0'
            else:
                logger.info("  No models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=user_id,
                    use_case='developer_startup',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        else:
            # Hot users: Ensemble
            logger.info(f"[for-developer] routing=ensemble (hot user: {interaction_count})")
            if ensemble_model:
                results = ensemble_model.recommend(user_id, limit, filters)
                method_used = 'ensemble'
                model_version = 'ensemble_v1.0'
            elif als_model:
                logger.info("  Ensemble unavailable, falling back to ALS")
                results = als_model.recommend(user_id, limit, filters)
                method_used = 'als'
                model_version = 'als_v1.0'
            elif two_tower_model:
                logger.info("  ALS/Ensemble unavailable, falling back to Two-Tower")
                results = two_tower_model.recommend(user_id, limit, filters)
                method_used = 'two_tower'
                model_version = 'two_tower_v1.0'
            else:
                logger.info("  No models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=user_id,
                    use_case='developer_startup',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply ranker to reorder recommendations
        results = apply_ranker(results, user_id, limit, method_used)
        
        # Convert 'startups' array to 'item_ids' if needed (for session service compatibility)
        # Session service expects 'item_ids' but ensemble returns 'startups'
        if 'startups' in results and 'item_ids' not in results:
            results['item_ids'] = [startup.get('id') for startup in results.get('startups', [])]
            # Also ensure scores/match_reasons use item_ids as keys
            if results.get('item_ids'):
                new_scores = {}
                new_match_reasons = {}
                scores = results.get('scores', {})
                match_reasons = results.get('match_reasons', {})
                
                for startup in results.get('startups', []):
                    startup_id = startup.get('id')
                    if startup_id:
                        # Try both normalized and original format
                        startup_id_str = str(startup_id)
                        startup_id_normalized = startup_id_str.replace('-', '')
                        
                        # Look up score/match_reasons
                        score = scores.get(startup_id_normalized) or scores.get(startup_id_str) or scores.get(startup_id)
                        match_reason = match_reasons.get(startup_id_normalized) or match_reasons.get(startup_id_str) or match_reasons.get(startup_id)
                        
                        if score is not None:
                            new_scores[startup_id] = score
                        if match_reason:
                            new_match_reasons[startup_id] = match_reason
                
                results['scores'] = new_scores if new_scores else scores
                results['match_reasons'] = new_match_reasons if new_match_reasons else match_reasons
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=user_id,
            use_case='developer_startup',
            method=method_used,
            recommendations=results,
            model_version=model_version
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['interaction_count'] = interaction_count
        response['method_used'] = method_used
        
        cache_meta = _set_personalized_cache_entry(namespace, user_id, params_hash, cache_indicator, response)
        response_payload = deepcopy(response)
        response_payload['cache_meta'] = cache_meta
        
        return jsonify(response_payload), 200
        
    except Exception as e:
        logger.error(f"Error in get_startups_for_developer: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        db.close()


@app.route('/api/recommendations/startups/for-investor/<user_id>', methods=['GET'])
def get_startups_for_investor(user_id):
    """
    Get marketplace startup recommendations for an investor
    
    Query params:
        limit: Number of results (default: 10)
        category: Filter by category (optional)
    
    Returns:
        List of recommended marketplace startups with scores and match reasons
    """
    db = SessionLocal()
    try:
        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            return jsonify({'error': 'Invalid user_id'}), 400
        
        cache_indicator = request.args.get('cache_indicator', 'default')
        force_refresh = _should_bust_cache()
        
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 10))
            if limit < 1:
                limit = 10
            elif limit > 100:
                limit = 100  # Cap at 100 for performance
        except (ValueError, TypeError):
            limit = 10
        
        category = request.args.get('category', None)
        
        # Build filters (investors only see marketplace)
        filters = {'type': 'marketplace'}
        if category:
            filters['category'] = category
        
        cache_params = {
            'limit': limit,
            'category': category or '',
            'filters': json.dumps(filters, sort_keys=True),
            'min_funding': request.args.get('min_funding'),
            'max_funding': request.args.get('max_funding')
        }
        params_hash = _make_params_fingerprint(cache_params)
        namespace = 'investor_startup'
        if not force_refresh:
            cached_entry = _get_personalized_cache_entry(namespace, user_id, params_hash, cache_indicator)
            if cached_entry:
                cached_data, cache_meta = cached_entry
                logger.info(f"📦 Flask: Cache hit for {namespace} user={user_id} params={params_hash}")
                cached_data['cache_meta'] = cache_meta
                return jsonify(cached_data), 200
        
        # Check interaction count for routing
        # Normalize UUID format (remove dashes) since SQLite stores UUIDs without dashes
        normalized_user_id = str(user_id).replace('-', '')
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.user_id == normalized_user_id
        ).count()
        
        logger.info(f"Investor {user_id} has {interaction_count} interactions")
        
        # Smart routing based on interaction count (same as developer routing)
        if interaction_count < 5:
            logger.info(f"-> Using Content-Based (cold start: {interaction_count} interactions)")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
            results = rec_service.get_recommendations(
                user_id=user_id,
                use_case='investor_startup',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count < 20:
            logger.info(f"-> Using ALS (warm investor: {interaction_count} interactions)")
            if als_model:
                results = als_model.recommend(user_id, limit, filters)
                method_used = 'als'
                model_version = 'als_v1.0'
            elif two_tower_model:
                logger.info("  ALS unavailable, falling back to Two-Tower")
                results = two_tower_model.recommend(user_id, limit, filters)
                method_used = 'two_tower'
                model_version = 'two_tower_v1.0'
            else:
                logger.info("  No models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=user_id,
                    use_case='investor_startup',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        else:
            logger.info(f"-> Using Ensemble (hot investor: {interaction_count} interactions)")
            if ensemble_model:
                results = ensemble_model.recommend(user_id, limit, filters)
                method_used = 'ensemble'
                model_version = 'ensemble_v1.0'
            elif als_model:
                logger.info("  Ensemble unavailable, falling back to ALS")
                results = als_model.recommend(user_id, limit, filters)
                method_used = 'als'
                model_version = 'als_v1.0'
            elif two_tower_model:
                logger.info("  ALS/Ensemble unavailable, falling back to Two-Tower")
                results = two_tower_model.recommend(user_id, limit, filters)
                method_used = 'two_tower'
                model_version = 'two_tower_v1.0'
            else:
                logger.info("  No models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=user_id,
                    use_case='investor_startup',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply ranker to reorder recommendations
        results = apply_ranker(results, user_id, limit, method_used)
        
        # Convert 'startups' array to 'item_ids' if needed (for session service compatibility)
        # Session service expects 'item_ids' but ensemble returns 'startups'
        if 'startups' in results and 'item_ids' not in results:
            results['item_ids'] = [startup.get('id') for startup in results.get('startups', [])]
            # Also ensure scores/match_reasons use item_ids as keys
            if results.get('item_ids'):
                new_scores = {}
                new_match_reasons = {}
                scores = results.get('scores', {})
                match_reasons = results.get('match_reasons', {})
                
                for startup in results.get('startups', []):
                    startup_id = startup.get('id')
                    if startup_id:
                        # Try both normalized and original format
                        startup_id_str = str(startup_id)
                        startup_id_normalized = startup_id_str.replace('-', '')
                        
                        # Look up score/match_reasons
                        score = scores.get(startup_id_normalized) or scores.get(startup_id_str) or scores.get(startup_id)
                        match_reason = match_reasons.get(startup_id_normalized) or match_reasons.get(startup_id_str) or match_reasons.get(startup_id)
                        
                        if score is not None:
                            new_scores[startup_id] = score
                        if match_reason:
                            new_match_reasons[startup_id] = match_reason
                
                results['scores'] = new_scores if new_scores else scores
                results['match_reasons'] = new_match_reasons if new_match_reasons else match_reasons
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=user_id,
            use_case='investor_startup',
            method=method_used,
            recommendations=results,
            model_version=model_version
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['interaction_count'] = interaction_count
        response['method_used'] = method_used
        
        cache_meta = _set_personalized_cache_entry(namespace, user_id, params_hash, cache_indicator, response)
        response_payload = deepcopy(response)
        response_payload['cache_meta'] = cache_meta
        
        return jsonify(response_payload), 200
        
    except Exception as e:
        logger.error(f"Error in get_startups_for_investor: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        db.close()


@app.route('/api/recommendations/developers/for-startup/<startup_id>', methods=['GET'])
def get_developers_for_startup(startup_id):
    """
    Get developer recommendations for a startup (founder's view)
    Recommends developers for the positions opened in the startup
    
    Query params:
        limit: Number of results (default: 10)
        position_id: Filter by specific position (optional)
    
    Returns:
        List of recommended developers with scores and match reasons
    """
    db = SessionLocal()
    try:
        force_refresh = _should_bust_cache()
        cache_indicator = request.args.get('cache_indicator', 'default')
        requesting_user_id = request.args.get('requesting_user_id')
        
        # Validate startup_id
        if not startup_id or not isinstance(startup_id, str):
            return jsonify({'error': 'Invalid startup_id'}), 400
        
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 10))
            if limit < 1:
                limit = 10
            elif limit > 100:
                limit = 100  # Cap at 100 for performance
        except (ValueError, TypeError):
            limit = 10
        
        position_id = request.args.get('position_id', None)
        
        # Get startup owner as user_id for routing (robust to UUID format)
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
        startup = db.query(Startup).filter(Startup.id.in_(candidates)).first()
        if not startup:
            logger.info(f"Developer route: Startup not found for any id variants: {candidates}")
            return jsonify({'error': 'Startup not found'}), 404
        
        founder_id = str(startup.owner_id)
        user_cache_key = requesting_user_id or founder_id
        
        # Build filters (role: student for developers)
        filters = {'role': 'student'}
        if position_id:
            filters['position_id'] = position_id
        
        cache_params = {
            'limit': limit,
            'position_id': position_id or '',
            'startup_id': startup_id,
            'skills': request.args.get('skills')
        }
        params_hash = _make_params_fingerprint(cache_params)
        namespace = 'startup_developer'
        if not force_refresh:
            cached_entry = _get_personalized_cache_entry(namespace, user_cache_key, params_hash, cache_indicator)
            if cached_entry:
                cached_data, cache_meta = cached_entry
                logger.info(f"📦 Flask: Cache hit for {namespace} startup={startup_id} requester={user_cache_key}")
                cached_data['cache_meta'] = cache_meta
                return jsonify(cached_data), 200
        
        # Check interaction count for this startup (reverse direction)
        # Note: For reverse, we check how many users this startup has interacted with
        # Using UserInteraction as proxy (startup_id -> user_id interactions)
        # In production, this should check StartupInteraction table via API
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.startup_id == startup_id
        ).distinct(UserInteraction.user_id).count()
        
        logger.info(f"Startup {startup_id} has {interaction_count} unique user interactions")
        
        # Smart routing based on interaction count
        if interaction_count < 5:
            # Cold start: content-based
            logger.info(f"-> Using Content-Based (cold start: {interaction_count} interactions)")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
            results = rec_service.get_recommendations(
                user_id=startup_id,  # Pass startup_id as per service API
                use_case='startup_developer',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count >= 5 and interaction_count < 20:
            # Warm startups: Try Two-Tower Reverse, fallback to ALS Reverse
            logger.info(f"-> Using Two-Tower Reverse or ALS Reverse (warm startup: {interaction_count} interactions)")
            if two_tower_reverse_model:
                try:
                    results = two_tower_reverse_model.recommend(startup_id, limit, filters)
                    method_used = 'two_tower_reverse'
                    model_version = 'two_tower_reverse_v1.0'
                except Exception as e:
                    logger.warning(f"Two-Tower Reverse failed: {e}, falling back to ALS Reverse")
                    if als_reverse_model:
                        results = als_reverse_model.recommend(startup_id, limit, filters)
                        method_used = 'als_reverse'
                        model_version = 'als_reverse_v1.0'
                    else:
                        # Fallback to content-based
                        from services.recommendation_service import RecommendationService
                        rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                        results = rec_service.get_recommendations(
                            user_id=startup_id,
                            use_case='startup_developer',
                            limit=limit,
                            filters=filters
                        )
                        method_used = results.get('method_used', 'content_based')
                        model_version = 'content_based_v1.0'
            elif als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  No reverse models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=startup_id,
                    use_case='startup_developer',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        else:
            # Hot startups: Prefer Two-Tower Reverse, fallback to ALS Reverse
            logger.info(f"-> Using Two-Tower Reverse or ALS Reverse (hot startup: {interaction_count} interactions)")
            if two_tower_reverse_model:
                try:
                    results = two_tower_reverse_model.recommend(startup_id, limit, filters)
                    method_used = 'two_tower_reverse'
                    model_version = 'two_tower_reverse_v1.0'
                except Exception as e:
                    logger.warning(f"Two-Tower Reverse failed: {e}, falling back to ALS Reverse")
                    if als_reverse_model:
                        results = als_reverse_model.recommend(startup_id, limit, filters)
                        method_used = 'als_reverse'
                        model_version = 'als_reverse_v1.0'
                    else:
                        # Fallback to content-based
                        from services.recommendation_service import RecommendationService
                        rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                        results = rec_service.get_recommendations(
                            user_id=startup_id,
                            use_case='startup_developer',
                            limit=limit,
                            filters=filters
                        )
                        method_used = results.get('method_used', 'content_based')
                        model_version = 'content_based_v1.0'
            elif als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  No reverse models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=startup_id,
                    use_case='startup_developer',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply reverse ranker to reorder recommendations
        results = apply_ranker(results, startup_id, limit, method_used, use_reverse_ranker=True)
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=startup_id,  # Use startup_id for reverse use cases
            use_case='startup_developer',
            method=method_used,
            recommendations=results,
            model_version=model_version,
            startup_id=startup_id  # Pass startup_id for reverse
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['startup_id'] = startup_id
        
        cache_meta = _set_personalized_cache_entry(namespace, user_cache_key, params_hash, cache_indicator, response)
        response_payload = deepcopy(response)
        response_payload['cache_meta'] = cache_meta
        
        return jsonify(response_payload), 200
        
    except Exception as e:
        logger.error(f"Error in get_developers_for_startup: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        db.close()


@app.route('/api/recommendations/investors/for-startup/<startup_id>', methods=['GET'])
def get_investors_for_startup(startup_id):
    """
    Get investor recommendations for a startup (founder's view)
    Shows potential investors who might be interested in the startup
    
    Query params:
        limit: Number of results (default: 10)
    
    Returns:
        List of recommended investors with scores and match reasons
    """
    db = SessionLocal()
    try:
        force_refresh = _should_bust_cache()
        cache_indicator = request.args.get('cache_indicator', 'default')
        requesting_user_id = request.args.get('requesting_user_id')
        
        # Validate startup_id
        if not startup_id or not isinstance(startup_id, str):
            return jsonify({'error': 'Invalid startup_id'}), 400
        
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 10))
            if limit < 1:
                limit = 10
            elif limit > 100:
                limit = 100  # Cap at 100 for performance
        except (ValueError, TypeError):
            limit = 10
        
        # Get startup owner as user_id for routing (robust to UUID format)
        sid = str(startup_id)
        candidates = [sid]
        # If hyphenated, also try without hyphens; if 32-hex, also try hyphenated UUID form
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
        
        startup = db.query(Startup).filter(Startup.id.in_(candidates)).first()
        if not startup:
            logger.info(f"Investor route: Startup not found for any id variants: {candidates}")
            return jsonify({'error': 'Startup not found'}), 404
        
        founder_id = str(startup.owner_id)
        user_cache_key = requesting_user_id or founder_id
        
        # Build filters (role: investor)
        filters = {'role': 'investor'}
        
        cache_params = {
            'limit': limit,
            'startup_id': startup_id
        }
        params_hash = _make_params_fingerprint(cache_params)
        namespace = 'startup_investor'
        if not force_refresh:
            cached_entry = _get_personalized_cache_entry(namespace, user_cache_key, params_hash, cache_indicator)
            if cached_entry:
                cached_data, cache_meta = cached_entry
                logger.info(f"📦 Flask: Cache hit for {namespace} startup={startup_id} requester={user_cache_key}")
                cached_data['cache_meta'] = cache_meta
                return jsonify(cached_data), 200
        
        # Check interaction count for this startup (reverse direction)
        # Note: For reverse, we check how many users this startup has interacted with
        # Using UserInteraction as proxy (startup_id -> user_id interactions)
        # In production, this should check StartupInteraction table via API
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.startup_id == startup_id
        ).distinct(UserInteraction.user_id).count()
        
        logger.info(f"Startup {startup_id} has {interaction_count} unique user interactions")
        
        # Smart routing based on interaction count
        if interaction_count < 5:
            # Cold start: content-based
            logger.info(f"-> Using Content-Based (cold start: {interaction_count} interactions)")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
            results = rec_service.get_recommendations(
                user_id=startup_id,  # Pass startup_id as per service API
                use_case='startup_investor',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count >= 5 and interaction_count < 20:
            # Warm startups: Try Two-Tower Reverse, fallback to ALS Reverse
            logger.info(f"-> Using Two-Tower Reverse or ALS Reverse (warm startup: {interaction_count} interactions)")
            if two_tower_reverse_model:
                try:
                    results = two_tower_reverse_model.recommend(startup_id, limit, filters)
                    method_used = 'two_tower_reverse'
                    model_version = 'two_tower_reverse_v1.0'
                except Exception as e:
                    logger.warning(f"Two-Tower Reverse failed: {e}, falling back to ALS Reverse")
                    if als_reverse_model:
                        results = als_reverse_model.recommend(startup_id, limit, filters)
                        method_used = 'als_reverse'
                        model_version = 'als_reverse_v1.0'
                    else:
                        # Fallback to content-based
                        from services.recommendation_service import RecommendationService
                        rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                        results = rec_service.get_recommendations(
                            user_id=startup_id,
                            use_case='startup_investor',
                            limit=limit,
                            filters=filters
                        )
                        method_used = results.get('method_used', 'content_based')
                        model_version = 'content_based_v1.0'
            elif als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  No reverse models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=startup_id,
                    use_case='startup_investor',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        else:
            # Hot startups: Prefer Two-Tower Reverse, fallback to ALS Reverse
            logger.info(f"-> Using Two-Tower Reverse or ALS Reverse (hot startup: {interaction_count} interactions)")
            if two_tower_reverse_model:
                try:
                    results = two_tower_reverse_model.recommend(startup_id, limit, filters)
                    method_used = 'two_tower_reverse'
                    model_version = 'two_tower_reverse_v1.0'
                except Exception as e:
                    logger.warning(f"Two-Tower Reverse failed: {e}, falling back to ALS Reverse")
                    if als_reverse_model:
                        results = als_reverse_model.recommend(startup_id, limit, filters)
                        method_used = 'als_reverse'
                        model_version = 'als_reverse_v1.0'
                    else:
                        # Fallback to content-based
                        from services.recommendation_service import RecommendationService
                        rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                        results = rec_service.get_recommendations(
                            user_id=startup_id,
                            use_case='startup_investor',
                            limit=limit,
                            filters=filters
                        )
                        method_used = results.get('method_used', 'content_based')
                        model_version = 'content_based_v1.0'
            elif als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  No reverse models available, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=startup_id,
                    use_case='startup_investor',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply reverse ranker to reorder recommendations
        results = apply_ranker(results, startup_id, limit, method_used, use_reverse_ranker=True)
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=startup_id,  # Use startup_id for reverse use cases
            use_case='startup_investor',
            method=method_used,
            recommendations=results,
            model_version=model_version,
            startup_id=startup_id  # Pass startup_id for reverse
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['startup_id'] = startup_id
        
        cache_meta = _set_personalized_cache_entry(namespace, user_cache_key, params_hash, cache_indicator, response)
        response_payload = deepcopy(response)
        response_payload['cache_meta'] = cache_meta
        
        return jsonify(response_payload), 200
        
    except Exception as e:
        logger.error(f"Error in get_investors_for_startup: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        db.close()


# Cache for trending startups (very short TTL for near real-time updates)
_trending_cache = {
    'data': None,
    'timestamp': None,
    'params': None,
    'version': 0
}
TRENDING_CACHE_TTL = 15  # seconds
PERSONALIZED_CACHE_TTL = 30  # seconds
_personalized_cache = {}


def _truthy_flag(value) -> bool:
    return str(value).lower() in ('1', 'true', 'yes', 'force', 'refresh')


def _should_bust_cache() -> bool:
    param_flag = request.args.get('cache_bust')
    header_flag = request.headers.get('X-Cache-Bust')
    return (_truthy_flag(param_flag) if param_flag else False) or (_truthy_flag(header_flag) if header_flag else False)


def _make_params_fingerprint(params: dict) -> str:
    cleaned = {k: str(v) for k, v in sorted(params.items()) if v is not None}
    normalized = json.dumps(cleaned, separators=(',', ':'))
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]


def _get_personalized_cache_entry(namespace: str, user_key: str, params_hash: str, indicator: str):
    cache_key = (namespace, str(user_key), params_hash)
    entry = _personalized_cache.get(cache_key)
    if not entry:
        return None
    age = (datetime.now() - entry['timestamp']).total_seconds()
    if age >= entry['ttl']:
        return None
    if entry['indicator'] != indicator:
        return None
    return deepcopy(entry['data']), {
        'hit': True,
        'indicator': entry['indicator'],
        'cached_at': entry['timestamp'].isoformat(),
        'ttl_seconds': entry['ttl'] - age
    }


def _set_personalized_cache_entry(namespace: str, user_key: str, params_hash: str, indicator: str, data: dict, ttl: int = PERSONALIZED_CACHE_TTL):
    cache_key = (namespace, str(user_key), params_hash)
    now = datetime.now()
    _personalized_cache[cache_key] = {
        'data': deepcopy(data),
        'timestamp': now,
        'indicator': indicator,
        'ttl': ttl
    }
    return {
        'hit': False,
        'indicator': indicator,
        'cached_at': now.isoformat(),
        'ttl_seconds': ttl
    }


@app.route('/api/recommendations/trending/startups', methods=['GET'])
def get_trending_startups():
    """
    Get non-personalized trending/popular startups
    Available for all users (no authentication required)
    
    Query params:
        limit: Number of results (default: 50)
        sort_by: Sort by 'trending_score', 'popularity_score', 'velocity_score', 'views', 'created_at' (default: 'trending_score')
    
    Returns:
        List of trending startups with metrics
    """
    try:
        limit_param = request.args.get('limit', '50')
        sort_by_param = request.args.get('sort_by', 'trending_score')
        cache_key = f"{limit_param}_{sort_by_param}"
        force_refresh = _should_bust_cache()
        
        # Check cache
        if not force_refresh and _trending_cache['data'] and _trending_cache['timestamp'] and _trending_cache['params'] == cache_key:
            time_elapsed = (datetime.now() - _trending_cache['timestamp']).total_seconds()
            if time_elapsed < TRENDING_CACHE_TTL:
                logger.info(f"📦 Flask: Returning cached trending data (age: {time_elapsed:.1f}s)")
                cached_payload = deepcopy(_trending_cache['data'])
                cached_payload['cache_meta'] = {
                    'hit': True,
                    'indicator': f"trending::{_trending_cache['version']}",
                    'cached_at': _trending_cache['timestamp'].isoformat(),
                    'ttl_seconds': max(TRENDING_CACHE_TTL - time_elapsed, 0)
                }
                return jsonify(cached_payload), 200
            else:
                logger.info(f"⏰ Flask: Cache expired (age: {time_elapsed:.1f}s), fetching fresh data")
        
        # Proceed with normal logic if cache miss
        logger.info(f"🔍 Flask: Cache miss, fetching from database")
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 50))
            if limit < 1:
                limit = 50
            elif limit > 200:
                limit = 200  # Cap at 200 for trending
        except (ValueError, TypeError):
            limit = 50
        
        sort_by = request.args.get('sort_by', 'trending_score')
        valid_sort_options = ['trending_score', 'popularity_score', 'velocity_score', 'views', 'created_at']
        if sort_by not in valid_sort_options:
            sort_by = 'trending_score'
        
        # Query trending startups from database
        db = SessionLocal()
        try:
            from database.models import Startup, StartupTrendingMetrics
            from sqlalchemy import desc
            
            # Get trending metrics sorted by requested field
            metrics_query = db.query(StartupTrendingMetrics)
            
            # Sort by requested field
            if sort_by == 'trending_score':
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.trending_score))
            elif sort_by == 'popularity_score':
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.popularity_score))
            elif sort_by == 'velocity_score':
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.velocity_score))
            elif sort_by == 'views':
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.view_count_7d))
            elif sort_by == 'created_at':
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.computed_at))
            
            total = metrics_query.count()
            trending_metrics = metrics_query.limit(limit).all()
            
            # Get latest computed_at timestamp
            computed_at = trending_metrics[0].computed_at.isoformat() if trending_metrics else None
            
            # Format response
            result = []
            for metrics in trending_metrics:
                startup = db.query(Startup).filter(
                    Startup.id == metrics.startup_id
                ).first()
                
                if startup and startup.status == 'active':
                    startup_data = {
                        'id': str(startup.id),  # Explicitly convert UUID to string for JSON serialization
                        'title': startup.title,
                        'description': startup.description,
                        'field': startup.field,
                        'website_url': startup.website_url,
                        'type': startup.type,
                        'category': startup.category,
                        'views': startup.views,
                        'trending_score': float(metrics.trending_score),
                        'popularity_score': float(metrics.popularity_score),
                        'velocity_score': float(metrics.velocity_score),
                        'view_count_24h': metrics.view_count_24h,
                        'view_count_7d': metrics.view_count_7d,
                        'application_count_7d': metrics.application_count_7d,
                        'favorite_count_7d': metrics.favorite_count_7d,
                        'interest_count_7d': metrics.interest_count_7d,
                        'active_positions_count': metrics.active_positions_count,
                    }
                    
                    # Verify ID is present
                    if not startup_data.get('id'):
                        logger.warning(f"⚠️ Flask: Startup missing ID! startup_id from metrics: {metrics.startup_id}")
                    
                    result.append(startup_data)
            
            logger.info(f"📊 Flask: Returning {len(result)} startups")
            if result:
                logger.info(f"📊 Flask: Sample startup ID: {result[0].get('id')}")
            
            response_data = {
                'startups': result,
                'total': len(result),
                'limit': limit,
                'sort_by': sort_by,
                'computed_at': computed_at
            }
            
            # Update cache
            _trending_cache['data'] = deepcopy(response_data)
            _trending_cache['timestamp'] = datetime.now()
            _trending_cache['params'] = cache_key
            _trending_cache['version'] += 1
            logger.info(f"💾 Flask: Cached trending data for {TRENDING_CACHE_TTL}s")
            
            response_payload = deepcopy(response_data)
            response_payload['cache_meta'] = {
                'hit': False,
                'indicator': f"trending::{_trending_cache['version']}",
                'cached_at': _trending_cache['timestamp'].isoformat(),
                'ttl_seconds': TRENDING_CACHE_TTL
            }
            
            return jsonify(response_payload), 200
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error in get_trending_startups: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics_dashboard():
    """Serve the metrics dashboard HTML page"""
    try:
        from flask import render_template
        return render_template('metrics_dashboard.html')
    except Exception as e:
        logger.error(f"Error serving metrics dashboard: {e}")
        return f"Error loading dashboard: {e}", 500


@app.route('/api/recommendations/metrics', methods=['GET'])
def get_recommendation_metrics():
    """
    Get evaluation metrics and statistics for the recommendation system
    
    Query params:
        use_case: Filter by use case - 'developer_startup', 'founder_developer', 'founder_startup', 'investor_startup' (optional)
        model_type: Filter by model type - 'content_based', 'als', 'two_tower', 'ranker' (optional)
    
    Returns:
        Metrics and evaluation data for recommendation models
    """
    try:
        use_case = request.args.get('use_case', None)
        model_type = request.args.get('model_type', None)
        
        # Validate use_case if provided
        if use_case and use_case not in ['developer_startup', 'founder_developer', 'founder_startup', 'investor_startup']:
            use_case = None
        
        # Validate model_type if provided
        if model_type and model_type not in ['content_based', 'als', 'two_tower', 'ranker']:
            model_type = None
        
        # Load metrics from evaluation file
        metrics_file = MODELS_DIR / "all_models_metrics.json"
        all_metrics_data = {}
        
        if metrics_file.exists():
            try:
                import json
                with open(metrics_file, 'r') as f:
                    all_metrics_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metrics file: {e}")
        
        # Filter models based on query params
        filtered_models = []
        for model_key, model_data in all_metrics_data.items():
            # Skip if only training_history exists (incomplete evaluation)
            if 'metrics' not in model_data:
                continue
            
            # Apply filters
            if use_case and model_data.get('use_case') != use_case:
                continue
            if model_type and model_data.get('model_type') != model_type:
                continue
            
            filtered_models.append({
                'name': model_data.get('model_name', model_key),
                'type': model_data.get('model_type', 'unknown'),
                'use_case': model_data.get('use_case', 'unknown'),
                'is_reverse': model_data.get('is_reverse', False),
                'evaluation_date': model_data.get('evaluation_date'),
                'metrics': model_data.get('metrics', {}),
                'num_test_entities': model_data.get('num_test_entities', 0),
                'coverage_details': model_data.get('coverage_details', {})
            })
        
        # Calculate overall metrics
        if filtered_models:
            total_models = len(filtered_models)
            active_models = total_models  # All evaluated models are considered active
            
            # Average metrics across all models
            precision_values = []
            recall_values = []
            ndcg_values = []
            
            for model in filtered_models:
                metrics = model.get('metrics', {})
                # Get metrics at K=10 for averaging
                if 'precision@10' in metrics:
                    precision_values.append(metrics['precision@10'])
                if 'recall@10' in metrics:
                    recall_values.append(metrics['recall@10'])
                if 'ndcg@10' in metrics:
                    ndcg_values.append(metrics['ndcg@10'])
            
            overall_metrics = {
                'total_models': total_models,
                'active_models': active_models,
                'average_precision': float(np.mean(precision_values)) if precision_values else 0.0,
                'average_recall': float(np.mean(recall_values)) if recall_values else 0.0,
                'average_ndcg': float(np.mean(ndcg_values)) if ndcg_values else 0.0
            }
        else:
            overall_metrics = {
                'total_models': 0,
                'active_models': 0,
                'average_precision': 0.0,
                'average_recall': 0.0,
                'average_ndcg': 0.0
            }
        
        return jsonify({
            'models': filtered_models,
            'overall_metrics': overall_metrics,
            'filters': {
                'use_case': use_case,
                'model_type': model_type
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in get_recommendation_metrics: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    logger.info(f"Starting Flask Recommendation Service on {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Debug mode: {FLASK_DEBUG}")
    
    # Check database connection on startup
    if check_db_connection():
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed - service may not work correctly")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

