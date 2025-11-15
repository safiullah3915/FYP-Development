"""
Flask Recommendation Service Application
Minimal Flask app for testing database connection and models
"""
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, CORS_ORIGINS
from database.connection import check_db_connection, SessionLocal
from database.models import User, Startup, UserInteraction
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

# Load Two-Tower Model
try:
    from inference_two_tower import TwoTowerInference
    model_path = Path(__file__).parent / "models" / "two_tower_v1_best.pth"
    
    if model_path.exists():
        two_tower_model = TwoTowerInference(str(model_path))
        logger.info("✓ Two-Tower model loaded successfully!")
    else:
        logger.warning(f"Two-Tower model not found at {model_path}")
except Exception as e:
    logger.error(f"Failed to load Two-Tower model: {e}")

# Load ALS Model (Forward: User → Startup)
try:
    from inference_als import ALSInference
    als_path = Path(__file__).parent / "models" / "als_v1.pkl"
    
    if als_path.exists():
        als_model = ALSInference(str(als_path))
        logger.info("✓ ALS Forward model loaded successfully!")
    else:
        logger.warning(f"ALS Forward model not found at {als_path}")
except Exception as e:
    logger.error(f"Failed to load ALS Forward model: {e}")

# Load ALS Reverse Model (Reverse: Startup → User)
als_reverse_model = None
try:
    from inference_als_reverse import ALSReverseInference
    als_reverse_path = Path(__file__).parent / "models" / "als_reverse_v1.pkl"
    
    if als_reverse_path.exists():
        als_reverse_model = ALSReverseInference(str(als_reverse_path))
        logger.info("✓ ALS Reverse model loaded successfully!")
        logger.info("  -> Will be used for Founder → Developer/Investor recommendations")
    else:
        logger.warning(f"ALS Reverse model not found at {als_reverse_path}")
        logger.info("  -> Founder use cases will use content-based only")
except Exception as e:
    logger.error(f"Failed to load ALS Reverse model: {e}")

# Load Ensemble Model (if both base models available)
try:
    if two_tower_model and als_model:
        from inference_ensemble import EnsembleInference
        ensemble_model = EnsembleInference(
            als_model_path=str(Path(__file__).parent / "models" / "als_v1.pkl"),
            two_tower_model_path=str(Path(__file__).parent / "models" / "two_tower_v1_best.pth"),
            als_weight=0.6
        )
        logger.info("✓ Ensemble model initialized successfully!")
        logger.info("  -> Routing: cold start(<5) -> content, warm(5-19) -> ALS, hot(20+) -> ensemble")
    else:
        logger.warning("Ensemble not initialized (requires both ALS and Two-Tower)")
        if two_tower_model:
            logger.info("  -> Will use: content-based + Two-Tower")
        elif als_model:
            logger.info("  -> Will use: content-based + ALS")
        else:
            logger.info("  -> Will use: content-based only")
except Exception as e:
    logger.error(f"Failed to initialize ensemble: {e}")

# Load Ranker Model (reranks recommendations for better quality)
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
        logger.info("✓ Ranker model loaded successfully!")
        logger.info("  -> Will rerank all personalized recommendations")
    else:
        logger.info("Ranker model not found, using rule-based ranker")
        ranker_model = NeuralRanker(use_rule_based=True)
except Exception as e:
    logger.warning(f"Could not load ranker: {e}")
    logger.info("  -> Recommendations will work without ranker")
    ranker_model = None


def apply_ranker(results, user_id, limit, method_used):
    """
    Apply ranker to reorder recommendations
    
    Args:
        results: Dict with 'startups' or similar key containing recommendations
        user_id: User ID for context
        limit: Final number of items to return
        method_used: Which recommendation method was used
    
    Returns:
        Updated results dict with reranked items
    """
    if not ranker_model:
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
        reranked = ranker_model.rank(
            candidates=candidates,
            user_id=user_id,
            already_ranked=[]
        )
        
        # Update results
        results[candidates_key] = reranked[:limit]
        results['reranked'] = True
        
        logger.info(f"Reranked {len(candidates)} candidates to {len(results[candidates_key])} items")
        
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

def generate_recommendation_session(
    user_id: str,
    use_case: str,
    method: str,
    recommendations: list,
    model_version: str = None
) -> dict:
    """
    Generate recommendation session data
    This will be stored in Django via API call
    Abstraction allows future direct DB writes or async processing
    """
    import uuid
    from datetime import datetime, timedelta
    
    session_id = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(hours=24)
    
    # Format recommendations with rank
    recommendations_with_rank = []
    for idx, rec in enumerate(recommendations):
        if isinstance(rec, dict):
            rec_data = {
                'startup_id': rec.get('startup_id') or rec.get('id'),
                'rank': idx + 1,
                'score': rec.get('score', 0.0),
                'match_reasons': rec.get('match_reasons', [])
            }
        else:
            # If rec is just startup_id
            rec_data = {
                'startup_id': str(rec),
                'rank': idx + 1,
                'score': 0.0,
                'match_reasons': []
            }
        recommendations_with_rank.append(rec_data)
    
    return {
        'recommendation_session_id': session_id,
        'user_id': user_id,
        'use_case': use_case,
        'method': method,
        'model_version': model_version or '',
        'recommendations': recommendations_with_rank,
        'created_at': datetime.now().isoformat(),
        'expires_at': expires_at.isoformat()
    }


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
        
        # Validate and sanitize limit
        try:
            limit = int(request.args.get('limit', 10))
            if limit < 1:
                limit = 10
            elif limit > 100:
                limit = 100  # Cap at 100 for performance
        except (ValueError, TypeError):
            limit = 10
        
        startup_type = request.args.get('type', None)
        if startup_type and startup_type not in ['marketplace', 'collaboration']:
            startup_type = None
        
        # Build filters
        filters = {}
        if startup_type:
            filters['type'] = startup_type
        
        # Determine if we should force open positions only (default for collaboration)
        require_open_positions_param = request.args.get('require_open_positions')
        require_open_positions = False
        if require_open_positions_param is not None:
            require_open_positions = require_open_positions_param.lower() in ['1', 'true', 'yes']
        elif startup_type == 'collaboration':
            require_open_positions = True
        
        if require_open_positions:
            filters['require_open_positions'] = True
        
        # Check interaction count for routing
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).count()
        
        logger.info(f"User {user_id} has {interaction_count} interactions")
        
        # Smart routing based on interaction count
        if interaction_count < 5:
            # Cold start: content-based
            logger.info(f"-> Using Content-Based (cold start: {interaction_count} interactions)")
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
            logger.info(f"-> Using ALS (warm user: {interaction_count} interactions)")
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
            logger.info(f"-> Using Ensemble (hot user: {interaction_count} interactions)")
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
        
        return jsonify(response), 200
        
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
        
        # Check interaction count for routing
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
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
        
        return jsonify(response), 200
        
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
        
        # Get startup owner as user_id for routing
        startup = db.query(Startup).filter(Startup.id == startup_id).first()
        if not startup:
            return jsonify({'error': 'Startup not found'}), 404
        
        founder_id = str(startup.owner_id)
        
        # Build filters (role: student for developers)
        filters = {'role': 'student'}
        if position_id:
            filters['position_id'] = position_id
        
        # Check interaction count for this startup (reverse direction)
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
                user_id=founder_id,
                use_case='founder_developer',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count >= 5:
            # Warm/Hot startups: ALS Reverse
            logger.info(f"-> Using ALS Reverse (warm/hot startup: {interaction_count} interactions)")
            if als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  ALS Reverse unavailable, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=founder_id,
                    use_case='founder_developer',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply ranker to reorder recommendations
        results = apply_ranker(results, founder_id, limit, method_used)
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=founder_id,
            use_case='founder_developer',
            method=method_used,
            recommendations=results,
            model_version=model_version
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['startup_id'] = startup_id
        
        return jsonify(response), 200
        
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
        
        # Get startup owner as user_id for routing
        startup = db.query(Startup).filter(Startup.id == startup_id).first()
        if not startup:
            return jsonify({'error': 'Startup not found'}), 404
        
        founder_id = str(startup.owner_id)
        
        # Build filters (role: investor)
        filters = {'role': 'investor'}
        
        # Check interaction count for this startup (reverse direction)
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
                user_id=founder_id,
                use_case='founder_investor',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        elif interaction_count >= 5:
            # Warm/Hot startups: ALS Reverse
            logger.info(f"-> Using ALS Reverse (warm/hot startup: {interaction_count} interactions)")
            if als_reverse_model:
                results = als_reverse_model.recommend(startup_id, limit, filters)
                method_used = 'als_reverse'
                model_version = 'als_reverse_v1.0'
            else:
                logger.info("  ALS Reverse unavailable, falling back to content-based")
                from services.recommendation_service import RecommendationService
                rec_service = RecommendationService(db, enable_two_tower=False, enable_als=False, enable_ensemble=False)
                results = rec_service.get_recommendations(
                    user_id=founder_id,
                    use_case='founder_investor',
                    limit=limit,
                    filters=filters
                )
                method_used = results.get('method_used', 'content_based')
                model_version = 'content_based_v1.0'
        
        # Apply ranker to reorder recommendations
        results = apply_ranker(results, founder_id, limit, method_used)
        
        # Create session data
        from services.session_service import SessionService
        session_service = SessionService()
        
        session_data = session_service.create_session_data(
            user_id=founder_id,
            use_case='founder_investor',
            method=method_used,
            recommendations=results,
            model_version=model_version
        )
        
        # Format for API response
        response = session_service.format_for_api_response(session_data, results)
        response['startup_id'] = startup_id
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in get_investors_for_startup: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        db.close()


# Cache for trending startups (5-minute TTL)
_trending_cache = {
    'data': None,
    'timestamp': None,
    'params': None
}
TRENDING_CACHE_TTL = 300  # 5 minutes in seconds


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
        from datetime import datetime
        
        # Get request params
        limit_param = request.args.get('limit', '50')
        sort_by_param = request.args.get('sort_by', 'trending_score')
        cache_key = f"{limit_param}_{sort_by_param}"
        
        # Check cache
        if _trending_cache['data'] and _trending_cache['timestamp'] and _trending_cache['params'] == cache_key:
            time_elapsed = (datetime.now() - _trending_cache['timestamp']).total_seconds()
            if time_elapsed < TRENDING_CACHE_TTL:
                logger.info(f"📦 Flask: Returning cached trending data (age: {time_elapsed:.1f}s)")
                return jsonify(_trending_cache['data']), 200
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
                metrics_query = metrics_query.order_by(desc(StartupTrendingMetrics.total_views))
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
            _trending_cache['data'] = response_data
            _trending_cache['timestamp'] = datetime.now()
            _trending_cache['params'] = cache_key
            logger.info(f"💾 Flask: Cached trending data for {TRENDING_CACHE_TTL}s")
            
            return jsonify(response_data), 200
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error in get_trending_startups: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


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
        
        # TODO: Implement metrics retrieval from RecommendationModel table
        # For now, return empty structure
        return jsonify({
            'models': [],
            'overall_metrics': {
                'total_models': 0,
                'active_models': 0,
                'average_precision': 0.0,
                'average_recall': 0.0,
                'average_ndcg': 0.0
            },
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

