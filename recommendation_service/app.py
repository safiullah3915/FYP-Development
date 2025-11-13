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
# Initialize Two-Tower Model
# ============================================================================
two_tower_model = None
try:
    from inference_two_tower import TwoTowerInference
    model_path = Path(__file__).parent / "models" / "two_tower_v1_best.pth"
    
    if model_path.exists():
        two_tower_model = TwoTowerInference(str(model_path))
        logger.info("✓ Two-Tower model loaded successfully!")
    else:
        logger.warning(f"Two-Tower model not found at {model_path}")
        logger.warning("  → Will use content-based recommendations only")
except Exception as e:
    logger.error(f"Failed to load Two-Tower model: {e}")
    logger.warning("  → Will use content-based recommendations only")


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
        
        # Check interaction count for routing
        interaction_count = db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).count()
        
        logger.info(f"User {user_id} has {interaction_count} interactions")
        
        # Use Two-Tower for warm/hot users, Content-Based for cold start
        if two_tower_model and interaction_count >= 5:
            logger.info(f"→ Using Two-Tower model (warm/hot user)")
            results = two_tower_model.recommend(user_id, limit, filters)
            method_used = 'two_tower'
            model_version = 'two_tower_v1.0'
        else:
            logger.info(f"→ Using Content-Based (cold start: {interaction_count} interactions)")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False)
            results = rec_service.get_recommendations(
                user_id=user_id,
                use_case='developer_startup',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        
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
        
        # Use Two-Tower for warm/hot users, Content-Based for cold start
        if two_tower_model and interaction_count >= 5:
            logger.info(f"→ Using Two-Tower model (warm/hot investor)")
            results = two_tower_model.recommend(user_id, limit, filters)
            method_used = 'two_tower'
            model_version = 'two_tower_v1.0'
        else:
            logger.info(f"→ Using Content-Based (cold start: {interaction_count} interactions)")
            from services.recommendation_service import RecommendationService
            rec_service = RecommendationService(db, enable_two_tower=False)
            results = rec_service.get_recommendations(
                user_id=user_id,
                use_case='investor_startup',
                limit=limit,
                filters=filters
            )
            method_used = results.get('method_used', 'content_based')
            model_version = 'content_based_v1.0'
        
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
        
        # Build filters
        filters = {'startup_id': startup_id}
        if position_id:
            filters['position_id'] = position_id
        
        # Get recommendations
        from services.recommendation_service import RecommendationService
        from services.session_service import SessionService
        
        rec_service = RecommendationService(db)
        session_service = SessionService()
        
        results = rec_service.get_recommendations(
            user_id=founder_id,
            use_case='founder_developer',
            limit=limit,
            filters=filters
        )
        
        # Create session data
        session_data = session_service.create_session_data(
            user_id=founder_id,
            use_case='founder_developer',
            method=results.get('method_used', 'content_based'),
            recommendations=results,
            model_version='content_based_v1.0'
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
        
        # Build filters
        filters = {'startup_id': startup_id}
        
        # Get recommendations
        from services.recommendation_service import RecommendationService
        from services.session_service import SessionService
        
        rec_service = RecommendationService(db)
        session_service = SessionService()
        
        results = rec_service.get_recommendations(
            user_id=founder_id,
            use_case='founder_investor',
            limit=limit,
            filters=filters
        )
        
        # Create session data
        session_data = session_service.create_session_data(
            user_id=founder_id,
            use_case='founder_investor',
            method=results.get('method_used', 'content_based'),
            recommendations=results,
            model_version='content_based_v1.0'
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
        
        # TODO: Implement trending logic using StartupTrendingMetrics
        # For now, return empty structure
        return jsonify({
            'startups': [],
            'total': 0,
            'limit': limit,
            'sort_by': sort_by,
            'computed_at': None
        }), 200
    except Exception as e:
        logger.error(f"Error in get_trending_startups: {e}")
        return jsonify({'error': 'Internal server error'}), 500


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

