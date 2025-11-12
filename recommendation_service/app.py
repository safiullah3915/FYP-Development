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
from database.models import User, Startup
from utils.logger import get_logger

logger = get_logger(__name__)

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, origins=CORS_ORIGINS)


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
    limit = int(request.args.get('limit', 10))
    startup_type = request.args.get('type', None)
    
    # TODO: Implement recommendation logic
    recommendations = []  # Placeholder
    
    # Generate session data
    session_data = generate_recommendation_session(
        user_id=user_id,
        use_case='developer_startup',
        method='content_based',  # Will be dynamic
        recommendations=recommendations
    )
    
    return jsonify({
        **session_data,  # Include all session fields
        'recommendations': session_data['recommendations'],
        'total': len(recommendations),
        'limit': limit,
        'filters': {
            'type': startup_type
        }
    }), 200


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
    limit = int(request.args.get('limit', 10))
    category = request.args.get('category', None)
    
    # TODO: Implement recommendation logic
    recommendations = []  # Placeholder
    
    # Generate session data
    session_data = generate_recommendation_session(
        user_id=user_id,
        use_case='investor_startup',
        method='content_based',
        recommendations=recommendations
    )
    
    return jsonify({
        **session_data,  # Include all session fields
        'recommendations': session_data['recommendations'],
        'total': len(recommendations),
        'limit': limit,
        'filters': {
            'type': 'marketplace',  # Only marketplace startups for investors
            'category': category
        }
    }), 200


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
    limit = int(request.args.get('limit', 10))
    position_id = request.args.get('position_id', None)
    
    # TODO: Implement recommendation logic
    recommendations = []  # Placeholder
    
    # Generate session data
    session_data = generate_recommendation_session(
        user_id=startup_id,  # For founder use case, startup_id represents the founder
        use_case='founder_developer',
        method='content_based',
        recommendations=recommendations
    )
    
    return jsonify({
        **session_data,  # Include all session fields
        'startup_id': startup_id,
        'recommendations': session_data['recommendations'],
        'total': len(recommendations),
        'limit': limit,
        'filters': {
            'position_id': position_id
        }
    }), 200


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
    limit = int(request.args.get('limit', 10))
    
    # TODO: Implement recommendation logic
    recommendations = []  # Placeholder
    
    # Generate session data
    session_data = generate_recommendation_session(
        user_id=startup_id,  # For founder use case, startup_id represents the founder
        use_case='founder_startup',
        method='content_based',
        recommendations=recommendations
    )
    
    return jsonify({
        **session_data,  # Include all session fields
        'startup_id': startup_id,
        'recommendations': session_data['recommendations'],
        'total': len(recommendations),
        'limit': limit
    }), 200


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
    limit = int(request.args.get('limit', 50))
    sort_by = request.args.get('sort_by', 'trending_score')
    
    # TODO: Implement trending logic using StartupTrendingMetrics
    # For now, return empty structure
    return jsonify({
        'startups': [],
        'total': 0,
        'limit': limit,
        'sort_by': sort_by,
        'computed_at': None
    }), 200


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
    use_case = request.args.get('use_case', None)
    model_type = request.args.get('model_type', None)
    
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

