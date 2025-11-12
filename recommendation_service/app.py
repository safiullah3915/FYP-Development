"""
Flask Recommendation Service Application
Minimal Flask app for testing database connection and models
"""
import sys
from pathlib import Path
from flask import Flask, jsonify
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

