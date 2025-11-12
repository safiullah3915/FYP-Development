"""
Configuration management for Flask Recommendation Service
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory (parent of recommendation_service)
BASE_DIR = Path(__file__).resolve().parent.parent

# Database Configuration
# Point to Django's SQLite database
DATABASE_PATH = BASE_DIR / 'backend' / 'db.sqlite3'

# Flask Configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Database Connection String
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# SQLAlchemy Configuration
SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'False').lower() == 'true'
SQLALCHEMY_POOL_SIZE = int(os.getenv('SQLALCHEMY_POOL_SIZE', 5))
SQLALCHEMY_MAX_OVERFLOW = int(os.getenv('SQLALCHEMY_MAX_OVERFLOW', 10))
SQLALCHEMY_POOL_TIMEOUT = int(os.getenv('SQLALCHEMY_POOL_TIMEOUT', 20))
SQLALCHEMY_POOL_RECYCLE = int(os.getenv('SQLALCHEMY_POOL_RECYCLE', 3600))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'recommendation_service.log')

# CORS Configuration
CORS_ORIGINS = os.getenv(
    'CORS_ORIGINS',
    'http://localhost:5173,http://localhost:8000'
).split(',')


