"""
Database connection setup for Flask Recommendation Service
Connects to Django's SQLite database
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATABASE_URL,
    DATABASE_PATH,
    SQLALCHEMY_ECHO,
    SQLALCHEMY_POOL_SIZE,
    SQLALCHEMY_MAX_OVERFLOW,
    SQLALCHEMY_POOL_TIMEOUT,
    SQLALCHEMY_POOL_RECYCLE
)
from utils.logger import get_logger

logger = get_logger(__name__)

# SQLite-specific engine configuration
# Use StaticPool for SQLite to handle concurrent access better
engine_kwargs = {
    'echo': SQLALCHEMY_ECHO,
    'poolclass': StaticPool,
    'connect_args': {
        'check_same_thread': False,  # Allow multi-threaded access
        'timeout': 20,  # Connection timeout
    },
    'pool_pre_ping': True,  # Verify connections before using
}

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    Dependency function to get database session
    Use this in route handlers for database access
    
    Example:
        @app.route('/users')
        def get_users():
            db = get_db()
            users = db.query(User).all()
            db.close()
            return jsonify([u.to_dict() for u in users])
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Check if database connection is working
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # Database connection logged at app startup
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# SQLite optimizations
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Set SQLite pragmas for better performance
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints
    cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
    cursor.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and speed
    cursor.close()


