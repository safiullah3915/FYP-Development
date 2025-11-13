"""
Database module for Flask Recommendation Service
"""
from .connection import engine, SessionLocal, get_db, check_db_connection
from .models import (
    User, UserProfile, UserOnboardingPreferences,
    Startup, StartupTag, Position,
    Application, Notification, Favorite, Interest,
    Conversation, Message, FileUpload,
    UserInteraction, StartupTrendingMetrics, RecommendationModel
)

__all__ = [
    'engine',
    'SessionLocal',
    'get_db',
    'check_db_connection',
    'User',
    'UserProfile',
    'UserOnboardingPreferences',
    'Startup',
    'StartupTag',
    'Position',
    'Application',
    'Notification',
    'Favorite',
    'Interest',
    'Conversation',
    'Message',
    'FileUpload',
    'UserInteraction',
    'StartupTrendingMetrics',
    'RecommendationModel',
]


