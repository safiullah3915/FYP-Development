"""
SQLAlchemy models matching Django database schema exactly
All models correspond to Django tables in the startup marketplace platform
"""
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import json
from typing import Optional, List, Dict, Any

Base = declarative_base()


# Helper function to parse JSON fields
def parse_json_field(value: Optional[str]) -> Any:
    """Parse JSON string field to Python object"""
    if value is None or value == '':
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


# ============================================================================
# CORE MODELS
# ============================================================================

class User(Base):
    """User model - matches Django users table"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)  # UUID stored as string
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_superuser = Column(Boolean, default=False)
    is_staff = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    role = Column(String(32), default='entrepreneur', index=True)  # entrepreneur, student, investor
    phone_number = Column(String(20), nullable=True)
    last_login = Column(DateTime, nullable=True)
    date_joined = Column(DateTime, nullable=False)
    
    # Embedding fields for recommendation system
    profile_embedding = Column(Text, nullable=True)  # JSON string
    embedding_model = Column(String(50), default='all-MiniLM-L6-v2')
    embedding_version = Column(Integer, default=1)
    embedding_updated_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    startups = relationship("Startup", back_populates="owner_rel", foreign_keys="Startup.owner_id")
    applications = relationship("Application", back_populates="applicant_rel")
    notifications = relationship("Notification", back_populates="user_rel")
    favorites = relationship("Favorite", back_populates="user_rel")
    interests = relationship("Interest", back_populates="user_rel")
    uploads = relationship("FileUpload", back_populates="user_rel")
    sent_messages = relationship("Message", back_populates="sender_rel")
    interactions = relationship("UserInteraction", back_populates="user_rel")
    profile = relationship("UserProfile", back_populates="user_rel", uselist=False)
    onboarding_preferences = relationship("UserOnboardingPreferences", back_populates="user_rel", uselist=False)
    
    def get_embedding(self) -> Optional[List[float]]:
        """Parse profile_embedding JSON string to list"""
        return parse_json_field(self.profile_embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'role': self.role,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
        }


class UserProfile(Base):
    """UserProfile model - matches Django user_profiles table"""
    __tablename__ = 'user_profiles'
    
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    bio = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)
    website = Column(String, nullable=True)
    profile_picture = Column(String, nullable=True)  # File path
    is_public = Column(Boolean, default=False)
    selected_regions = Column(Text, nullable=True)  # JSON string
    skills = Column(Text, nullable=True)  # JSON string
    experience = Column(Text, nullable=True)  # JSON string
    references = Column(Text, nullable=True)  # JSON string
    
    # Preference fields for recommendation system
    onboarding_completed = Column(Boolean, default=False)
    preferred_work_modes = Column(Text, nullable=True)  # JSON string
    preferred_compensation_types = Column(Text, nullable=True)  # JSON string
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="profile")
    
    def get_skills(self) -> List[str]:
        """Parse skills JSON field"""
        return parse_json_field(self.skills) or []
    
    def get_experience(self) -> List[Dict]:
        """Parse experience JSON field"""
        return parse_json_field(self.experience) or []
    
    def get_preferred_work_modes(self) -> List[str]:
        """Parse preferred_work_modes JSON field"""
        return parse_json_field(self.preferred_work_modes) or []


class UserOnboardingPreferences(Base):
    """UserOnboardingPreferences model - matches Django user_onboarding_preferences table"""
    __tablename__ = 'user_onboarding_preferences'
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False, index=True)
    selected_categories = Column(Text, nullable=True)  # JSON string
    selected_fields = Column(Text, nullable=True)  # JSON string
    selected_tags = Column(Text, nullable=True)  # JSON string
    preferred_startup_stages = Column(Text, nullable=True)  # JSON string
    preferred_engagement_types = Column(Text, nullable=True)  # JSON string
    preferred_skills = Column(Text, nullable=True)  # JSON string
    investor_profile = Column(Text, nullable=True)  # JSON string
    onboarding_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="onboarding_preferences")
    
    def get_selected_categories(self) -> List[str]:
        """Parse selected_categories JSON field"""
        return parse_json_field(self.selected_categories) or []
    
    def get_selected_tags(self) -> List[str]:
        """Parse selected_tags JSON field"""
        return parse_json_field(self.selected_tags) or []
    
    def get_investor_profile(self) -> Dict[str, Any]:
        """Parse investor_profile JSON field"""
        return parse_json_field(self.investor_profile) or {}


# ============================================================================
# STARTUP MODELS
# ============================================================================

class Startup(Base):
    """Startup model - matches Django startups table"""
    __tablename__ = 'startups'
    
    id = Column(String, primary_key=True)  # UUID
    owner_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    role_title = Column(String(100), nullable=True)
    description = Column(Text, nullable=False)
    field = Column(String(100), nullable=False)
    website_url = Column(String, nullable=True)
    stages = Column(Text, nullable=True)  # JSON string
    
    # Financial fields (for marketplace type)
    revenue = Column(String(50), nullable=True)
    profit = Column(String(50), nullable=True)
    asking_price = Column(String(50), nullable=True)
    ttm_revenue = Column(String(50), nullable=True)
    ttm_profit = Column(String(50), nullable=True)
    last_month_revenue = Column(String(50), nullable=True)
    last_month_profit = Column(String(50), nullable=True)
    
    # Collaboration fields
    earn_through = Column(String(50), nullable=True)
    phase = Column(String(50), nullable=True)
    team_size = Column(String(50), nullable=True)
    
    type = Column(String(20), default='marketplace', index=True)  # marketplace, collaboration
    category = Column(String(50), default='other', index=True)
    status = Column(String(20), default='active', index=True)  # active, inactive, sold, paused
    views = Column(Integer, default=0)
    featured = Column(Boolean, default=False)
    
    # Embedding fields for recommendation system
    profile_embedding = Column(Text, nullable=True)  # JSON string
    embedding_model = Column(String(50), default='all-MiniLM-L6-v2')
    embedding_version = Column(Integer, default=1)
    embedding_updated_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    owner_rel = relationship("User", back_populates="startups", foreign_keys=[owner_id])
    tags = relationship("StartupTag", back_populates="startup_rel", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="startup_rel", cascade="all, delete-orphan")
    applications = relationship("Application", back_populates="startup_rel")
    favorited_by = relationship("Favorite", back_populates="startup_rel")
    interests = relationship("Interest", back_populates="startup_rel")
    interactions = relationship("UserInteraction", back_populates="startup_rel")
    trending_metrics = relationship("StartupTrendingMetrics", back_populates="startup_rel", uselist=False)
    
    def get_embedding(self) -> Optional[List[float]]:
        """Parse profile_embedding JSON string to list"""
        return parse_json_field(self.profile_embedding)
    
    def get_stages(self) -> List[str]:
        """Parse stages JSON field"""
        return parse_json_field(self.stages) or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert startup to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'type': self.type,
            'status': self.status,
            'owner_id': self.owner_id,
        }


class StartupTag(Base):
    """StartupTag model - matches Django startup_tags table"""
    __tablename__ = 'startup_tags'
    __table_args__ = (
        UniqueConstraint('startup_id', 'tag', name='unique_startup_tag'),
        Index('idx_startup_tag', 'startup_id', 'tag'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False)
    tag = Column(String(100), nullable=False)
    
    # Relationships
    startup_rel = relationship("Startup", back_populates="tags")


class Position(Base):
    """Position model - matches Django positions table"""
    __tablename__ = 'positions'
    
    id = Column(String, primary_key=True)  # UUID
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False, index=True)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    requirements = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    startup_rel = relationship("Startup", back_populates="positions")
    applications = relationship("Application", back_populates="position_rel")
    interactions = relationship("UserInteraction", back_populates="position_rel")


class Application(Base):
    """Application model - matches Django applications table"""
    __tablename__ = 'applications'
    __table_args__ = (
        UniqueConstraint('startup_id', 'applicant_id', name='unique_startup_applicant'),
        Index('idx_app_startup', 'startup_id'),
        Index('idx_app_applicant', 'applicant_id'),
        Index('idx_app_status', 'status'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False)
    position_id = Column(String, ForeignKey('positions.id', ondelete='CASCADE'), nullable=False)
    applicant_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    cover_letter = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    portfolio_url = Column(String, nullable=True)
    resume_url = Column(String, nullable=True)
    status = Column(String(20), default='pending')  # pending, approved, rejected, withdrawn
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relationships
    startup_rel = relationship("Startup", back_populates="applications")
    position_rel = relationship("Position", back_populates="applications")
    applicant_rel = relationship("User", back_populates="applications")


# ============================================================================
# ENGAGEMENT MODELS
# ============================================================================

class Notification(Base):
    """Notification model - matches Django notifications table"""
    __tablename__ = 'notifications'
    __table_args__ = (
        Index('idx_notif_user_read', 'user_id', 'is_read'),
        Index('idx_notif_created', 'created_at'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(50), nullable=False)  # application_status, new_application, pitch, general
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=True)
    data = Column(Text, nullable=True)  # JSON string
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="notifications")
    
    def get_data(self) -> Dict[str, Any]:
        """Parse data JSON field"""
        return parse_json_field(self.data) or {}


class Favorite(Base):
    """Favorite model - matches Django favorites table"""
    __tablename__ = 'favorites'
    __table_args__ = (
        UniqueConstraint('user_id', 'startup_id', name='unique_user_startup_favorite'),
        Index('idx_fav_user', 'user_id'),
        Index('idx_fav_startup', 'startup_id'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="favorites")
    startup_rel = relationship("Startup", back_populates="favorited_by")


class Interest(Base):
    """Interest model - matches Django interests table"""
    __tablename__ = 'interests'
    __table_args__ = (
        UniqueConstraint('user_id', 'startup_id', name='unique_user_startup_interest'),
        Index('idx_int_user', 'user_id'),
        Index('idx_int_startup', 'startup_id'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="interests")
    startup_rel = relationship("Startup", back_populates="interests")


# ============================================================================
# MESSAGING MODELS
# ============================================================================

class Conversation(Base):
    """Conversation model - matches Django conversations table"""
    __tablename__ = 'conversations'
    __table_args__ = (
        Index('idx_conv_created', 'created_at'),
        Index('idx_conv_active', 'is_active'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    title = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Many-to-Many relationship with User through conversation_participants table
    # This is handled by Django's ManyToManyField, creating a junction table
    messages = relationship("Message", back_populates="conversation_rel")


# Junction table for Conversation-User Many-to-Many relationship
class ConversationParticipant(Base):
    """Junction table for Conversation-User Many-to-Many relationship"""
    __tablename__ = 'conversations_participants'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    __table_args__ = (
        UniqueConstraint('conversation_id', 'user_id', name='unique_conv_user'),
    )


class Message(Base):
    """Message model - matches Django messages table"""
    __tablename__ = 'messages'
    __table_args__ = (
        Index('idx_msg_conv_created', 'conversation_id', 'created_at'),
        Index('idx_msg_sender', 'sender_id'),
        Index('idx_msg_read', 'is_read'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    conversation_id = Column(String, ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    sender_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String(20), default='text')  # text, image, file
    attachment = Column(String, nullable=True)  # File path
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    conversation_rel = relationship("Conversation", back_populates="messages")
    sender_rel = relationship("User", back_populates="sent_messages")


class FileUpload(Base):
    """FileUpload model - matches Django file_uploads table"""
    __tablename__ = 'file_uploads'
    __table_args__ = (
        Index('idx_upload_user', 'user_id'),
        Index('idx_upload_type', 'file_type'),
        Index('idx_upload_created', 'created_at'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    file = Column(String, nullable=False)  # File path
    file_type = Column(String(20), nullable=False)  # resume, startup_image, profile_picture, message_attachment, other
    original_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)  # BigInteger in Django, Integer in SQLite
    mime_type = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="uploads")


# ============================================================================
# RECOMMENDATION MODELS
# ============================================================================

class UserInteraction(Base):
    """UserInteraction model - matches Django user_interactions table"""
    __tablename__ = 'user_interactions'
    __table_args__ = (
        UniqueConstraint('user_id', 'startup_id', 'interaction_type', name='unique_user_startup_interaction'),
        Index('idx_inter_user_startup_created', 'user_id', 'startup_id', 'created_at'),
        Index('idx_inter_startup_created', 'startup_id', 'created_at'),
        Index('idx_inter_type', 'interaction_type'),
        Index('idx_inter_position', 'position_id'),
        Index('idx_interaction_session_time', 'recommendation_session_id', 'created_at'),
        Index('idx_interaction_source_time', 'recommendation_source', 'created_at'),
        Index('idx_interaction_user_type_time', 'user_id', 'interaction_type', 'created_at'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), nullable=False)
    interaction_type = Column(String(30), nullable=False)  # view, click, like, dislike, favorite, apply, interest
    position_id = Column(String, ForeignKey('positions.id', ondelete='SET NULL'), nullable=True)
    weight = Column(Float, default=1.0)
    value_score = Column(Float, default=0.0)
    recommendation_session_id = Column(String, nullable=True)
    recommendation_source = Column(String(50), nullable=True)
    recommendation_rank = Column(Integer, nullable=True)
    recommendation_score = Column(Float, nullable=True)
    recommendation_method = Column(String(50), nullable=True)
    interaction_metadata = Column('metadata', Text, nullable=True)  # JSON string - using db_column to keep DB name as 'metadata'
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    user_rel = relationship("User", back_populates="interactions")
    startup_rel = relationship("Startup", back_populates="interactions")
    position_rel = relationship("Position", back_populates="interactions")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Parse metadata JSON field"""
        return parse_json_field(self.interaction_metadata) or {}


class StartupTrendingMetrics(Base):
    """StartupTrendingMetrics model - matches Django startup_trending_metrics table"""
    __tablename__ = 'startup_trending_metrics'
    __table_args__ = (
        Index('idx_trending_startup', 'startup_id'),
        Index('idx_trending_popularity', 'popularity_score'),
        Index('idx_trending_trending', 'trending_score'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    startup_id = Column(String, ForeignKey('startups.id', ondelete='CASCADE'), unique=True, nullable=False)
    popularity_score = Column(Float, default=0.0)
    trending_score = Column(Float, default=0.0)
    view_count_24h = Column(Integer, default=0)
    view_count_7d = Column(Integer, default=0)
    view_count_30d = Column(Integer, default=0)
    application_count_24h = Column(Integer, default=0)
    application_count_7d = Column(Integer, default=0)
    application_count_30d = Column(Integer, default=0)
    favorite_count_7d = Column(Integer, default=0)
    interest_count_7d = Column(Integer, default=0)
    active_positions_count = Column(Integer, default=0)
    velocity_score = Column(Float, default=0.0)
    computed_at = Column(DateTime, nullable=False)
    
    # Relationships
    startup_rel = relationship("Startup", back_populates="trending_metrics")


class RecommendationModel(Base):
    """RecommendationModel model - matches Django recommendation_models table"""
    __tablename__ = 'recommendation_models'
    __table_args__ = (
        Index('idx_rec_model_use_case', 'use_case', 'model_type', 'is_active'),
    )
    
    id = Column(String, primary_key=True)  # UUID
    model_name = Column(String(50), nullable=False)
    use_case = Column(String(50), nullable=False)  # developer_startup, founder_developer, founder_startup, investor_startup
    model_type = Column(String(30), nullable=False)  # content_based, als, two_tower, ranker
    file_path = Column(String(500), nullable=True)
    training_config = Column(Text, nullable=True)  # JSON string
    performance_metrics = Column(Text, nullable=True)  # JSON string
    is_active = Column(Boolean, default=True)
    trained_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Parse training_config JSON field"""
        return parse_json_field(self.training_config) or {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Parse performance_metrics JSON field"""
        return parse_json_field(self.performance_metrics) or {}


