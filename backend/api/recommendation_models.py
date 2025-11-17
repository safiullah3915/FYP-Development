import uuid
from django.db import models
from django.utils import timezone


class UserInteraction(models.Model):
    """Unified interaction tracking for recommendation system"""
    INTERACTION_TYPE_CHOICES = [
        ('view', 'View'),
        ('click', 'Click'),
        ('like', 'Like'),
        ('dislike', 'Dislike'),
        ('favorite', 'Favorite'),
        ('apply', 'Apply'),
        ('interest', 'Interest'),
    ]
    
    # Weight mapping (computed on save via signal or save() override):
    # 'view': 0.5
    # 'click': 1.0
    # 'like': 2.0
    # 'dislike': -1.0 (negative signal)
    # 'favorite': 2.5
    # 'apply': 3.0
    # 'interest': 3.5
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey('api.User', on_delete=models.CASCADE, related_name='interactions', db_column='user_id')
    startup = models.ForeignKey('api.Startup', on_delete=models.CASCADE, related_name='interactions', db_column='startup_id')
    interaction_type = models.CharField(max_length=30, choices=INTERACTION_TYPE_CHOICES)
    
    # Optional: Track which position if applicable (for analytics only)
    position = models.ForeignKey('api.Position', on_delete=models.SET_NULL, null=True, blank=True, db_column='position_id')
    
    # Precomputed weight for training (calculated on save)
    weight = models.FloatField(default=1.0)
    value_score = models.FloatField(default=0.0, db_index=True)
    
    # Recommendation context (extracted from metadata for analytics/perf)
    recommendation_session = models.ForeignKey(
        'api.RecommendationSession',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        db_column='recommendation_session_id',
        related_name='interaction_events'
    )
    recommendation_source = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        choices=[('organic', 'Organic'), ('recommendation', 'Recommendation')],
        db_index=True
    )
    recommendation_rank = models.IntegerField(null=True, blank=True)
    recommendation_score = models.FloatField(null=True, blank=True)
    recommendation_method = models.CharField(max_length=50, null=True, blank=True)
    
    # Additional context
    metadata = models.JSONField(default=dict, blank=True)  # session_id, device, etc.
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_interactions'
        indexes = [
            models.Index(fields=['user', 'startup', 'created_at']),
            models.Index(fields=['startup', 'created_at']),
            models.Index(fields=['interaction_type']),
            models.Index(fields=['position']),
            # ETL-optimized index for date-range queries
            models.Index(fields=['created_at', 'interaction_type']),
            models.Index(fields=['recommendation_session', 'created_at'], name='idx_interaction_session_time'),
            models.Index(fields=['recommendation_source', 'created_at'], name='idx_interaction_source_time'),
            models.Index(fields=['user', 'interaction_type', 'created_at'], name='idx_interaction_user_type_time'),
        ]
        # Unique constraint to prevent duplicate interactions of same type
        # Allows multiple interactions of different types, but only one per type per user-startup pair
        unique_together = [['user', 'startup', 'interaction_type']]
    
    def save(self, *args, **kwargs):
        # Calculate weight based on interaction type
        weight_mapping = {
            'view': 0.5,
            'click': 1.0,
            'like': 2.0,
            'dislike': -1.0,
            'favorite': 2.5,
            'apply': 3.0,
            'interest': 3.5,
        }
        self.weight = weight_mapping.get(self.interaction_type, 1.0)
        super().save(*args, **kwargs)
    
    def get_metadata_value(self, key: str, default=None):
        """Helper to safely get metadata value"""
        if isinstance(self.metadata, dict):
            return self.metadata.get(key, default)
        return default
    
    def is_recommendation_interaction(self) -> bool:
        """Check if interaction is from recommendation"""
        return self.get_metadata_value('source') == 'recommendation'
    
    def get_recommendation_session_id(self) -> str:
        """Get recommendation session ID if exists"""
        return self.get_metadata_value('recommendation_session_id')
    
    def get_recommendation_rank(self) -> int:
        """Get recommendation rank if exists"""
        return self.get_metadata_value('recommendation_rank')
    
    def __str__(self):
        return f"{self.user.username} - {self.interaction_type} - {self.startup.title}"


class UserOnboardingPreferences(models.Model):
    """Initial user preferences for cold-start recommendations"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        'api.User',
        on_delete=models.CASCADE,
        db_column='user_id',
        related_name='onboarding_preferences'
    )
    
    # Selected categories (from startup.category choices)
    selected_categories = models.JSONField(default=list, blank=True)  # e.g., ['saas', 'ecommerce', 'web3']
    
    # Selected fields (from startup.field)
    selected_fields = models.JSONField(default=list, blank=True)  # e.g., ['fintech', 'healthcare', 'education']
    
    # Selected tags (from startup_tags.tag)
    selected_tags = models.JSONField(default=list, blank=True)  # e.g., ['AI', 'blockchain', 'mobile']
    
    # Preferred startup stages (from startup.stages)
    preferred_startup_stages = models.JSONField(default=list, blank=True)  # e.g., ['early', 'growth', 'mature']
    
    # Preferred engagement types (for developers)
    preferred_engagement_types = models.JSONField(default=list, blank=True)  # e.g., ['full-time', 'part-time', 'equity', 'paid']
    
    # Preferred skills (for developers - skills they want to work with)
    preferred_skills = models.JSONField(default=list, blank=True)  # e.g., ['Python', 'React', 'Machine Learning']
    
    # Investor specific structured payload (mirrors serializer schema)
    investor_profile = models.JSONField(default=dict, blank=True)
    
    onboarding_completed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'user_onboarding_preferences'
        indexes = [
            models.Index(fields=['user']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - Onboarding Preferences"


class StartupTrendingMetrics(models.Model):
    """Computed trending/popularity metrics for non-personalized recommendations"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    startup = models.OneToOneField(
        'api.Startup',
        on_delete=models.CASCADE,
        db_column='startup_id',
        related_name='trending_metrics'
    )
    
    # Overall popularity score (30-day window)
    popularity_score = models.FloatField(default=0.0)
    
    # Recent trending score (7-day window with decay)
    trending_score = models.FloatField(default=0.0)
    
    # View counts
    view_count_24h = models.IntegerField(default=0)
    view_count_7d = models.IntegerField(default=0)
    view_count_30d = models.IntegerField(default=0)
    
    # Application counts
    application_count_24h = models.IntegerField(default=0)
    application_count_7d = models.IntegerField(default=0)
    application_count_30d = models.IntegerField(default=0)
    
    # Engagement counts
    favorite_count_7d = models.IntegerField(default=0)
    interest_count_7d = models.IntegerField(default=0)
    
    # Active positions count (updated daily)
    active_positions_count = models.IntegerField(default=0)
    
    # Growth velocity (activity_7d / activity_30d)
    velocity_score = models.FloatField(default=0.0)
    
    # Last computation timestamp
    computed_at = models.DateTimeField(auto_now=True)
    
    # Track when metrics were last updated (for real-time updates)
    last_interaction_at = models.DateTimeField(null=True, blank=True)
    
    # Track when time decay was last applied
    last_decay_applied_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'startup_trending_metrics'
        indexes = [
            models.Index(fields=['startup']),
            models.Index(fields=['-popularity_score']),  # For sorting popular
            models.Index(fields=['-trending_score']),    # For sorting trending
        ]
    
    def __str__(self):
        return f"{self.startup.title} - Popularity: {self.popularity_score:.2f}, Trending: {self.trending_score:.2f}"


class RecommendationModel(models.Model):
    """Model metadata tracking for recommendation system"""
    USE_CASE_CHOICES = [
        ('developer_startup', 'Developer → Startup'),
        ('founder_developer', 'Founder → Developer'),
        ('founder_startup', 'Founder → Startup'),
        ('investor_startup', 'Investor → Startup'),
    ]
    
    MODEL_TYPE_CHOICES = [
        ('content_based', 'Content-Based'),
        ('als', 'ALS Collaborative Filtering'),
        ('two_tower', 'Two-Tower Deep Learning'),
        ('ranker', 'Ranking Model'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    model_name = models.CharField(max_length=50)  # e.g., 'als_developer_startup', 'ranker_founder_developer'
    
    use_case = models.CharField(
        max_length=50,
        choices=USE_CASE_CHOICES
    )
    
    model_type = models.CharField(
        max_length=30,
        choices=MODEL_TYPE_CHOICES
    )
    
    file_path = models.CharField(max_length=500, blank=True)  # Path to saved model file
    
    training_config = models.JSONField(default=dict, blank=True)  # Hyperparameters used
    
    performance_metrics = models.JSONField(default=dict, blank=True)  # Latest offline metrics: {'precision_at_20': 0.31, 'recall_at_20': 0.42, 'ndcg_at_20': 0.47, 'map': 0.28}
    
    is_active = models.BooleanField(default=True)  # Whether this model is currently in use
    
    trained_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'recommendation_models'
        indexes = [
            models.Index(fields=['use_case', 'model_type', 'is_active']),
        ]
    
    def __str__(self):
        return f"{self.model_name} - {self.use_case} - {self.model_type}"


class RecommendationSession(models.Model):
    """Track recommendation sessions for feedback analysis and training"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.ForeignKey('api.User', on_delete=models.CASCADE, db_column='user_id')
    use_case = models.CharField(max_length=50)  # 'developer_startup', 'founder_developer', etc.
    recommendation_method = models.CharField(max_length=50)  # 'content_based', 'collaborative', etc.
    model_version = models.CharField(max_length=50, blank=True)  # Model version used (for future training tracking)
    recommendations_shown = models.JSONField(default=list)  # Full recommendation data with ranks
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)  # Indexed for cleanup queries
    
    # Analytics fields (computed periodically - ETL-ready)
    # These fields are pre-computed for performance, but can be recalculated via ETL
    total_interactions = models.IntegerField(default=0, db_index=True)  # Indexed for ETL filtering
    total_clicks = models.IntegerField(default=0)
    total_likes = models.IntegerField(default=0)
    total_dislikes = models.IntegerField(default=0)
    total_applies = models.IntegerField(default=0)
    ctr = models.FloatField(default=0.0, db_index=True)  # Indexed for ETL sorting
    engagement_rate = models.FloatField(default=0.0)
    metrics_computed_at = models.DateTimeField(null=True, blank=True, db_index=True)  # For ETL: find uncomputed sessions
    
    # ETL metadata (for future dataset extraction)
    etl_processed = models.BooleanField(default=False, db_index=True)  # Mark if used in training dataset
    etl_processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'recommendation_sessions'
        indexes = [
            # Existing indexes
            models.Index(fields=['user_id', 'created_at']),
            models.Index(fields=['use_case', 'created_at']),
            models.Index(fields=['recommendation_method']),
            # ETL-optimized indexes
            models.Index(fields=['created_at', 'use_case']),  # For date-range ETL queries
            models.Index(fields=['etl_processed', 'created_at']),  # Find unprocessed sessions
            models.Index(fields=['use_case', 'metrics_computed_at']),  # For metrics update jobs
        ]
        ordering = ['-created_at']
    
    def get_interactions(self):
        """Get all interactions linked to this session - ETL-ready method"""
        from django.db.models import Q
        return UserInteraction.objects.filter(
            Q(metadata__recommendation_session_id=str(self.id)) |
            Q(metadata__contains={'recommendation_session_id': str(self.id)})
        )
    
    def calculate_metrics(self):
        """Calculate metrics - can be called by ETL or cron jobs"""
        interactions = self.get_interactions()
        total_shown = len(self.recommendations_shown)
        
        if total_shown == 0:
            return {}
        
        metrics = {
            'total_shown': total_shown,
            'total_interactions': interactions.count(),
            'ctr': interactions.count() / total_shown,
            'likes': interactions.filter(interaction_type='like').count(),
            'dislikes': interactions.filter(interaction_type='dislike').count(),
            'applies': interactions.filter(interaction_type='apply').count(),
            'favorites': interactions.filter(interaction_type='favorite').count(),
            'engagement_rate': interactions.filter(
                interaction_type__in=['like', 'favorite', 'apply', 'interest']
            ).count() / total_shown
        }
        
        # Update model fields (for ETL queries)
        self.total_interactions = metrics['total_interactions']
        self.total_clicks = interactions.filter(interaction_type='click').count()
        self.total_likes = metrics['likes']
        self.total_dislikes = metrics['dislikes']
        self.total_applies = metrics['applies']
        self.ctr = metrics['ctr']
        self.engagement_rate = metrics['engagement_rate']
        self.metrics_computed_at = timezone.now()
        
        return metrics
    
    def __str__(self):
        return f"Session {self.id} - {self.use_case} - {self.user_id.username}"

