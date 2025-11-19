"""
Real-time Trending Metrics Service
Handles incremental updates to trending metrics when user interactions occur
"""
from django.utils import timezone
from django.db import transaction
from django.db.models import Max
from datetime import timedelta
import math


class TrendingMetricsService:
    """Service for real-time trending metrics calculation"""
    
    MIN_VIEWS_7D = 1  # Lowered from 3 to allow startups with at least 1 view to have trending scores
    MIN_ACTIVITY_7D = 5
    
    def __init__(self):
        self.interaction_weights = {
            'view': 0.5,
            'click': 1.0,
            'like': 2.0,
            'dislike': -1.0,
            'favorite': 2.5,
            'apply': 3.0,
            'interest': 3.5,
        }
    
    @transaction.atomic
    def increment_interaction(self, startup_id, interaction_type):
        """
        Increment metrics immediately when interaction is created
        
        Args:
            startup_id: UUID of the startup
            interaction_type: Type of interaction (view, favorite, apply, etc.)
        """
        from api.models import Startup
        from api.recommendation_models import StartupTrendingMetrics, UserInteraction
        
        try:
            startup = Startup.objects.get(id=startup_id)
        except Startup.DoesNotExist:
            print(f"⚠️ [TrendingMetrics] Startup {startup_id} not found")
            return
        
        metrics, _ = StartupTrendingMetrics.objects.get_or_create(
            startup=startup,
            defaults={'popularity_score': 0.0, 'trending_score': 0.0}
        )
        
        now = timezone.now()
        counts = self._collect_interaction_counts(startup, UserInteraction, now)
        
        metrics.view_count_24h = counts['view_count_24h']
        metrics.view_count_7d = counts['view_count_7d']
        metrics.view_count_30d = counts['view_count_30d']
        metrics.application_count_24h = counts['application_count_24h']
        metrics.application_count_7d = counts['application_count_7d']
        metrics.application_count_30d = counts['application_count_30d']
        metrics.favorite_count_7d = counts['favorite_count_7d']
        metrics.interest_count_7d = counts['interest_count_7d']
        metrics.active_positions_count = startup.positions.filter(is_active=True).count()
        metrics.last_interaction_at = now
        
        self.recalculate_scores(metrics, startup, counts)
        metrics.save()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"✅ [TrendingMetrics] Updated metrics for startup {startup_id} - {interaction_type}")
        logger.info(f"   Trending Score: {metrics.trending_score:.3f}, Popularity: {metrics.popularity_score:.3f}, View Count 7d: {metrics.view_count_7d}")
    
    def recalculate_scores(self, metrics, startup, counts):
        """
        Recalculate trending and popularity scores for a single startup
        Uses same formula as compute_trending_metrics command
        
        Args:
            metrics: StartupTrendingMetrics instance
            startup: Startup instance
            counts: Precomputed interaction counts for various windows
        """
        # Log normalize function (same as in compute_trending_metrics)
        def log_normalize(value, max_value=100):
            if max_value == 0 or value == 0:
                return 0.0
            return math.log(1 + value) / math.log(1 + max_value)
        
        from api.recommendation_models import StartupTrendingMetrics as MetricsModel
        
        max_values = MetricsModel.objects.aggregate(
            max_views_30d=Max('view_count_30d'),
            max_apps_30d=Max('application_count_30d'),
            max_views_7d=Max('view_count_7d'),
            max_apps_7d=Max('application_count_7d'),
            max_favs_7d=Max('favorite_count_7d'),
            max_ints_7d=Max('interest_count_7d')
        )
        
        max_views_30d = max(max_values.get('max_views_30d') or 1, counts['view_count_30d'])
        max_apps_30d = max(max_values.get('max_apps_30d') or 1, counts['application_count_30d'])
        max_favs_30d = max(max_values.get('max_favs_7d') or 1, counts['favorite_count_30d'])
        max_ints_30d = max(max_values.get('max_ints_7d') or 1, counts['interest_count_30d'])
        max_views_7d = max(max_values.get('max_views_7d') or 1, counts['view_count_7d'])
        max_apps_7d = max(max_values.get('max_apps_7d') or 1, counts['application_count_7d'])
        max_engagement_7d = max(
            (max_values.get('max_favs_7d') or 0) + (max_values.get('max_ints_7d') or 0),
            counts['favorite_count_7d'] + counts['interest_count_7d'],
            1
        )
        
        # Calculate popularity score (30-day window)
        popularity_score = (
            0.25 * log_normalize(counts['view_count_30d'], max_views_30d) +
            0.35 * log_normalize(counts['application_count_30d'], max_apps_30d) +
            0.25 * log_normalize(counts['favorite_count_30d'], max_favs_30d) +
            0.15 * log_normalize(counts['interest_count_30d'], max_ints_30d)
        )
        popularity_score = min(popularity_score, 1.0)
        
        # Calculate trending score (7-day window)
        trending_score = (
            0.40 * log_normalize(counts['view_count_7d'], max_views_7d) +
            0.30 * log_normalize(counts['application_count_7d'], max_apps_7d) +
            0.30 * log_normalize(
                counts['favorite_count_7d'] + counts['interest_count_7d'],
                max_engagement_7d
            )
        )
        
        # Calculate velocity score
        activity_7d = (
            counts['view_count_7d'] +
            counts['application_count_7d'] +
            counts['favorite_count_7d'] +
            counts['interest_count_7d']
        )
        activity_prev_7d = (
            counts['view_count_prev_7d'] +
            counts['application_count_prev_7d'] +
            counts['favorite_count_prev_7d'] +
            counts['interest_count_prev_7d']
        )
        velocity_score = activity_7d / max(activity_prev_7d, 1)
        if activity_7d < self.MIN_ACTIVITY_7D:
            velocity_score = 0.0
        
        # Apply velocity boost
        velocity_boost = min(velocity_score, 2.0) * 0.25
        trending_score = trending_score * (1 + velocity_boost)
        trending_score = min(trending_score, 1.0)
        
        # Count active positions
        active_positions_count = startup.positions.filter(is_active=True).count()
        
        if counts['view_count_7d'] < self.MIN_VIEWS_7D:
            trending_score = 0.0
        
        # Update metrics
        metrics.popularity_score = popularity_score
        metrics.trending_score = trending_score
        metrics.velocity_score = velocity_score
        metrics.active_positions_count = active_positions_count
    
    def _collect_interaction_counts(self, startup, interaction_model, now):
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)
        cutoff_30d = now - timedelta(days=30)
        
        interactions_30d = interaction_model.objects.filter(
            startup=startup,
            created_at__gte=cutoff_30d
        )
        interactions_7d = interactions_30d.filter(created_at__gte=cutoff_7d)
        interactions_prev_7d = interactions_30d.filter(created_at__lt=cutoff_7d, created_at__gte=cutoff_14d)
        interactions_24h = interactions_7d.filter(created_at__gte=cutoff_24h)
        
        def count(qs, interaction_type):
            return qs.filter(interaction_type=interaction_type).count()
        
        return {
            'view_count_24h': count(interactions_24h, 'view'),
            'view_count_7d': count(interactions_7d, 'view'),
            'view_count_30d': count(interactions_30d, 'view'),
            'view_count_prev_7d': count(interactions_prev_7d, 'view'),
            'application_count_24h': count(interactions_24h, 'apply'),
            'application_count_7d': count(interactions_7d, 'apply'),
            'application_count_30d': count(interactions_30d, 'apply'),
            'application_count_prev_7d': count(interactions_prev_7d, 'apply'),
            'favorite_count_7d': count(interactions_7d, 'favorite'),
            'favorite_count_30d': count(interactions_30d, 'favorite'),
            'favorite_count_prev_7d': count(interactions_prev_7d, 'favorite'),
            'interest_count_7d': count(interactions_7d, 'interest'),
            'interest_count_30d': count(interactions_30d, 'interest'),
            'interest_count_prev_7d': count(interactions_prev_7d, 'interest'),
        }
    
    @transaction.atomic
    def apply_time_decay_single(self, startup_id):
        """
        Apply time decay to a single startup's metrics
        Reduces counts based on time elapsed since last decay
        
        Args:
            startup_id: UUID of the startup
        """
        from api.models import Startup
        from api.recommendation_models import StartupTrendingMetrics, UserInteraction
        
        try:
            startup = Startup.objects.get(id=startup_id)
            metrics = StartupTrendingMetrics.objects.get(startup=startup)
        except (Startup.DoesNotExist, StartupTrendingMetrics.DoesNotExist):
            return
        
        now = timezone.now()
        counts = self._collect_interaction_counts(startup, UserInteraction, now)
        
        metrics.view_count_24h = counts['view_count_24h']
        metrics.view_count_7d = counts['view_count_7d']
        metrics.view_count_30d = counts['view_count_30d']
        metrics.application_count_24h = counts['application_count_24h']
        metrics.application_count_7d = counts['application_count_7d']
        metrics.application_count_30d = counts['application_count_30d']
        metrics.favorite_count_7d = counts['favorite_count_7d']
        metrics.interest_count_7d = counts['interest_count_7d']
        metrics.last_decay_applied_at = now
        
        self.recalculate_scores(metrics, startup, counts)
        
        metrics.save()
        
        print(f"✅ [TrendingMetrics] Applied decay to startup {startup_id} using recalculated counts")

