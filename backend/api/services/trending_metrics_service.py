"""
Real-time Trending Metrics Service
Handles incremental updates to trending metrics when user interactions occur
"""
from django.utils import timezone
from django.db import transaction
from datetime import timedelta
import math


class TrendingMetricsService:
    """Service for real-time trending metrics calculation"""
    
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
        from api.recommendation_models import StartupTrendingMetrics
        
        try:
            startup = Startup.objects.get(id=startup_id)
        except Startup.DoesNotExist:
            print(f"⚠️ [TrendingMetrics] Startup {startup_id} not found")
            return
        
        # Get or create metrics
        metrics, created = StartupTrendingMetrics.objects.get_or_create(
            startup=startup,
            defaults={
                'popularity_score': 0.0,
                'trending_score': 0.0,
                'view_count_24h': 0,
                'view_count_7d': 0,
                'view_count_30d': 0,
                'application_count_24h': 0,
                'application_count_7d': 0,
                'application_count_30d': 0,
                'favorite_count_7d': 0,
                'interest_count_7d': 0,
                'velocity_score': 0.0,
            }
        )
        
        if created:
            print(f"✅ [TrendingMetrics] Created new metrics for startup {startup_id}")
        
        # Increment appropriate counters based on interaction type
        now = timezone.now()
        
        if interaction_type == 'view':
            metrics.view_count_24h += 1
            metrics.view_count_7d += 1
            metrics.view_count_30d += 1
        elif interaction_type == 'apply':
            metrics.application_count_24h += 1
            metrics.application_count_7d += 1
            metrics.application_count_30d += 1
        elif interaction_type == 'favorite':
            metrics.favorite_count_7d += 1
        elif interaction_type == 'interest':
            metrics.interest_count_7d += 1
        
        # Update timestamp
        metrics.last_interaction_at = now
        
        # Recalculate scores
        self.recalculate_scores(metrics, startup)
        
        # Save
        metrics.save()
        
        print(f"✅ [TrendingMetrics] Updated metrics for startup {startup_id} - {interaction_type}")
        print(f"   Trending Score: {metrics.trending_score:.3f}, Popularity: {metrics.popularity_score:.3f}")
    
    def recalculate_scores(self, metrics, startup):
        """
        Recalculate trending and popularity scores for a single startup
        Uses same formula as compute_trending_metrics command
        
        Args:
            metrics: StartupTrendingMetrics instance
            startup: Startup instance
        """
        # Log normalize function (same as in compute_trending_metrics)
        def log_normalize(value, max_value=100):
            if max_value == 0 or value == 0:
                return 0.0
            return math.log(1 + value) / math.log(1 + max_value)
        
        # Use reasonable max values for normalization
        # These should ideally be global maximums, but we use estimates for real-time
        max_views_30d = 1000
        max_apps_30d = 100
        max_favs_30d = 100
        max_ints_30d = 100
        max_views_7d = 500
        max_apps_7d = 50
        max_engagement_7d = 100
        
        # Calculate popularity score (30-day window)
        popularity_score = (
            0.25 * log_normalize(metrics.view_count_30d, max_views_30d) +
            0.35 * log_normalize(metrics.application_count_30d, max_apps_30d) +
            0.25 * log_normalize(metrics.favorite_count_7d, max_favs_30d) +  # Using 7d for favorites
            0.15 * log_normalize(metrics.interest_count_7d, max_ints_30d)    # Using 7d for interests
        )
        popularity_score = min(popularity_score, 1.0)
        
        # Calculate trending score (7-day window)
        trending_score = (
            0.40 * log_normalize(metrics.view_count_7d, max_views_7d) +
            0.30 * log_normalize(metrics.application_count_7d, max_apps_7d) +
            0.30 * log_normalize(
                metrics.favorite_count_7d + metrics.interest_count_7d,
                max_engagement_7d
            )
        )
        
        # Calculate velocity score
        activity_7d = (
            metrics.view_count_7d +
            metrics.application_count_7d +
            metrics.favorite_count_7d +
            metrics.interest_count_7d
        )
        activity_30d = (
            metrics.view_count_30d +
            metrics.application_count_30d +
            metrics.favorite_count_7d +  # Using 7d counts for these
            metrics.interest_count_7d
        )
        velocity_score = activity_7d / max(activity_30d, 1)
        
        # Apply velocity boost
        velocity_boost = min(velocity_score, 2.0) * 0.25
        trending_score = trending_score * (1 + velocity_boost)
        trending_score = min(trending_score, 1.0)
        
        # Count active positions
        active_positions_count = startup.positions.filter(is_active=True).count()
        
        # Update metrics
        metrics.popularity_score = popularity_score
        metrics.trending_score = trending_score
        metrics.velocity_score = velocity_score
        metrics.active_positions_count = active_positions_count
    
    @transaction.atomic
    def apply_time_decay_single(self, startup_id):
        """
        Apply time decay to a single startup's metrics
        Reduces counts based on time elapsed since last decay
        
        Args:
            startup_id: UUID of the startup
        """
        from api.models import Startup
        from api.recommendation_models import StartupTrendingMetrics
        
        try:
            startup = Startup.objects.get(id=startup_id)
            metrics = StartupTrendingMetrics.objects.get(startup=startup)
        except (Startup.DoesNotExist, StartupTrendingMetrics.DoesNotExist):
            return
        
        now = timezone.now()
        
        # Calculate time since last decay
        if metrics.last_decay_applied_at:
            hours_since_decay = (now - metrics.last_decay_applied_at).total_seconds() / 3600
        else:
            hours_since_decay = 24  # Default to 1 day if never decayed
        
        # Apply exponential decay
        decay_rate = 0.15 / 24  # Decay rate per hour (0.15 per day)
        decay_factor = math.exp(-decay_rate * hours_since_decay)
        
        # Apply decay to time-sensitive counts
        metrics.view_count_24h = int(metrics.view_count_24h * decay_factor)
        metrics.view_count_7d = int(metrics.view_count_7d * decay_factor)
        metrics.application_count_24h = int(metrics.application_count_24h * decay_factor)
        metrics.application_count_7d = int(metrics.application_count_7d * decay_factor)
        
        # Update decay timestamp
        metrics.last_decay_applied_at = now
        
        # Recalculate scores
        self.recalculate_scores(metrics, startup)
        
        metrics.save()
        
        print(f"✅ [TrendingMetrics] Applied decay to startup {startup_id} (factor: {decay_factor:.3f})")

