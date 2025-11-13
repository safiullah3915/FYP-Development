from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import math
from api.models import Startup
from api.recommendation_models import UserInteraction, StartupTrendingMetrics


def normalize(value, max_value=1):
    """Normalize value between 0 and 1"""
    if max_value == 0:
        return 0.0
    return min(value / max_value, 1.0)


class Command(BaseCommand):
    help = 'Compute trending metrics for all active startups'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Computing trending metrics...'))
        
        now = timezone.now()
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)
        
        startups = Startup.objects.filter(status='active')
        total = startups.count()
        processed = 0
        
        # First pass: Collect all interaction counts to find global maximums
        all_metrics = []
        for startup in startups:
            interactions_30d = UserInteraction.objects.filter(
                startup=startup,
                created_at__gte=cutoff_30d
            )
            interactions_7d = interactions_30d.filter(created_at__gte=cutoff_7d)
            interactions_24h = interactions_7d.filter(created_at__gte=cutoff_24h)
            
            view_count_24h = interactions_24h.filter(interaction_type='view').count()
            view_count_7d = interactions_7d.filter(interaction_type='view').count()
            view_count_30d = interactions_30d.filter(interaction_type='view').count()
            
            application_count_24h = interactions_24h.filter(interaction_type='apply').count()
            application_count_7d = interactions_7d.filter(interaction_type='apply').count()
            application_count_30d = interactions_30d.filter(interaction_type='apply').count()
            
            favorite_count_7d = interactions_7d.filter(interaction_type='favorite').count()
            favorite_count_30d = interactions_30d.filter(interaction_type='favorite').count()
            interest_count_7d = interactions_7d.filter(interaction_type='interest').count()
            interest_count_30d = interactions_30d.filter(interaction_type='interest').count()
            
            all_metrics.append({
                'startup': startup,
                'view_count_24h': view_count_24h,
                'view_count_7d': view_count_7d,
                'view_count_30d': view_count_30d,
                'application_count_24h': application_count_24h,
                'application_count_7d': application_count_7d,
                'application_count_30d': application_count_30d,
                'favorite_count_7d': favorite_count_7d,
                'favorite_count_30d': favorite_count_30d,
                'interest_count_7d': interest_count_7d,
                'interest_count_30d': interest_count_30d,
            })
        
        # Find global maximums across all startups
        if all_metrics:
            max_views_30d = max([m['view_count_30d'] for m in all_metrics], default=1)
            max_apps_30d = max([m['application_count_30d'] for m in all_metrics], default=1)
            max_favs_30d = max([m['favorite_count_30d'] for m in all_metrics], default=1)
            max_ints_30d = max([m['interest_count_30d'] for m in all_metrics], default=1)
            max_views_7d = max([m['view_count_7d'] for m in all_metrics], default=1)
            max_apps_7d = max([m['application_count_7d'] for m in all_metrics], default=1)
            max_engagement_7d = max([m['favorite_count_7d'] + m['interest_count_7d'] for m in all_metrics], default=1)
        else:
            max_views_30d = max_apps_30d = max_favs_30d = max_ints_30d = 1
            max_views_7d = max_apps_7d = max_engagement_7d = 1
        
        # Second pass: Calculate scores using global maximums
        for metrics_data in all_metrics:
            startup = metrics_data['startup']
            view_count_24h = metrics_data['view_count_24h']
            view_count_7d = metrics_data['view_count_7d']
            view_count_30d = metrics_data['view_count_30d']
            application_count_24h = metrics_data['application_count_24h']
            application_count_7d = metrics_data['application_count_7d']
            application_count_30d = metrics_data['application_count_30d']
            favorite_count_7d = metrics_data['favorite_count_7d']
            favorite_count_30d = metrics_data['favorite_count_30d']
            interest_count_7d = metrics_data['interest_count_7d']
            interest_count_30d = metrics_data['interest_count_30d']
            
            # Count active positions
            active_positions_count = startup.positions.filter(is_active=True).count()
            
            # Compute popularity score (30-day window) - normalized against global max
            # Use logarithmic scaling for better distribution
            def log_normalize(value, max_value):
                if max_value == 0 or value == 0:
                    return 0.0
                # Use log scale: log(1 + value) / log(1 + max_value)
                # This creates a more natural distribution where differences are more visible
                return math.log(1 + value) / math.log(1 + max_value)
            
            popularity_score = (
                0.25 * log_normalize(view_count_30d, max_views_30d) +
                0.35 * log_normalize(application_count_30d, max_apps_30d) +
                0.25 * log_normalize(favorite_count_30d, max_favs_30d) +
                0.15 * log_normalize(interest_count_30d, max_ints_30d)
            )
            # Cap popularity score at 1.0
            popularity_score = min(popularity_score, 1.0)
            
            # Compute trending score (7-day window with recency boost) - normalized against global max
            trending_score = (
                0.40 * log_normalize(view_count_7d, max_views_7d) +
                0.30 * log_normalize(application_count_7d, max_apps_7d) +
                0.30 * log_normalize(favorite_count_7d + interest_count_7d, max_engagement_7d)
            )
            
            # Velocity score (activity_7d / activity_30d) - measures growth momentum
            activity_7d = view_count_7d + application_count_7d + favorite_count_7d + interest_count_7d
            activity_30d = view_count_30d + application_count_30d + favorite_count_30d + interest_count_30d
            velocity_score = activity_7d / max(activity_30d, 1)
            
            # Apply velocity boost to trending score (startups with increasing activity get higher scores)
            # Cap velocity boost to prevent extreme scores
            velocity_boost = min(velocity_score, 2.0) * 0.25  # Max 50% boost
            trending_score = trending_score * (1 + velocity_boost)
            # Cap trending score at 1.0
            trending_score = min(trending_score, 1.0)
            
            # Update or create metrics
            StartupTrendingMetrics.objects.update_or_create(
                startup=startup,
                defaults={
                    'popularity_score': popularity_score,
                    'trending_score': trending_score * velocity_score,
                    'view_count_24h': view_count_24h,
                    'view_count_7d': view_count_7d,
                    'view_count_30d': view_count_30d,
                    'application_count_24h': application_count_24h,
                    'application_count_7d': application_count_7d,
                    'application_count_30d': application_count_30d,
                    'favorite_count_7d': favorite_count_7d,
                    'interest_count_7d': interest_count_7d,
                    'active_positions_count': active_positions_count,
                    'velocity_score': velocity_score,
                    # Note: StartupTrendingMetrics model doesn't have favorite_count_30d/interest_count_30d fields
                    # These are only used for velocity calculation
                }
            )
            
            processed += 1
            if processed % 10 == 0:
                self.stdout.write(f'Processed {processed}/{total} startups...')
        
        self.stdout.write(self.style.SUCCESS(f'Computed metrics for {processed} startups'))

