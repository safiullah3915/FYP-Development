from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
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
        
        for startup in startups:
            # Get interactions for this startup
            interactions_30d = UserInteraction.objects.filter(
                startup=startup,
                created_at__gte=cutoff_30d
            )
            interactions_7d = interactions_30d.filter(created_at__gte=cutoff_7d)
            interactions_24h = interactions_7d.filter(created_at__gte=cutoff_24h)
            
            # Count by type
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
            
            # Count active positions
            active_positions_count = startup.positions.filter(is_active=True).count()
            
            # Compute scores (simplified normalization)
            max_views = max(view_count_30d, 1)
            max_apps = max(application_count_30d, 1)
            max_favs = max(favorite_count_30d, 1)
            max_ints = max(interest_count_30d, 1)
            
            popularity_score = (
                0.3 * normalize(view_count_30d, max_views) +
                0.4 * normalize(application_count_30d, max_apps) +
                0.2 * normalize(favorite_count_30d, max_favs) +
                0.1 * normalize(interest_count_30d, max_ints)
            )
            
            trending_score = (
                0.5 * normalize(view_count_7d, max_views) +
                0.3 * normalize(application_count_7d, max_apps) +
                0.2 * normalize(favorite_count_7d + interest_count_7d, max_favs + max_ints)
            )
            
            # Velocity score (activity_7d / activity_30d) - Fixed: use 30-day counts for all interaction types
            activity_7d = view_count_7d + application_count_7d + favorite_count_7d + interest_count_7d
            activity_30d = view_count_30d + application_count_30d + favorite_count_30d + interest_count_30d
            velocity_score = activity_7d / max(activity_30d, 1)
            
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

