"""
Django management command to cleanup old recommendation sessions without interactions

Removes sessions that:
1. Have no linked interactions (orphaned sessions)
2. Are older than a specified date (optional)
3. Could create false hard negatives in training data
"""
from django.core.management.base import BaseCommand
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta
from api.recommendation_models import RecommendationSession, UserInteraction
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Cleanup old recommendation sessions without linked interactions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )
        parser.add_argument(
            '--days-old',
            type=int,
            default=None,
            help='Only delete sessions older than N days (default: delete all orphaned sessions)',
        )
        parser.add_argument(
            '--before-date',
            type=str,
            default=None,
            help='Only delete sessions created before this date (YYYY-MM-DD format)',
        )
        parser.add_argument(
            '--keep-recent',
            type=int,
            default=0,
            help='Always keep sessions from last N days even if orphaned (default: 0 - delete all orphaned)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        days_old = options['days_old']
        before_date_str = options['before_date']
        keep_recent_days = options['keep_recent']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be deleted'))

        self.stdout.write(self.style.SUCCESS('\n=== Cleaning Up Old Recommendation Sessions ===\n'))

        # Build query for orphaned sessions (no linked interactions)
        # A session is orphaned if no UserInteraction has recommendation_session_id pointing to it
        # Get all session IDs that have interactions
        sessions_with_interactions = UserInteraction.objects.filter(
            recommendation_session__isnull=False
        ).values_list('recommendation_session_id', flat=True).distinct()
        
        # Find sessions without any interactions
        orphaned_sessions = RecommendationSession.objects.exclude(
            id__in=sessions_with_interactions
        )

        # Apply date filters
        if before_date_str:
            try:
                from datetime import datetime
                before_date = datetime.strptime(before_date_str, '%Y-%m-%d').date()
                before_datetime = timezone.make_aware(
                    datetime.combine(before_date, datetime.min.time())
                )
                orphaned_sessions = orphaned_sessions.filter(created_at__lt=before_datetime)
                self.stdout.write(f'Filtering sessions before: {before_date_str}')
            except ValueError:
                self.stdout.write(self.style.ERROR(f'Invalid date format: {before_date_str}. Use YYYY-MM-DD'))
                return
        elif days_old:
            cutoff_date = timezone.now() - timedelta(days=days_old)
            orphaned_sessions = orphaned_sessions.filter(created_at__lt=cutoff_date)
            self.stdout.write(f'Filtering sessions older than {days_old} days')

        # Optionally keep recent sessions (even if orphaned) - they might get interactions soon
        if keep_recent_days > 0:
            recent_cutoff = timezone.now() - timedelta(days=keep_recent_days)
            orphaned_sessions = orphaned_sessions.filter(created_at__lt=recent_cutoff)
            self.stdout.write(f'Keeping sessions from last {keep_recent_days} days')
        else:
            self.stdout.write('Deleting ALL orphaned sessions (no date restriction)')

        total_count = orphaned_sessions.count()
        
        if total_count == 0:
            self.stdout.write(self.style.SUCCESS('No orphaned sessions found to clean up!'))
            return

        self.stdout.write(f'\nFound {total_count} orphaned sessions (no linked interactions)')
        
        # Show some statistics
        if total_count > 0:
            oldest = orphaned_sessions.order_by('created_at').first()
            newest = orphaned_sessions.order_by('-created_at').first()
            
            self.stdout.write(f'\nSession Statistics:')
            self.stdout.write(f'  Oldest orphaned session: {oldest.created_at if oldest else "N/A"}')
            self.stdout.write(f'  Newest orphaned session: {newest.created_at if newest else "N/A"}')
            
            # Count by use case
            use_case_counts = {}
            for session in orphaned_sessions[:100]:  # Sample first 100 for performance
                use_case = session.use_case
                use_case_counts[use_case] = use_case_counts.get(use_case, 0) + 1
            
            if use_case_counts:
                self.stdout.write(f'\nUse Case Distribution (sample):')
                for use_case, count in use_case_counts.items():
                    self.stdout.write(f'  {use_case}: {count}')

        # Confirm deletion
        if not dry_run:
            self.stdout.write(f'\n⚠️  About to delete {total_count} orphaned sessions')
            self.stdout.write(self.style.WARNING('This will remove sessions that have no linked interactions.'))
            self.stdout.write(self.style.WARNING('These sessions could create false hard negatives in training.'))
            
            # Delete in batches for better performance
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, total_count, batch_size):
                batch = orphaned_sessions[i:i + batch_size]
                batch_ids = list(batch.values_list('id', flat=True))
                
                # Delete batch
                deleted, _ = RecommendationSession.objects.filter(id__in=batch_ids).delete()
                deleted_count += deleted
                
                if (i + batch_size) % 500 == 0:
                    self.stdout.write(f'  Deleted {deleted_count}/{total_count} sessions...')

            self.stdout.write(self.style.SUCCESS(f'\n✅ Successfully deleted {deleted_count} orphaned sessions'))
        else:
            self.stdout.write(self.style.WARNING(f'\nDRY RUN - Would delete {total_count} sessions'))
            self.stdout.write('Run without --dry-run to actually delete them')

        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS('Cleanup completed!'))
        self.stdout.write('=' * 60 + '\n')

