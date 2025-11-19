"""
Django management command to backfill recommendation context for existing interactions

This script links existing UserInteraction records to RecommendationSession records
and populates recommendation context fields (rank, score, method, source).
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone
from datetime import timedelta
from api.recommendation_models import UserInteraction, RecommendationSession
from api.models import User, Startup
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Backfill recommendation context for existing UserInteraction records'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without actually updating records',
        )
        parser.add_argument(
            '--time-window-hours',
            type=int,
            default=24,
            help='Time window in hours to match interactions to sessions (default: 24)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of interactions to process per batch (default: 100)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        time_window_hours = options['time_window_hours']
        batch_size = options['batch_size']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be updated'))

        self.stdout.write(self.style.SUCCESS('\n=== Backfilling Recommendation Context ===\n'))

        # Find interactions without recommendation context
        interactions_without_context = UserInteraction.objects.filter(
            Q(recommendation_source__isnull=True) | Q(recommendation_source=''),
            Q(recommendation_rank__isnull=True),
            Q(recommendation_score__isnull=True),
        ).select_related('user', 'startup', 'recommendation_session')

        total_count = interactions_without_context.count()
        self.stdout.write(f'Found {total_count} interactions without recommendation context')

        if total_count == 0:
            self.stdout.write(self.style.SUCCESS('No interactions need backfilling'))
            return

        # Process in batches
        updated_count = 0
        linked_count = 0
        skipped_count = 0

        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch = interactions_without_context[batch_start:batch_end]

            self.stdout.write(f'\nProcessing batch {batch_start + 1}-{batch_end} of {total_count}...')

            for interaction in batch:
                try:
                    # Try to find matching recommendation session
                    session_match = self.find_matching_session(
                        interaction,
                        time_window_hours=time_window_hours
                    )

                    if session_match:
                        session, rank, score, method = session_match
                        
                        # Update interaction with recommendation context
                        if not dry_run:
                            interaction.recommendation_session = session
                            interaction.recommendation_source = 'recommendation'
                            interaction.recommendation_rank = rank
                            interaction.recommendation_score = score
                            interaction.recommendation_method = method
                            
                            # Update metadata if needed
                            if not interaction.metadata:
                                interaction.metadata = {}
                            interaction.metadata['recommendation_session_id'] = str(session.id)
                            interaction.metadata['source'] = 'recommendation'
                            
                            interaction.save(update_fields=[
                                'recommendation_session',
                                'recommendation_source',
                                'recommendation_rank',
                                'recommendation_score',
                                'recommendation_method',
                                'metadata',
                            ])
                        
                        linked_count += 1
                        updated_count += 1
                        
                        if linked_count % 10 == 0:
                            self.stdout.write(f'  Linked {linked_count} interactions...')
                    else:
                        # No matching session found - mark as organic
                        if not dry_run:
                            interaction.recommendation_source = 'organic'
                            interaction.save(update_fields=['recommendation_source'])
                        
                        updated_count += 1
                        skipped_count += 1

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error processing interaction {interaction.id}: {e}')
                    )
                    skipped_count += 1

        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS('BACKFILL SUMMARY'))
        self.stdout.write('=' * 60)
        self.stdout.write(f'Total interactions processed: {updated_count}')
        self.stdout.write(f'  Linked to sessions: {linked_count}')
        self.stdout.write(f'  Marked as organic: {skipped_count}')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No records were actually updated'))
        else:
            self.stdout.write(self.style.SUCCESS('\nBackfill completed successfully!'))

    def find_matching_session(self, interaction, time_window_hours=24):
        """
        Find matching RecommendationSession for an interaction
        
        Returns:
            Tuple of (session, rank, score, method) or None if no match found
        """
        user_id = interaction.user_id
        startup_id = interaction.startup_id
        interaction_time = interaction.created_at

        # Time window for matching
        time_window_start = interaction_time - timedelta(hours=time_window_hours)
        time_window_end = interaction_time + timedelta(hours=1)  # Allow 1 hour after interaction

        # Find sessions for this user within time window
        sessions = RecommendationSession.objects.filter(
            user_id=user_id,
            created_at__gte=time_window_start,
            created_at__lte=time_window_end,
        ).order_by('-created_at')

        # Check each session for matching startup
        for session in sessions:
            recommendations = session.recommendations_shown or []
            
            for rec in recommendations:
                rec_startup_id = rec.get('startup_id')
                
                # Handle both UUID and string formats
                if str(rec_startup_id) == str(startup_id):
                    rank = rec.get('rank')
                    score = rec.get('score')
                    method = rec.get('method') or session.recommendation_method
                    
                    return (session, rank, score, method)

        return None

