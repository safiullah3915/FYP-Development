"""
Analyze recommendation engagement by method and rank bucket.
Calculates exposures, interactions, CTR, and engagement rate.
"""
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Optional, Tuple
import csv

from django.core.management.base import BaseCommand
from django.utils import timezone

from api.recommendation_models import RecommendationSession, UserInteraction


class Command(BaseCommand):
    help = 'Analyze recommendation performance by method and rank bucket'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Lookback window in days (default: 30)'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='',
            help='Optional CSV output path'
        )

    def handle(self, *args, **options):
        lookback_days = options['days']
        output_path = options['output']

        cutoff = timezone.now() - timedelta(days=lookback_days)
        self.stdout.write(self.style.SUCCESS(f'Analyzing recommendation performance (last {lookback_days} days)...'))

        metrics = defaultdict(lambda: {'exposures': 0, 'interactions': 0, 'engagements': 0})
        exposure_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}

        self._aggregate_exposures(cutoff, metrics, exposure_lookup)
        self._aggregate_interactions(cutoff, metrics, exposure_lookup)

        rows = self._format_results(metrics)
        self._print_results(rows)

        if output_path:
            self._write_csv(rows, output_path)
            self.stdout.write(self.style.SUCCESS(f'Results written to {output_path}'))

    def _aggregate_exposures(self, cutoff, metrics, exposure_lookup):
        sessions = RecommendationSession.objects.filter(created_at__gte=cutoff)
        for session in sessions:
            method = session.recommendation_method or 'unknown'
            for rec in session.recommendations_shown or []:
                startup_id = rec.get('startup_id')
                if not startup_id:
                    continue
                rank = rec.get('rank')
                bucket = self.get_rank_bucket(rank)
                key = (method, bucket)
                metrics[key]['exposures'] += 1
                exposure_lookup[(str(session.id), str(startup_id))] = {
                    'method': method,
                    'bucket': bucket
                }

    def _aggregate_interactions(self, cutoff, metrics, exposure_lookup):
        interactions = UserInteraction.objects.filter(
            recommendation_source='recommendation',
            created_at__gte=cutoff
        )
        positive_types = {'like', 'favorite', 'apply', 'interest'}

        for interaction in interactions:
            method = interaction.recommendation_method
            bucket = self.get_rank_bucket(interaction.recommendation_rank)

            if (not method or bucket == 'unknown') and interaction.recommendation_session_id:
                lookup_key = (str(interaction.recommendation_session_id), str(interaction.startup_id))
                exposure_info = exposure_lookup.get(lookup_key)
                if exposure_info:
                    method = method or exposure_info['method']
                    if bucket == 'unknown':
                        bucket = exposure_info['bucket']

            method = method or 'unknown'
            bucket = bucket or 'unknown'
            key = (method, bucket)
            metrics[key]['interactions'] += 1
            if interaction.interaction_type in positive_types:
                metrics[key]['engagements'] += 1

    def _format_results(self, metrics):
        rows = []
        for (method, bucket), data in metrics.items():
            exposures = data['exposures']
            interactions = data['interactions']
            engagements = data['engagements']
            ctr = (interactions / exposures) if exposures else 0.0
            engagement_rate = (engagements / exposures) if exposures else 0.0
            rows.append({
                'method': method,
                'rank_bucket': bucket,
                'exposures': exposures,
                'interactions': interactions,
                'engagements': engagements,
                'ctr': round(ctr, 4),
                'engagement_rate': round(engagement_rate, 4),
            })
        rows.sort(key=lambda row: (row['method'], row['rank_bucket']))
        return rows

    def _print_results(self, rows):
        if not rows:
            self.stdout.write('No recommendation data found for the selected window.')
            return

        self.stdout.write('\nMethod\tRank Bucket\tExposures\tInteractions\tEngagements\tCTR\tEngagement Rate')
        for row in rows:
            self.stdout.write(
                f"{row['method']}\t{row['rank_bucket']}\t{row['exposures']}\t"
                f"{row['interactions']}\t{row['engagements']}\t"
                f"{row['ctr']:.4f}\t{row['engagement_rate']:.4f}"
            )

    def _write_csv(self, rows, output_path):
        fieldnames = ['method', 'rank_bucket', 'exposures', 'interactions', 'engagements', 'ctr', 'engagement_rate']
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_rank_bucket(self, rank: Optional[int]) -> str:
        if rank is None:
            return 'unknown'
        if rank <= 3:
            return '01-03'
        if rank <= 6:
            return '04-06'
        if rank <= 10:
            return '07-10'
        return '11+'

