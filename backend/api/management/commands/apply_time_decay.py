"""
Management command to apply time decay to all trending metrics
Should be run hourly via cron job or scheduler
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import Startup
from api.recommendation_models import StartupTrendingMetrics
from api.services.trending_metrics_service import TrendingMetricsService


class Command(BaseCommand):
    help = 'Apply time decay to all trending metrics (run hourly)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--startup-id',
            type=str,
            help='Apply decay to specific startup only (optional)',
        )
    
    def handle(self, *args, **options):
        startup_id = options.get('startup_id')
        
        if startup_id:
            self.stdout.write(self.style.SUCCESS(f'Applying time decay to startup {startup_id}...'))
            service = TrendingMetricsService()
            service.apply_time_decay_single(startup_id)
            self.stdout.write(self.style.SUCCESS(f'✅ Applied decay to startup {startup_id}'))
        else:
            self.stdout.write(self.style.SUCCESS('Applying time decay to all startups...'))
            
            # Get all startups with metrics
            metrics_qs = StartupTrendingMetrics.objects.select_related('startup').all()
            total = metrics_qs.count()
            processed = 0
            
            service = TrendingMetricsService()
            
            for metrics in metrics_qs:
                try:
                    service.apply_time_decay_single(str(metrics.startup.id))
                    processed += 1
                    
                    if processed % 10 == 0:
                        self.stdout.write(f'  Processed {processed}/{total}...')
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f'  ⚠️ Failed for startup {metrics.startup.id}: {str(e)}')
                    )
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ Applied decay to {processed}/{total} startups')
            )

