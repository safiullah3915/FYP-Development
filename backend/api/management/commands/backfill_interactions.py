from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import Application, Favorite, Interest
from api.recommendation_models import UserInteraction


class Command(BaseCommand):
    help = 'Backfill UserInteraction table from existing applications, favorites, and interests'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting interaction backfill...'))
        
        # Backfill from applications
        applications = Application.objects.filter(status__in=['pending', 'approved', 'rejected'])
        app_count = 0
        for app in applications:
            UserInteraction.objects.get_or_create(
                user=app.applicant,
                startup=app.startup,
                position=app.position,
                interaction_type='apply',
                defaults={
                    'weight': 3.0,
                    'created_at': app.created_at
                }
            )
            app_count += 1
        self.stdout.write(self.style.SUCCESS(f'Backfilled {app_count} application interactions'))
        
        # Backfill from favorites
        favorites = Favorite.objects.all()
        fav_count = 0
        for fav in favorites:
            UserInteraction.objects.get_or_create(
                user=fav.user,
                startup=fav.startup,
                interaction_type='favorite',
                defaults={
                    'weight': 2.5,
                    'created_at': fav.created_at
                }
            )
            fav_count += 1
        self.stdout.write(self.style.SUCCESS(f'Backfilled {fav_count} favorite interactions'))
        
        # Backfill from interests
        interests = Interest.objects.all()
        int_count = 0
        for interest in interests:
            UserInteraction.objects.get_or_create(
                user=interest.user,
                startup=interest.startup,
                interaction_type='interest',
                defaults={
                    'weight': 3.5,
                    'created_at': interest.created_at
                }
            )
            int_count += 1
        self.stdout.write(self.style.SUCCESS(f'Backfilled {int_count} interest interactions'))
        
        total = app_count + fav_count + int_count
        self.stdout.write(self.style.SUCCESS(f'Backfill complete! Total interactions: {total}'))

