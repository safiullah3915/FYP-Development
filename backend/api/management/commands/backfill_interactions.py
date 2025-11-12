from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import IntegrityError
from api.models import Application, Favorite, Interest
from api.recommendation_models import UserInteraction


class Command(BaseCommand):
    help = 'Backfill UserInteraction table from existing applications, favorites, and interests'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating records',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be created'))
        
        self.stdout.write(self.style.SUCCESS('Starting interaction backfill...'))
        
        # Count existing records
        existing_count = UserInteraction.objects.count()
        self.stdout.write(f'Existing UserInteraction records: {existing_count}')
        
        # Backfill from applications (exclude withdrawn)
        applications = Application.objects.exclude(status='withdrawn').select_related('applicant', 'startup', 'position')
        app_total = applications.count()
        app_created = 0
        app_existing = 0
        app_errors = 0
        
        self.stdout.write(f'\nProcessing {app_total} applications...')
        for idx, app in enumerate(applications, 1):
            if idx % 50 == 0:
                self.stdout.write(f'  Processed {idx}/{app_total} applications...')
            
            try:
                if dry_run:
                    # Check if would be created
                    exists = UserInteraction.objects.filter(
                        user=app.applicant,
                        startup=app.startup,
                        interaction_type='apply'
                    ).exists()
                    if not exists:
                        app_created += 1
                    else:
                        app_existing += 1
                else:
                    interaction, created = UserInteraction.objects.get_or_create(
                        user=app.applicant,
                        startup=app.startup,
                        interaction_type='apply',
                        defaults={
                            'position': app.position,
                            'weight': 3.0,
                            'created_at': app.created_at
                        }
                    )
                    if created:
                        app_created += 1
                    else:
                        app_existing += 1
            except IntegrityError as e:
                app_errors += 1
                self.stdout.write(self.style.ERROR(f'  Error processing application {app.id}: {str(e)}'))
            except Exception as e:
                app_errors += 1
                self.stdout.write(self.style.ERROR(f'  Unexpected error processing application {app.id}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(
            f'Applications: {app_created} created, {app_existing} existing, {app_errors} errors'
        ))
        
        # Backfill from favorites
        favorites = Favorite.objects.all().select_related('user', 'startup')
        fav_total = favorites.count()
        fav_created = 0
        fav_existing = 0
        fav_errors = 0
        
        self.stdout.write(f'\nProcessing {fav_total} favorites...')
        for idx, fav in enumerate(favorites, 1):
            if idx % 50 == 0:
                self.stdout.write(f'  Processed {idx}/{fav_total} favorites...')
            
            try:
                if dry_run:
                    exists = UserInteraction.objects.filter(
                        user=fav.user,
                        startup=fav.startup,
                        interaction_type='favorite'
                    ).exists()
                    if not exists:
                        fav_created += 1
                    else:
                        fav_existing += 1
                else:
                    interaction, created = UserInteraction.objects.get_or_create(
                        user=fav.user,
                        startup=fav.startup,
                        interaction_type='favorite',
                        defaults={
                            'weight': 2.5,
                            'created_at': fav.created_at
                        }
                    )
                    if created:
                        fav_created += 1
                    else:
                        fav_existing += 1
            except IntegrityError as e:
                fav_errors += 1
                self.stdout.write(self.style.ERROR(f'  Error processing favorite {fav.id}: {str(e)}'))
            except Exception as e:
                fav_errors += 1
                self.stdout.write(self.style.ERROR(f'  Unexpected error processing favorite {fav.id}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(
            f'Favorites: {fav_created} created, {fav_existing} existing, {fav_errors} errors'
        ))
        
        # Backfill from interests
        interests = Interest.objects.all().select_related('user', 'startup')
        int_total = interests.count()
        int_created = 0
        int_existing = 0
        int_errors = 0
        
        self.stdout.write(f'\nProcessing {int_total} interests...')
        for idx, interest in enumerate(interests, 1):
            if idx % 50 == 0:
                self.stdout.write(f'  Processed {idx}/{int_total} interests...')
            
            try:
                if dry_run:
                    exists = UserInteraction.objects.filter(
                        user=interest.user,
                        startup=interest.startup,
                        interaction_type='interest'
                    ).exists()
                    if not exists:
                        int_created += 1
                    else:
                        int_existing += 1
                else:
                    interaction, created = UserInteraction.objects.get_or_create(
                        user=interest.user,
                        startup=interest.startup,
                        interaction_type='interest',
                        defaults={
                            'weight': 3.5,
                            'created_at': interest.created_at
                        }
                    )
                    if created:
                        int_created += 1
                    else:
                        int_existing += 1
            except IntegrityError as e:
                int_errors += 1
                self.stdout.write(self.style.ERROR(f'  Error processing interest {interest.id}: {str(e)}'))
            except Exception as e:
                int_errors += 1
                self.stdout.write(self.style.ERROR(f'  Unexpected error processing interest {interest.id}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(
            f'Interests: {int_created} created, {int_existing} existing, {int_errors} errors'
        ))
        
        # Summary
        total_created = app_created + fav_created + int_created
        total_existing = app_existing + fav_existing + int_existing
        total_errors = app_errors + fav_errors + int_errors
        
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        if dry_run:
            self.stdout.write(self.style.SUCCESS('DRY RUN SUMMARY:'))
            self.stdout.write(f'  Would create: {total_created} interactions')
            self.stdout.write(f'  Already exist: {total_existing} interactions')
        else:
            self.stdout.write(self.style.SUCCESS('BACKFILL SUMMARY:'))
            self.stdout.write(f'  Created: {total_created} interactions')
            self.stdout.write(f'  Already existed: {total_existing} interactions')
            self.stdout.write(f'  Errors: {total_errors}')
            final_count = UserInteraction.objects.count()
            self.stdout.write(f'  Total UserInteraction records: {final_count}')
        self.stdout.write(self.style.SUCCESS('='*50))

