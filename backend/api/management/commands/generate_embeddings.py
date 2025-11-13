from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import models
from api.models import User, Startup
from api.services.embedding_service import EmbeddingService


class Command(BaseCommand):
    help = 'Generate embeddings for users and startups using sentence-transformers'

    def add_arguments(self, parser):
        parser.add_argument(
            '--users',
            action='store_true',
            help='Generate embeddings for users',
        )
        parser.add_argument(
            '--startups',
            action='store_true',
            help='Generate embeddings for startups',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Generate embeddings for all users and startups',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of embeddings even if they already exist',
        )
        parser.add_argument(
            '--only-pending',
            action='store_true',
            help='Only process users/startups with embedding_needs_update=True or no embedding',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of users/startups to process in each batch (default: 50)',
        )

    def handle(self, *args, **options):
        embedding_service = EmbeddingService()
        
        if options['all'] or options['users']:
            self.stdout.write(self.style.SUCCESS('Generating user embeddings...'))
            
            # Build query based on options
            users_query = User.objects.filter(is_active=True)
            
            if options['only_pending']:
                # Only process users that need updates
                users_query = users_query.filter(
                    models.Q(embedding_needs_update=True) | 
                    models.Q(embedding_updated_at__isnull=True) |
                    models.Q(profile_embedding__isnull=True) |
                    models.Q(profile_embedding='')
                )
                self.stdout.write(self.style.WARNING(
                    f'Processing only pending users (embedding_needs_update=True or no embedding)'
                ))
            elif not options['force']:
                # By default, skip users that already have embeddings
                users_query = users_query.filter(
                    models.Q(profile_embedding__isnull=True) |
                    models.Q(profile_embedding='')
                )
                self.stdout.write(self.style.WARNING(
                    'Skipping users with existing embeddings. Use --force to regenerate all.'
                ))
            
            users = list(users_query)
            total = len(users)
            
            if total == 0:
                self.stdout.write(self.style.WARNING('No users to process.'))
            else:
                self.stdout.write(f'Processing {total} users...')
                
                # Process in batches
                batch_size = options['batch_size']
                results = embedding_service.generate_embeddings_batch(users, batch_size=batch_size)
                
                # Report results
                self.stdout.write(self.style.SUCCESS(
                    f'\nEmbedding generation complete!'
                ))
                self.stdout.write(f'  Success: {results["success"]}')
                self.stdout.write(f'  Failed: {results["failed"]}')
                
                if results['errors']:
                    self.stdout.write(self.style.WARNING(f'\nErrors ({len(results["errors"])}):'))
                    for error in results['errors'][:10]:  # Show first 10 errors
                        self.stdout.write(self.style.ERROR(f'  - {error}'))
                    if len(results['errors']) > 10:
                        self.stdout.write(self.style.WARNING(
                            f'  ... and {len(results["errors"]) - 10} more errors'
                        ))
        
        if options['all'] or options['startups']:
            self.stdout.write(self.style.SUCCESS('Generating startup embeddings...'))
            
            # Build query based on options (only active startups)
            startups_query = Startup.objects.filter(status='active')
            
            if options['only_pending']:
                # Only process startups that need updates
                startups_query = startups_query.filter(
                    models.Q(embedding_needs_update=True) |
                    models.Q(embedding_updated_at__isnull=True) |
                    models.Q(profile_embedding__isnull=True) |
                    models.Q(profile_embedding='')
                )
                self.stdout.write(self.style.WARNING(
                    f'Processing only pending startups (embedding_needs_update=True or no embedding)'
                ))
            elif not options['force']:
                # By default, skip startups that already have embeddings
                startups_query = startups_query.filter(
                    models.Q(profile_embedding__isnull=True) |
                    models.Q(profile_embedding='')
                )
                self.stdout.write(self.style.WARNING(
                    'Skipping startups with existing embeddings. Use --force to regenerate all.'
                ))
            
            startups = list(startups_query)
            total = len(startups)
            
            if total == 0:
                self.stdout.write(self.style.WARNING('No startups to process.'))
            else:
                self.stdout.write(f'Processing {total} startups...')
                
                # Process in batches
                batch_size = options['batch_size']
                results = embedding_service.generate_startup_embeddings_batch(startups, batch_size=batch_size)
                
                # Report results
                self.stdout.write(self.style.SUCCESS(
                    f'\nStartup embedding generation complete!'
                ))
                self.stdout.write(f'  Success: {results["success"]}')
                self.stdout.write(f'  Failed: {results["failed"]}')
                
                if results['errors']:
                    self.stdout.write(self.style.WARNING(f'\nErrors ({len(results["errors"])}):'))
                    for error in results['errors'][:10]:  # Show first 10 errors
                        self.stdout.write(self.style.ERROR(f'  - {error}'))
                    if len(results['errors']) > 10:
                        self.stdout.write(self.style.WARNING(
                            f'  ... and {len(results["errors"]) - 10} more errors'
                        ))
        
        self.stdout.write(self.style.SUCCESS('\nCommand completed!'))

