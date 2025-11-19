"""
Generate realistic StartupInteraction data from existing database
Creates diverse, meaningful interactions that reflect real-world patterns
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Q, Count
from django.db import transaction
import random
import json
from datetime import timedelta
from collections import defaultdict

from api.models import User, Startup, Position, StartupTag
from api.messaging_models import UserProfile
from api.recommendation_models import StartupInteraction, RecommendationSession, UserInteraction


class Command(BaseCommand):
    help = 'Generate realistic StartupInteraction data from existing startups and users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interactions-per-startup',
            type=int,
            default=15,
            help='Target number of interactions per startup (default: 15)',
        )
        parser.add_argument(
            '--min-interactions',
            type=int,
            default=5,
            help='Minimum interactions per startup (default: 5)',
        )
        parser.add_argument(
            '--max-interactions',
            type=int,
            default=50,
            help='Maximum interactions per startup (default: 50)',
        )
        parser.add_argument(
            '--use-case',
            type=str,
            default='both',
            choices=['startup_developer', 'startup_investor', 'both'],
            help='Which use case to generate interactions for (default: both)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating records',
        )

    def handle(self, *args, **options):
        interactions_per_startup = options['interactions_per_startup']
        min_interactions = options['min_interactions']
        max_interactions = options['max_interactions']
        use_case = options['use_case']
        dry_run = options['dry_run']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be created'))

        self.stdout.write(self.style.SUCCESS('\n=== Generating Realistic StartupInteraction Data ===\n'))

        # Step 1: Load and analyze existing data
        self.stdout.write('Step 1: Analyzing existing data...')
        startups = Startup.objects.filter(status='active').select_related('owner').prefetch_related('tags', 'positions')
        students = User.objects.filter(role='student', is_active=True).select_related('profile')
        investors = User.objects.filter(role='investor', is_active=True).select_related('profile')

        startup_count = startups.count()
        student_count = students.count()
        investor_count = investors.count()

        self.stdout.write(f'  Found {startup_count} active startups')
        self.stdout.write(f'  Found {student_count} active students/developers')
        self.stdout.write(f'  Found {investor_count} active investors')

        if startup_count == 0:
            self.stdout.write(self.style.ERROR('No active startups found!'))
            return

        # Analyze startup characteristics
        startup_categories = defaultdict(int)
        startup_fields = defaultdict(int)
        startup_types = defaultdict(int)
        
        for startup in startups:
            startup_categories[startup.category] += 1
            startup_fields[startup.field.lower()] += 1
            startup_types[startup.type] += 1

        self.stdout.write(f'\nStartup distribution:')
        self.stdout.write(f'  Categories: {dict(startup_categories)}')
        self.stdout.write(f'  Types: {dict(startup_types)}')
        self.stdout.write(f'  Top fields: {dict(list(startup_fields.items())[:10])}')

        # Analyze user characteristics
        user_skills_map = {}
        for user in list(students) + list(investors):
            try:
                profile = user.profile
                if profile and profile.skills:
                    user_skills_map[user.id] = profile.skills if isinstance(profile.skills, list) else []
            except:
                user_skills_map[user.id] = []

        # Step 2: Create recommendation sessions for context
        self.stdout.write('\nStep 2: Creating recommendation sessions...')
        sessions_created = 0
        sessions_by_startup = {}

        if not dry_run:
            # Create sessions for startups that will have interactions
            collaboration_startups = [s for s in startups if s.type == 'collaboration']
            marketplace_startups = [s for s in startups if s.type == 'marketplace']

            # Create sessions for collaboration startups (developer recommendations)
            if use_case in ['startup_developer', 'both'] and student_count > 0:
                for startup in collaboration_startups[:min(len(collaboration_startups), startup_count)]:
                    session = RecommendationSession.objects.create(
                        startup_id=startup,
                        use_case='startup_developer',
                        recommendation_method='content_based',
                        model_version='content_based_v1.0',
                        recommendations_shown=[],
                        expires_at=timezone.now() + timedelta(days=7)
                    )
                    sessions_by_startup[startup.id] = {'developer': session}
                    sessions_created += 1

            # Create sessions for marketplace startups (investor recommendations)
            if use_case in ['startup_investor', 'both'] and investor_count > 0:
                for startup in marketplace_startups[:min(len(marketplace_startups), startup_count)]:
                    session = RecommendationSession.objects.create(
                        startup_id=startup,
                        use_case='startup_investor',
                        recommendation_method='content_based',
                        model_version='content_based_v1.0',
                        recommendations_shown=[],
                        expires_at=timezone.now() + timedelta(days=7)
                    )
                    if startup.id not in sessions_by_startup:
                        sessions_by_startup[startup.id] = {}
                    sessions_by_startup[startup.id]['investor'] = session
                    sessions_created += 1

            self.stdout.write(f'  Created {sessions_created} recommendation sessions')

        # Step 3: Generate interactions with realistic patterns
        self.stdout.write('\nStep 3: Generating interactions...')
        
        interactions_to_create = []
        interaction_types = ['view', 'click', 'contact', 'apply_received']
        interaction_weights = {
            'view': 0.5,
            'click': 1.0,
            'contact': 2.0,
            'apply_received': 3.0,
        }

        # Create interaction distribution (power law - some startups get more)
        startup_list = list(startups)
        random.shuffle(startup_list)

        total_interactions = 0
        interactions_by_startup = defaultdict(int)
        interactions_by_user = defaultdict(int)

        # Generate interactions for each startup
        for startup_idx, startup in enumerate(startup_list):
            # Power law distribution: earlier startups get more interactions
            base_interactions = min_interactions + int(
                (max_interactions - min_interactions) * (1.0 / (startup_idx + 1) ** 0.5)
            )
            num_interactions = random.randint(min_interactions, base_interactions)

            # Determine target users based on startup type and use case
            if startup.type == 'collaboration' and use_case in ['startup_developer', 'both']:
                target_users = list(students)
                use_case_type = 'startup_developer'
            elif startup.type == 'marketplace' and use_case in ['startup_investor', 'both']:
                target_users = list(investors)
                use_case_type = 'startup_investor'
            else:
                continue  # Skip if no matching use case

            if len(target_users) == 0:
                continue

            # Get startup characteristics for matching
            startup_field = startup.field.lower()
            startup_category = startup.category
            startup_tags = [tag.tag.lower() for tag in startup.tags.all()]

            # Score and rank users by relevance
            user_scores = []
            for user in target_users:
                score = 0.0
                
                # Field matching
                user_skills = user_skills_map.get(user.id, [])
                if isinstance(user_skills, list):
                    user_skills_lower = [s.lower() for s in user_skills]
                    # Match if user has skills related to startup field
                    if startup_field in ' '.join(user_skills_lower) or any(
                        skill in startup_field for skill in user_skills_lower
                    ):
                        score += 2.0
                    
                    # Match if user skills match startup tags
                    for tag in startup_tags:
                        if any(tag in skill or skill in tag for skill in user_skills_lower):
                            score += 1.5
                
                # Random component for diversity
                score += random.random() * 1.0
                
                user_scores.append((user, score))

            # Sort by score and select top users
            user_scores.sort(key=lambda x: x[1], reverse=True)
            selected_users = [u for u, _ in user_scores[:num_interactions * 2]]  # Get more candidates
            
            if len(selected_users) == 0:
                # Fallback: random selection
                selected_users = random.sample(target_users, min(num_interactions, len(target_users)))

            # Create interaction sequence (view -> click -> contact -> apply_received)
            interaction_sequence = ['view', 'click', 'contact', 'apply_received']
            
            for interaction_idx in range(num_interactions):
                user = random.choice(selected_users)
                
                # Determine interaction type based on sequence and probability
                if interaction_idx < num_interactions * 0.6:  # 60% are views
                    interaction_type = 'view'
                elif interaction_idx < num_interactions * 0.8:  # 20% are clicks
                    interaction_type = 'click'
                elif interaction_idx < num_interactions * 0.95:  # 15% are contacts
                    interaction_type = 'contact'
                else:  # 5% are apply_received
                    interaction_type = 'apply_received'

                # Determine if from recommendation (70% chance)
                from_recommendation = random.random() < 0.7
                recommendation_source = 'recommendation' if from_recommendation else 'organic'
                
                # Get or create session
                session = None
                if from_recommendation and startup.id in sessions_by_startup:
                    session_dict = sessions_by_startup[startup.id]
                    session = session_dict.get(use_case_type)

                # Create timestamp (spread over last 90 days)
                days_ago = random.randint(0, 90)
                interaction_time = timezone.now() - timedelta(days=days_ago)

                # Create interaction data
                interaction_data = {
                    'startup': startup,
                    'target_user': user,
                    'interaction_type': interaction_type,
                    'weight': interaction_weights[interaction_type],
                    'recommendation_source': recommendation_source,
                    'recommendation_rank': random.randint(1, 20) if from_recommendation else None,
                    'recommendation_score': round(random.uniform(0.6, 0.95), 3) if from_recommendation else None,
                    'recommendation_method': 'content_based' if from_recommendation else None,
                    'recommendation_session': session,
                    'metadata': {
                        'startup_id': str(startup.id),
                        'startup_title': startup.title,
                        'target_user_id': str(user.id),
                        'target_user_username': user.username,
                        'source': recommendation_source,
                        'generated': True,
                        'generated_at': timezone.now().isoformat()
                    },
                    'created_at': interaction_time
                }
                
                # Add value_score if field exists (for backward compatibility)
                # This will be removed in migration 0019, but handle it gracefully
                try:
                    # Check if value_score field exists in model
                    if hasattr(StartupInteraction, 'value_score'):
                        interaction_data['value_score'] = interaction_weights[interaction_type]
                except:
                    pass

                interactions_to_create.append(interaction_data)
                interactions_by_startup[startup.id] += 1
                interactions_by_user[user.id] += 1
                total_interactions += 1

        self.stdout.write(f'  Prepared {total_interactions} interactions')
        self.stdout.write(f'  Affecting {len(interactions_by_startup)} startups')
        self.stdout.write(f'  From {len(interactions_by_user)} unique users')

        # Step 4: Create interactions in bulk
        if not dry_run:
            self.stdout.write('\nStep 4: Creating interactions in database...')
            
            # Create interactions one by one (not in atomic block to avoid transaction rollback)
            created_count = 0
            skipped_count = 0
            error_count = 0
            
            for idx, interaction_data in enumerate(interactions_to_create):
                if idx % 100 == 0 and idx > 0:
                    self.stdout.write(f'  Processed {idx}/{total_interactions} interactions... (created: {created_count}, skipped: {skipped_count}, errors: {error_count})')
                
                try:
                    # Check if interaction already exists
                    exists = StartupInteraction.objects.filter(
                        startup=interaction_data['startup'],
                        target_user=interaction_data['target_user'],
                        interaction_type=interaction_data['interaction_type']
                    ).exists()
                    
                    if exists:
                        skipped_count += 1
                        continue
                    
                    # Remove value_score if it doesn't exist in model (handle migration state)
                    interaction_data_copy = interaction_data.copy()
                    if 'value_score' in interaction_data_copy:
                        # Check if model actually has this field
                        if not hasattr(StartupInteraction, 'value_score'):
                            del interaction_data_copy['value_score']
                    
                    # Create interaction
                    interaction = StartupInteraction(**interaction_data_copy)
                    interaction.save()
                    created_count += 1
                    
                except Exception as e:
                    error_count += 1
                    skipped_count += 1
                    if error_count <= 10:  # Show first 10 errors
                        self.stdout.write(self.style.ERROR(f'  Error creating interaction {idx}: {e}'))
                    # Continue processing other interactions
        else:
            created_count = total_interactions
            skipped_count = 0

        # Step 5: Update recommendation sessions with actual recommendations
        if not dry_run:
            self.stdout.write('\nStep 5: Updating recommendation sessions...')
            sessions_updated = 0
            
            for startup_id, session_dict in sessions_by_startup.items():
                startup = Startup.objects.get(id=startup_id)
                
                for use_case_type, session in session_dict.items():
                    # Get interactions for this session
                    session_interactions = StartupInteraction.objects.filter(
                        startup=startup,
                        recommendation_session=session
                    ).order_by('recommendation_rank', 'created_at')[:20]
                    
                    recommendations = []
                    for interaction in session_interactions:
                        recommendations.append({
                            'user_id': str(interaction.target_user_id),
                            'rank': interaction.recommendation_rank or 1,
                            'score': interaction.recommendation_score or 0.0,
                            'method': interaction.recommendation_method or 'content_based'
                        })
                    
                    if recommendations:
                        session.recommendations_shown = recommendations
                        session.save()
                        sessions_updated += 1
            
            self.stdout.write(f'  Updated {sessions_updated} recommendation sessions')

        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('GENERATION SUMMARY'))
        self.stdout.write(self.style.SUCCESS('='*60))
        
        if dry_run:
            self.stdout.write(f'Would create: {total_interactions} interactions')
            self.stdout.write(f'Would affect: {len(interactions_by_startup)} startups')
            self.stdout.write(f'Would involve: {len(interactions_by_user)} users')
        else:
            final_count = StartupInteraction.objects.count()
            self.stdout.write(f'Created: {created_count} interactions')
            self.stdout.write(f'Skipped (duplicates): {skipped_count}')
            self.stdout.write(f'Total StartupInteraction records: {final_count}')
            
            # Show distribution
            interaction_type_dist = defaultdict(int)
            for interaction_data in interactions_to_create[:created_count]:
                interaction_type_dist[interaction_data['interaction_type']] += 1
            
            self.stdout.write(f'\nInteraction type distribution:')
            for itype, count in interaction_type_dist.items():
                self.stdout.write(f'  {itype}: {count}')
            
            # Show startup coverage
            startups_with_interactions = StartupInteraction.objects.values('startup_id').distinct().count()
            self.stdout.write(f'\nStartups with interactions: {startups_with_interactions}/{startup_count}')
            
            # Show user engagement
            users_with_interactions = StartupInteraction.objects.values('target_user_id').distinct().count()
            if use_case == 'startup_developer':
                total_users = student_count
            elif use_case == 'startup_investor':
                total_users = investor_count
            else:
                total_users = student_count + investor_count
            self.stdout.write(f'Users with interactions: {users_with_interactions}/{total_users}')

        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))

