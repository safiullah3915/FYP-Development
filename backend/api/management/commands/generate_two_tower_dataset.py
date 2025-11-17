"""
Django management command to generate training dataset for Two-Tower model
Implements smart labeling strategy using all interaction types
"""
from django.core.management.base import BaseCommand
from django.db.models import Q, Prefetch
from api.models import User, Startup, StartupTag, Position
from api.messaging_models import UserProfile
from api.recommendation_models import UserInteraction, UserOnboardingPreferences, RecommendationSession
import json
import random
import csv
from collections import defaultdict
from datetime import datetime
import os
import math
from typing import Optional


class Command(BaseCommand):
    help = 'Generate training dataset for Two-Tower model with smart labeling'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='two_tower_dataset.csv',
            help='Output CSV file path (default: two_tower_dataset.csv)',
        )
        parser.add_argument(
            '--negative-samples',
            type=int,
            default=2,
            help='Number of negative samples per user (default: 2)',
        )
        parser.add_argument(
            '--use-case',
            type=str,
            default='developer_startup',
            choices=['developer_startup', 'investor_startup', 'founder_startup'],
            help='Use case to generate dataset for (default: developer_startup)',
        )
        parser.add_argument(
            '--min-interactions',
            type=int,
            default=1,
            help='Minimum interactions per user to include (default: 1)',
        )

    def handle(self, *args, **options):
        output_path = options['output']
        negative_samples = options['negative_samples']
        use_case = options['use_case']
        min_interactions = options['min_interactions']

        self.stdout.write(self.style.SUCCESS('Starting Two-Tower dataset generation...'))
        self.stdout.write(f'  Use case: {use_case}')
        self.stdout.write(f'  Negative samples per user: {negative_samples}')
        self.stdout.write(f'  Min interactions per user: {min_interactions}')
        
        # Define labeling strategy
        self.label_mapping = {
            'apply': 1.0,
            'interest': 1.0,
            'favorite': 0.9,
            'like': 0.8,
            'click': 0.6,
            'view': 0.4,
            'dislike': 0.0,
            'negative_sample': 0.0,
        }
        
        # Determine user roles based on use case
        if use_case == 'developer_startup':
            user_roles = ['student']
        elif use_case == 'investor_startup':
            user_roles = ['investor']
        elif use_case == 'founder_startup':
            user_roles = ['entrepreneur']
        else:
            user_roles = ['student', 'investor', 'entrepreneur']
        
        # Step 1: Load interactions
        self.stdout.write('Loading interactions...')
        interactions = self.load_interactions(user_roles, min_interactions)
        self.stdout.write(f'  Loaded {len(interactions)} interactions')
        
        # Step 2: Generate negative samples
        self.stdout.write('Generating negative samples...')
        negative_samples_data = self.generate_negative_samples(
            interactions, negative_samples, user_roles
        )
        self.stdout.write(f'  Generated {len(negative_samples_data)} negative samples')
        
        # Step 3: Combine and extract features
        self.stdout.write('Extracting features...')
        dataset = self.extract_features(interactions, negative_samples_data)
        self.stdout.write(f'  Extracted features for {len(dataset)} samples')
        
        # Step 4: Write to CSV
        self.stdout.write(f'Writing to {output_path}...')
        self.write_csv(dataset, output_path)
        
        # Step 5: Print statistics
        self.print_statistics(dataset)
        
        self.stdout.write(self.style.SUCCESS(f'\nDataset successfully generated: {output_path}'))

    def load_interactions(self, user_roles, min_interactions):
        """Load all interactions from database"""
        interactions_data = []
        
        # Get users with minimum interactions
        from django.db.models import Count
        users = User.objects.filter(
            role__in=user_roles,
            is_active=True
        ).annotate(
            interaction_count=Count('interactions')
        ).filter(
            interaction_count__gte=min_interactions
        )
        
        user_ids = list(users.values_list('id', flat=True))
        self.stdout.write(f'  Found {len(user_ids)} users with >= {min_interactions} interactions')
        
        # Load interactions with prefetch
        interactions = UserInteraction.objects.filter(
            user_id__in=user_ids
        ).select_related('user', 'startup', 'position').order_by('created_at')
        
        for interaction in interactions:
            interactions_data.append({
                'user_id': str(interaction.user_id),
                'startup_id': str(interaction.startup_id),
                'interaction_type': interaction.interaction_type,
                'weight': interaction.weight,
                'timestamp': interaction.created_at,
                'position_id': str(interaction.position_id) if interaction.position_id else None,
                'recommendation_rank': interaction.recommendation_rank,
                'recommendation_score': interaction.recommendation_score,
                'recommendation_method': interaction.recommendation_method,
            })
        
        return interactions_data

    def generate_negative_samples(self, interactions, num_samples, user_roles):
        """Generate negative samples for each user"""
        negative_samples = []
        
        # Build user interaction history
        user_interactions = defaultdict(set)
        for interaction in interactions:
            user_interactions[interaction['user_id']].add(interaction['startup_id'])
        
        # Get all active startups
        all_startups = list(
            Startup.objects.filter(
                status='active'
            ).values_list('id', flat=True)
        )
        all_startup_ids = [str(sid) for sid in all_startups]
        
        self.stdout.write(f'  Total active startups: {len(all_startup_ids)}')
        
        # Generate negative samples
        for user_id, interacted_startups in user_interactions.items():
            # Get startups user has NOT interacted with
            non_interacted = [
                sid for sid in all_startup_ids 
                if sid not in interacted_startups
            ]
            
            if not non_interacted:
                continue
            
            # Hard negative candidates from recent recommendation sessions
            recommendation_candidates = self.get_high_score_recommendation_candidates(user_id, interacted_startups)
            
            # Sample negative examples
            n_samples = min(num_samples, len(non_interacted))
            selected = []
            
            if recommendation_candidates:
                hard_negative_pool = [cand for cand in recommendation_candidates if cand['startup_id'] in non_interacted]
                hard_negative_count = min(len(hard_negative_pool), n_samples)
                selected.extend(hard_negative_pool[:hard_negative_count])
            
            remaining = n_samples - len(selected)
            if remaining > 0:
                fallback_ids = [sid for sid in non_interacted if sid not in {cand['startup_id'] for cand in selected}]
                if fallback_ids:
                    sampled_startups = random.sample(fallback_ids, min(remaining, len(fallback_ids)))
                    for startup_id in sampled_startups:
                        selected.append({
                            'startup_id': startup_id,
                            'rank': None,
                            'score': None,
                            'method': None,
                        })
            
            for candidate in selected:
                negative_samples.append({
                    'user_id': user_id,
                    'startup_id': candidate['startup_id'],
                    'interaction_type': 'negative_sample',
                    'weight': 1.0,
                    'timestamp': datetime.now(),
                    'position_id': None,
                    'recommendation_rank': candidate.get('rank'),
                    'recommendation_score': candidate.get('score'),
                    'recommendation_method': candidate.get('method'),
                })
        
        return negative_samples

    def extract_features(self, interactions, negative_samples):
        """Extract features for all samples"""
        all_samples = interactions + negative_samples
        dataset = []
        
        # Get unique user and startup IDs
        user_ids = list(set([s['user_id'] for s in all_samples]))
        startup_ids = list(set([s['startup_id'] for s in all_samples]))
        
        self.stdout.write(f'  Loading features for {len(user_ids)} users and {len(startup_ids)} startups...')
        
        # Load user features
        user_features = self.load_user_features(user_ids)
        
        # Load startup features
        startup_features = self.load_startup_features(startup_ids)
        
        # Process each sample
        skipped = 0
        for sample in all_samples:
            user_id = sample['user_id']
            startup_id = sample['startup_id']
            
            # Skip if features not available
            if user_id not in user_features:
                skipped += 1
                continue
            if startup_id not in startup_features:
                skipped += 1
                continue
            
            # Get label
            label = self.label_mapping.get(sample['interaction_type'], 0.0)
            rank_weight = self.compute_rank_weight(sample.get('recommendation_rank'))
            weighted_label = sample['weight'] * rank_weight
            
            dataset.append({
                'user_id': user_id,
                'startup_id': startup_id,
                'label': label,
                'weight': weighted_label,
                'interaction_type': sample['interaction_type'],
                'timestamp': sample['timestamp'].isoformat() if hasattr(sample['timestamp'], 'isoformat') else str(sample['timestamp']),
                'recommendation_rank': sample.get('recommendation_rank'),
                'recommendation_score': sample.get('recommendation_score'),
                'rank_weight': rank_weight,
                'recommendation_method': sample.get('recommendation_method'),
                **user_features[user_id],
                **startup_features[startup_id],
            })
        
        if skipped > 0:
            self.stdout.write(f'  Skipped {skipped} samples due to missing features')
        
        return dataset

    def load_user_features(self, user_ids):
        """Load user features from database"""
        features = {}
        
        users = User.objects.filter(
            id__in=user_ids
        ).select_related('profile', 'onboarding_preferences')
        
        for user in users:
            user_id = str(user.id)
            
            # Parse embedding
            embedding = None
            if user.profile_embedding:
                try:
                    embedding = json.loads(user.profile_embedding)
                except:
                    embedding = None
            
            # Get preferences
            preferences = {}
            try:
                prefs = user.onboarding_preferences
                preferences = {
                    'selected_categories': prefs.selected_categories or [],
                    'selected_fields': prefs.selected_fields or [],
                    'selected_tags': prefs.selected_tags or [],
                    'preferred_stages': prefs.preferred_startup_stages or [],
                    'preferred_engagement': prefs.preferred_engagement_types or [],
                    'preferred_skills': prefs.preferred_skills or [],
                }
            except UserOnboardingPreferences.DoesNotExist:
                preferences = {
                    'selected_categories': [],
                    'selected_fields': [],
                    'selected_tags': [],
                    'preferred_stages': [],
                    'preferred_engagement': [],
                    'preferred_skills': [],
                }
            
            # Get profile skills
            profile_skills = []
            try:
                profile = user.profile
                profile_skills = profile.skills or []
            except UserProfile.DoesNotExist:
                profile_skills = []
            
            features[user_id] = {
                'user_role': user.role,
                'user_embedding': json.dumps(embedding) if embedding else None,
                'user_categories': json.dumps(preferences['selected_categories']),
                'user_fields': json.dumps(preferences['selected_fields']),
                'user_tags': json.dumps(preferences['selected_tags']),
                'user_stages': json.dumps(preferences['preferred_stages']),
                'user_engagement': json.dumps(preferences['preferred_engagement']),
                'user_skills': json.dumps(profile_skills),
            }
        
        return features

    def load_startup_features(self, startup_ids):
        """Load startup features from database"""
        features = {}
        
        startups = Startup.objects.filter(
            id__in=startup_ids
        ).prefetch_related(
            Prefetch('tags', queryset=StartupTag.objects.all()),
            Prefetch('positions', queryset=Position.objects.filter(is_active=True))
        )
        
        for startup in startups:
            startup_id = str(startup.id)
            
            # Parse embedding
            embedding = None
            if startup.profile_embedding:
                try:
                    embedding = json.loads(startup.profile_embedding)
                except:
                    embedding = None
            
            # Get tags
            tags = list(startup.tags.values_list('tag', flat=True))
            
            # Get position requirements
            positions = startup.positions.all()
            position_titles = [p.title for p in positions]
            position_requirements = []
            for p in positions:
                if p.requirements:
                    position_requirements.append(p.requirements)
            
            features[startup_id] = {
                'startup_type': startup.type,
                'startup_category': startup.category,
                'startup_field': startup.field,
                'startup_phase': startup.phase or '',
                'startup_embedding': json.dumps(embedding) if embedding else None,
                'startup_stages': json.dumps(startup.stages or []),
                'startup_tags': json.dumps(tags),
                'startup_positions': json.dumps(position_titles),
                'startup_position_requirements': json.dumps(position_requirements),
            }
        
        return features

    def write_csv(self, dataset, output_path):
        """Write dataset to CSV file"""
        if not dataset:
            self.stdout.write(self.style.ERROR('No data to write'))
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define CSV columns
        columns = [
            'user_id', 'startup_id', 'label', 'weight', 'interaction_type', 'timestamp',
            'recommendation_rank', 'recommendation_score', 'rank_weight', 'recommendation_method',
            'user_role', 'user_embedding', 'user_categories', 'user_fields', 
            'user_tags', 'user_stages', 'user_engagement', 'user_skills',
            'startup_type', 'startup_category', 'startup_field', 'startup_phase',
            'startup_embedding', 'startup_stages', 'startup_tags', 
            'startup_positions', 'startup_position_requirements',
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(dataset)

    def print_statistics(self, dataset):
        """Print dataset statistics"""
        if not dataset:
            return
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('DATASET STATISTICS'))
        self.stdout.write('='*60)
        
        total_samples = len(dataset)
        self.stdout.write(f'\nTotal samples: {total_samples}')
        
        # Label distribution
        label_counts = defaultdict(int)
        interaction_counts = defaultdict(int)
        for sample in dataset:
            label_counts[sample['label']] += 1
            interaction_counts[sample['interaction_type']] += 1
        
        self.stdout.write('\nLabel Distribution:')
        for label in sorted(label_counts.keys(), reverse=True):
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            self.stdout.write(f'  label={label:.1f}: {count} ({percentage:.1f}%)')
        
        self.stdout.write('\nInteraction Type Distribution:')
        for itype in sorted(interaction_counts.keys()):
            count = interaction_counts[itype]
            percentage = (count / total_samples) * 100
            self.stdout.write(f'  {itype}: {count} ({percentage:.1f}%)')
        
        # Role distribution
        role_counts = defaultdict(int)
        for sample in dataset:
            role_counts[sample['user_role']] += 1
        
        self.stdout.write('\nUser Role Distribution:')
        for role, count in role_counts.items():
            percentage = (count / total_samples) * 100
            self.stdout.write(f'  {role}: {count} ({percentage:.1f}%)')
        
        # Startup type distribution
        type_counts = defaultdict(int)
        for sample in dataset:
            type_counts[sample['startup_type']] += 1
        
        self.stdout.write('\nStartup Type Distribution:')
        for stype, count in type_counts.items():
            percentage = (count / total_samples) * 100
            self.stdout.write(f'  {stype}: {count} ({percentage:.1f}%)')
        
        # Check for missing embeddings
        missing_user_emb = sum(1 for s in dataset if not s['user_embedding'] or s['user_embedding'] == 'null')
        missing_startup_emb = sum(1 for s in dataset if not s['startup_embedding'] or s['startup_embedding'] == 'null')
        
        self.stdout.write('\nEmbedding Coverage:')
        self.stdout.write(f'  Users with embeddings: {total_samples - missing_user_emb} / {total_samples} ({((total_samples - missing_user_emb) / total_samples * 100):.1f}%)')
        self.stdout.write(f'  Startups with embeddings: {total_samples - missing_startup_emb} / {total_samples} ({((total_samples - missing_startup_emb) / total_samples * 100):.1f}%)')
        
        self.stdout.write('\n' + '='*60 + '\n')

    def get_high_score_recommendation_candidates(self, user_id, interacted_startups, session_limit=5):
        """Return recommendation exposures with high scores for hard negative mining"""
        sessions = RecommendationSession.objects.filter(
            user_id=user_id
        ).order_by('-created_at')[:session_limit]
        
        candidates = []
        seen = set(interacted_startups)
        for session in sessions:
            for rec in session.recommendations_shown or []:
                startup_id = rec.get('startup_id')
                if not startup_id:
                    continue
                sid = str(startup_id)
                if sid in seen:
                    continue
                candidates.append({
                    'startup_id': sid,
                    'rank': rec.get('rank'),
                    'score': rec.get('score'),
                    'method': rec.get('method') or session.recommendation_method,
                })
        candidates.sort(key=lambda c: (c['score'] is not None, c['score']), reverse=True)
        return candidates

    def compute_rank_weight(self, rank: Optional[int]) -> float:
        """Weight interactions based on their original rank exposure"""
        if rank and rank > 0:
            return 1.0 / math.log2(rank + 1)
        return 1.0

