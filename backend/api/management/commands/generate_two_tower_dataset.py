"""
Django management command to generate training dataset for Two-Tower model
Implements smart labeling strategy using all interaction types
"""
from django.core.management.base import BaseCommand
from django.db.models import Q, Prefetch
from api.models import User, Startup, StartupTag, Position
from api.messaging_models import UserProfile
from api.recommendation_models import UserInteraction, UserOnboardingPreferences, RecommendationSession, StartupInteraction
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
            choices=['developer_startup', 'investor_startup', 'founder_startup', 'startup_developer', 'startup_investor'],
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
        
        # Determine if this is a reverse use case
        reverse_use_cases = ['startup_developer', 'startup_investor']
        is_reverse = use_case in reverse_use_cases
        
        # Determine user roles based on use case
        if use_case == 'developer_startup':
            user_roles = ['student']
        elif use_case == 'investor_startup':
            user_roles = ['investor']
        elif use_case == 'founder_startup':
            user_roles = ['entrepreneur']
        elif use_case == 'startup_developer':
            user_roles = ['student']  # Target users are developers/students
        elif use_case == 'startup_investor':
            user_roles = ['investor']  # Target users are investors
        else:
            user_roles = ['student', 'investor', 'entrepreneur']
        
        # Step 1: Load interactions
        self.stdout.write('Loading interactions...')
        if is_reverse:
            interactions = self.load_startup_interactions(user_roles, min_interactions)
        else:
            interactions = self.load_interactions(user_roles, min_interactions)
        self.stdout.write(f'  Loaded {len(interactions)} interactions')
        
        # Step 2: Generate negative samples
        self.stdout.write('Generating negative samples...')
        negative_samples_data = self.generate_negative_samples(
            interactions, negative_samples, user_roles, is_reverse=is_reverse
        )
        self.stdout.write(f'  Generated {len(negative_samples_data)} negative samples')
        
        # Step 3: Combine and extract features
        self.stdout.write('Extracting features...')
        dataset = self.extract_features(interactions, negative_samples_data, is_reverse=is_reverse)
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

    def load_startup_interactions(self, target_user_roles, min_interactions):
        """Load startup interactions for reverse use cases (startup → user)"""
        interactions_data = []
        
        # Get startups with minimum interactions
        from django.db.models import Count
        startups = Startup.objects.filter(
            status='active'
        ).annotate(
            interaction_count=Count('startup_interactions')
        ).filter(
            interaction_count__gte=min_interactions
        )
        
        startup_ids = list(startups.values_list('id', flat=True))
        self.stdout.write(f'  Found {len(startup_ids)} startups with >= {min_interactions} interactions')
        
        # Load startup interactions with prefetch
        interactions = StartupInteraction.objects.filter(
            startup_id__in=startup_ids,
            target_user__role__in=target_user_roles,
            target_user__is_active=True
        ).select_related('startup', 'target_user').order_by('created_at')
        
        for interaction in interactions:
            interactions_data.append({
                'user_id': str(interaction.startup_id),  # For reverse: user_id = startup_id
                'startup_id': str(interaction.target_user_id),  # For reverse: startup_id = user_id (target)
                'interaction_type': interaction.interaction_type,
                'weight': interaction.weight,
                'timestamp': interaction.created_at,
                'position_id': None,  # Not applicable for reverse
                'recommendation_rank': interaction.recommendation_rank,
                'recommendation_score': interaction.recommendation_score,
                'recommendation_method': interaction.recommendation_method,
            })
        
        return interactions_data

    def generate_negative_samples(self, interactions, num_samples, user_roles, is_reverse=False):
        """Generate negative samples for each user (or startup for reverse)"""
        negative_samples = []
        
        # Build interaction history
        entity_interactions = defaultdict(set)
        for interaction in interactions:
            entity_interactions[interaction['user_id']].add(interaction['startup_id'])
        
        if is_reverse:
            # For reverse: user_id = startup_id, startup_id = target_user_id
            # Get all target users (developers/investors)
            all_target_users = list(
                User.objects.filter(
                    role__in=user_roles,
                    is_active=True
                ).values_list('id', flat=True)
            )
            all_target_user_ids = [str(uid) for uid in all_target_users]
            self.stdout.write(f'  Total target users: {len(all_target_user_ids)}')
        else:
            # For forward: get all active startups
            all_startups = list(
                Startup.objects.filter(
                    status='active'
                ).values_list('id', flat=True)
            )
            all_startup_ids = [str(sid) for sid in all_startups]
            self.stdout.write(f'  Total active startups: {len(all_startup_ids)}')
        
        # Generate negative samples
        for entity_id, interacted_items in entity_interactions.items():
            # Get items entity has NOT interacted with
            if is_reverse:
                non_interacted = [
                    uid for uid in all_target_user_ids 
                    if uid not in interacted_items
                ]
            else:
                non_interacted = [
                    sid for sid in all_startup_ids 
                    if sid not in interacted_items
                ]
            
            if not non_interacted:
                continue
            
            # Hard negative candidates from recent recommendation sessions
            if is_reverse:
                # For reverse, get recommendation candidates for startup
                # entity_id = startup_id, interacted_items = set of user_ids
                recommendation_candidates = self.get_high_score_recommendation_candidates_reverse(entity_id, interacted_items)
            else:
                # For forward, get recommendation candidates for user
                # entity_id = user_id, interacted_items = set of startup_ids
                recommendation_candidates = self.get_high_score_recommendation_candidates(entity_id, interacted_items)
            
            # Sample negative examples
            n_samples = min(num_samples, len(non_interacted))
            selected = []
            
            if recommendation_candidates:
                hard_negative_pool = [cand for cand in recommendation_candidates if cand['item_id'] in non_interacted]
                hard_negative_count = min(len(hard_negative_pool), n_samples)
                selected.extend(hard_negative_pool[:hard_negative_count])
            
            remaining = n_samples - len(selected)
            if remaining > 0:
                fallback_ids = [item_id for item_id in non_interacted if item_id not in {cand['item_id'] for cand in selected}]
                if fallback_ids:
                    sampled_items = random.sample(fallback_ids, min(remaining, len(fallback_ids)))
                    for item_id in sampled_items:
                        selected.append({
                            'item_id': item_id,
                            'rank': None,
                            'score': None,
                            'method': None,
                        })
            
            for candidate in selected:
                negative_samples.append({
                    'user_id': entity_id,
                    'startup_id': candidate['item_id'],
                    'interaction_type': 'negative_sample',
                    'weight': 1.0,
                    'timestamp': datetime.now(),
                    'position_id': None,
                    'recommendation_rank': candidate.get('rank'),
                    'recommendation_score': candidate.get('score'),
                    'recommendation_method': candidate.get('method'),
                })
        
        return negative_samples

    def extract_features(self, interactions, negative_samples, is_reverse=False):
        """Extract features for all samples"""
        all_samples = interactions + negative_samples
        dataset = []
        
        # Get unique user and startup IDs
        user_ids = list(set([s['user_id'] for s in all_samples]))
        startup_ids = list(set([s['startup_id'] for s in all_samples]))
        
        if is_reverse:
            # For reverse: user_id = startup_id, startup_id = target_user_id
            # Load startup features as "user" features, user features as "item" features
            self.stdout.write(f'  Loading features for {len(user_ids)} startups (as users) and {len(startup_ids)} users (as items)...')
            entity_features = self.load_startup_features(user_ids)  # Startups as entities
            item_features = self.load_user_features(startup_ids)  # Users as items
        else:
            # For forward: normal user and startup features
            self.stdout.write(f'  Loading features for {len(user_ids)} users and {len(startup_ids)} startups...')
            entity_features = self.load_user_features(user_ids)
            item_features = self.load_startup_features(startup_ids)
        
        # Process each sample - NEVER SKIP, always include with actual values or None
        for sample in all_samples:
            entity_id = sample['user_id']  # For reverse: this is startup_id, for forward: user_id
            item_id = sample['startup_id']  # For reverse: this is user_id, for forward: startup_id
            
            # Get features - create empty dict if missing (shouldn't happen, but be safe)
            entity_feat = entity_features.get(entity_id, {})
            item_feat = item_features.get(item_id, {})
            
            # If features are missing, load them on-the-fly (shouldn't happen, but ensure we have them)
            if not entity_feat:
                if is_reverse:
                    # Load startup features
                    startup_feats = self.load_startup_features([entity_id])
                    entity_feat = startup_feats.get(entity_id, {})
                else:
                    # Load user features
                    user_feats = self.load_user_features([entity_id])
                    entity_feat = user_feats.get(entity_id, {})
            
            if not item_feat:
                if is_reverse:
                    # Load user features (as items)
                    user_feats = self.load_user_features([item_id])
                    item_feat = user_feats.get(item_id, {})
                else:
                    # Load startup features
                    startup_feats = self.load_startup_features([item_id])
                    item_feat = startup_feats.get(item_id, {})
            
            # Get label
            label = self.label_mapping.get(sample['interaction_type'], 0.0)
            # Rank-based weight: weight = base_weight * (1 / log2(rank + 1))
            # This corrects for exposure bias - interactions from deeper ranks are rarer but more meaningful
            rank_weight = self.compute_rank_weight(sample.get('recommendation_rank'))
            weighted_label = sample['weight'] * rank_weight
            
            # Merge features - use actual values from database, None for missing
            sample_data = {
                'user_id': entity_id,
                'startup_id': item_id,
                'label': label,
                'weight': weighted_label,
                'interaction_type': sample['interaction_type'],
                'timestamp': sample['timestamp'].isoformat() if hasattr(sample['timestamp'], 'isoformat') else str(sample['timestamp']),
                'recommendation_rank': sample.get('recommendation_rank'),
                'recommendation_score': sample.get('recommendation_score'),
                'rank_weight': rank_weight,
                'recommendation_method': sample.get('recommendation_method'),
            }
            
            # Add entity features (startup for reverse, user for forward)
            sample_data.update(entity_feat)
            # Add item features (user for reverse, startup for forward)
            sample_data.update(item_feat)
            
            dataset.append(sample_data)
        
        return dataset

    def load_user_features(self, user_ids):
        """Load user features from database - use actual values only"""
        features = {}
        
        users = User.objects.filter(
            id__in=user_ids
        ).select_related('profile', 'onboarding_preferences')
        
        for user in users:
            user_id = str(user.id)
            
            # Parse embedding - use actual value or None
            embedding = None
            if user.profile_embedding:
                try:
                    embedding = json.loads(user.profile_embedding)
                except:
                    embedding = None
            
            # Get preferences - use actual values only
            selected_categories = None
            selected_fields = None
            selected_tags = None
            preferred_stages = None
            preferred_engagement = None
            preferred_skills = None
            
            try:
                prefs = user.onboarding_preferences
                if prefs.selected_categories:
                    selected_categories = prefs.selected_categories
                if prefs.selected_fields:
                    selected_fields = prefs.selected_fields
                if prefs.selected_tags:
                    selected_tags = prefs.selected_tags
                if prefs.preferred_startup_stages:
                    preferred_stages = prefs.preferred_startup_stages
                if prefs.preferred_engagement_types:
                    preferred_engagement = prefs.preferred_engagement_types
                if prefs.preferred_skills:
                    preferred_skills = prefs.preferred_skills
            except UserOnboardingPreferences.DoesNotExist:
                pass  # Keep as None
            
            # Get profile skills - use actual value only
            profile_skills = None
            try:
                profile = user.profile
                if profile.skills:
                    profile_skills = profile.skills
            except UserProfile.DoesNotExist:
                pass  # Keep as None
            
            # Use actual values - None for missing, actual data for present
            features[user_id] = {
                'user_role': user.role,  # Actual role from database
                'user_embedding': json.dumps(embedding) if embedding else None,
                'user_categories': json.dumps(selected_categories) if selected_categories else None,
                'user_fields': json.dumps(selected_fields) if selected_fields else None,
                'user_tags': json.dumps(selected_tags) if selected_tags else None,
                'user_stages': json.dumps(preferred_stages) if preferred_stages else None,
                'user_engagement': json.dumps(preferred_engagement) if preferred_engagement else None,
                'user_skills': json.dumps(profile_skills) if profile_skills else None,
            }
        
        return features

    def load_startup_features(self, startup_ids):
        """Load startup features from database - use actual values only"""
        features = {}
        
        startups = Startup.objects.filter(
            id__in=startup_ids
        ).prefetch_related(
            Prefetch('tags', queryset=StartupTag.objects.all()),
            Prefetch('positions', queryset=Position.objects.filter(is_active=True))
        )
        
        for startup in startups:
            startup_id = str(startup.id)
            
            # Parse embedding - use actual value or None
            embedding = None
            if startup.profile_embedding:
                try:
                    embedding = json.loads(startup.profile_embedding)
                except:
                    embedding = None
            
            # Get tags - use actual tags only
            tags = list(startup.tags.values_list('tag', flat=True))
            
            # Get position requirements - use actual values only
            positions = startup.positions.all()
            position_titles = [p.title for p in positions] if positions.exists() else None
            position_requirements = None
            reqs = []
            for p in positions:
                if p.requirements:
                    reqs.append(p.requirements)
            if reqs:
                position_requirements = reqs
            
            # Use actual values from database - None for missing, actual data for present
            features[startup_id] = {
                'startup_type': startup.type,  # Actual type from database
                'startup_category': startup.category,  # Actual category from database
                'startup_field': startup.field,  # Actual field from database
                'startup_phase': startup.phase if startup.phase else None,
                'startup_embedding': json.dumps(embedding) if embedding else None,
                'startup_stages': json.dumps(startup.stages) if startup.stages else None,
                'startup_tags': json.dumps(tags) if tags else None,
                'startup_positions': json.dumps(position_titles) if position_titles else None,
                'startup_position_requirements': json.dumps(position_requirements) if position_requirements else None,
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

    def get_high_score_recommendation_candidates(self, user_id, interacted_startups, session_limit=10, score_threshold=0.7):
        """
        Return recommendation exposures with high scores for hard negative mining
        
        Hard negatives: Startups that were:
        1. Shown in recommendation sessions with high scores (> threshold)
        2. User did NOT interact positively with
        3. More informative than random negatives
        """
        sessions = RecommendationSession.objects.filter(
            user_id=user_id
        ).order_by('-created_at')[:session_limit]
        
        candidates = []
        seen = set(str(sid) for sid in interacted_startups)
        
        for session in sessions:
            recommendations = session.recommendations_shown or []
            for rec in recommendations:
                startup_id = str(rec.get('startup_id'))
                score = rec.get('score')
                
                if not startup_id or startup_id in seen:
                    continue
                
                # Prioritize high-score recommendations (hard negatives)
                if score is not None and score > score_threshold:
                    candidates.append({
                        'startup_id': startup_id,
                        'rank': rec.get('rank'),
                        'score': score,
                        'method': rec.get('method') or session.recommendation_method,
                    })
                    seen.add(startup_id)
        
        # Sort by score descending (highest scores first - these are the hardest negatives)
        candidates.sort(key=lambda c: (c['score'] is not None, c['score'] or 0), reverse=True)
        return candidates

    def get_high_score_recommendation_candidates_reverse(self, startup_id, interacted_users, session_limit=10, score_threshold=0.7):
        """
        Return recommendation exposures with high scores for hard negative mining (reverse use case)
        
        Hard negatives: Users (developers/investors) that were:
        1. Shown in recommendation sessions with high scores (> threshold)
        2. Startup did NOT interact positively with
        3. More informative than random negatives
        """
        sessions = RecommendationSession.objects.filter(
            startup_id=startup_id,
            use_case__in=['startup_developer', 'startup_investor']
        ).order_by('-created_at')[:session_limit]
        
        candidates = []
        seen = set(str(uid) for uid in interacted_users)
        
        for session in sessions:
            recommendations = session.recommendations_shown or []
            for rec in recommendations:
                user_id = str(rec.get('user_id'))
                score = rec.get('score')
                
                if not user_id or user_id in seen:
                    continue
                
                # Prioritize high-score recommendations (hard negatives)
                if score is not None and score > score_threshold:
                    candidates.append({
                        'item_id': user_id,
                        'rank': rec.get('rank'),
                        'score': score,
                        'method': rec.get('method') or session.recommendation_method,
                    })
                    seen.add(user_id)
        
        # Sort by score descending (highest scores first - these are the hardest negatives)
        candidates.sort(key=lambda c: (c['score'] is not None, c['score'] or 0), reverse=True)
        return candidates

    def compute_rank_weight(self, rank: Optional[int]) -> float:
        """Weight interactions based on their original rank exposure"""
        if rank and rank > 0:
            return 1.0 / math.log2(rank + 1)
        return 1.0

