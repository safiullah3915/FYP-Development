"""
Django management command to generate ranker training dataset
Creates positive and negative pairs from explicit user feedback
"""
import os
import pandas as pd
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from api.recommendation_models import UserInteraction, RecommendationSession, StartupInteraction
from api.models import User, Startup
import random
import math
from typing import Optional


class Command(BaseCommand):
    help = 'Generate ranker training dataset from explicit feedback'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='../recommendation_service/data/ranker_train.csv',
            help='Output CSV file path'
        )
        parser.add_argument(
            '--neg-ratio',
            type=int,
            default=2,
            help='Ratio of negative to positive samples'
        )
        parser.add_argument(
            '--use-case',
            type=str,
            default='developer_startup',
            choices=['developer_startup', 'investor_startup', 'startup_developer', 'startup_investor'],
            help='Use case to generate dataset for (default: developer_startup)'
        )

    def handle(self, *args, **options):
        output_path = options['output']
        neg_ratio = options['neg_ratio']
        use_case = options['use_case']

        self.stdout.write(self.style.SUCCESS('\n=== Ranker Dataset Generation ===\n'))
        self.stdout.write(f'Use case: {use_case}')

        # Determine if this is a reverse use case
        reverse_use_cases = ['startup_developer', 'startup_investor']
        is_reverse = use_case in reverse_use_cases

        # Step 1: Load explicit positive interactions
        self.stdout.write('Loading positive interactions...')
        if is_reverse:
            positive_interactions = self.load_positive_startup_interactions(use_case)
        else:
            positive_interactions = self.load_positive_interactions(use_case)
        
        if not positive_interactions:
            self.stdout.write(self.style.ERROR('No positive interactions found!'))
            self.stdout.write('Make sure users have liked/favorited/applied to startups')
            return

        self.stdout.write(self.style.SUCCESS(f'Found {len(positive_interactions)} positive interactions'))

        # Step 2: Generate negative samples
        self.stdout.write(f'Generating negative samples (ratio: {neg_ratio}:1)...')
        negative_samples = self.generate_negative_samples(
            positive_interactions,
            neg_ratio=neg_ratio,
            is_reverse=is_reverse,
            use_case=use_case
        )
        
        self.stdout.write(self.style.SUCCESS(f'Generated {len(negative_samples)} negative samples'))

        # Step 3: Combine and extract features
        self.stdout.write('Extracting features...')
        dataset = self.create_dataset(positive_interactions, negative_samples, is_reverse)
        
        self.stdout.write(self.style.SUCCESS(f'Total samples: {len(dataset)}'))

        # Step 4: Save to CSV
        self.stdout.write('Saving dataset...')
        self.save_dataset(dataset, output_path)
        
        self.stdout.write(self.style.SUCCESS(f'\n=== Dataset saved to: {output_path} ===\n'))

    def load_positive_interactions(self, use_case='developer_startup'):
        """
        Load explicit positive feedback (like, favorite, apply, interest)
        Only from recommendations (recommendation_source='recommendation')
        """
        # Filter by user roles based on use case
        user_filter = {}
        if use_case == 'developer_startup':
            user_filter['user__role'] = 'student'
        elif use_case == 'investor_startup':
            user_filter['user__role'] = 'investor'
        
        interactions = UserInteraction.objects.filter(
            interaction_type__in=['like', 'favorite', 'apply', 'interest'],
            recommendation_source='recommendation',  # Only from recommendations
            **user_filter
        ).select_related('user', 'startup', 'recommendation_session').values(
            'user_id',
            'startup_id',
            'interaction_type',
            'weight',
            'created_at',
            'recommendation_rank',
            'recommendation_score',
            'recommendation_source',
            'recommendation_method',
            'recommendation_session_id'
        )

        return list(interactions)

    def load_positive_startup_interactions(self, use_case='startup_developer'):
        """
        Load explicit positive feedback from startup interactions (contact, apply_received)
        Only from recommendations (recommendation_source='recommendation')
        """
        # Determine target user roles
        if use_case == 'startup_developer':
            target_user_roles = ['student']
        elif use_case == 'startup_investor':
            target_user_roles = ['investor']
        else:
            target_user_roles = ['student', 'investor']
        
        interactions = StartupInteraction.objects.filter(
            interaction_type__in=['contact', 'apply_received'],
            recommendation_source='recommendation',  # Only from recommendations
            target_user__role__in=target_user_roles
        ).select_related('startup', 'target_user', 'recommendation_session').values(
            'startup_id',
            'target_user_id',
            'interaction_type',
            'weight',
            'created_at',
            'recommendation_rank',
            'recommendation_score',
            'recommendation_source',
            'recommendation_method',
            'recommendation_session_id'
        )

        # Transform to match forward format: user_id = startup_id, startup_id = target_user_id
        transformed = []
        for interaction in interactions:
            transformed.append({
                'user_id': interaction['startup_id'],  # For reverse: user_id = startup_id
                'startup_id': interaction['target_user_id'],  # For reverse: startup_id = target_user_id
                'interaction_type': interaction['interaction_type'],
                'weight': interaction['weight'],
                'created_at': interaction['created_at'],
                'recommendation_rank': interaction['recommendation_rank'],
                'recommendation_score': interaction['recommendation_score'],
                'recommendation_source': interaction['recommendation_source'],
                'recommendation_method': interaction['recommendation_method'],
                'recommendation_session_id': interaction['recommendation_session_id']
            })

        return transformed

    def generate_negative_samples(self, positive_interactions, neg_ratio=2, is_reverse=False, use_case='developer_startup'):
        """
        Generate negative samples (users who didn't interact with startups, or startups that didn't interact with users for reverse)
        
        Strategy:
        - For each user/startup with positive interactions
        - Sample items they DIDN'T interact with
        - Preferably items they may have seen (views) but didn't engage
        """
        negative_samples = []
        
        # Group positive interactions by entity (user for forward, startup for reverse)
        entity_positive_items = {}
        for interaction in positive_interactions:
            entity_id = str(interaction['user_id'])  # For reverse: this is startup_id
            item_id = str(interaction['startup_id'])  # For reverse: this is user_id
            
            if entity_id not in entity_positive_items:
                entity_positive_items[entity_id] = set()
            entity_positive_items[entity_id].add(item_id)

        # For each entity, generate negative samples
        for entity_id, positive_item_ids in entity_positive_items.items():
            if is_reverse:
                # For reverse: entity_id = startup_id, item_id = user_id
                # Get startup's viewed users (potential negatives)
                viewed_items = StartupInteraction.objects.filter(
                    startup_id=entity_id,
                    interaction_type='view'
                ).exclude(
                    target_user_id__in=positive_item_ids
                ).values_list('target_user_id', flat=True)
                
                viewed_item_ids = [str(uid) for uid in viewed_items]
                
                # High-score recommendations without interaction (hard negatives)
                recommended_candidates = self.get_high_score_recommendation_candidates_reverse(entity_id, use_case)
                candidate_negatives = [
                    cand for cand in recommended_candidates
                    if cand['user_id'] not in positive_item_ids
                ]
                
                # If not enough viewed users, sample from all users
                if len(viewed_item_ids) < (len(positive_item_ids) * neg_ratio):
                    # Determine target user roles
                    if use_case == 'startup_developer':
                        target_user_roles = ['student']
                    elif use_case == 'startup_investor':
                        target_user_roles = ['investor']
                    else:
                        target_user_roles = ['student', 'investor']
                    
                    all_users = User.objects.filter(
                        role__in=target_user_roles,
                        is_active=True
                    ).exclude(
                        id__in=positive_item_ids
                    ).values_list('id', flat=True)[:100]  # Limit for performance
                    
                    all_user_ids = [str(uid) for uid in all_users]
                    viewed_item_ids.extend(all_user_ids)
            else:
                # For forward: entity_id = user_id, item_id = startup_id
                # Get user's viewed startups (potential negatives)
                viewed_items = UserInteraction.objects.filter(
                    user_id=entity_id,
                    interaction_type='view'
                ).exclude(
                    startup_id__in=positive_item_ids
                ).values_list('startup_id', flat=True)

                viewed_item_ids = [str(sid) for sid in viewed_items]

                # High-score recommendations without interaction (hard negatives)
                recommended_candidates = self.get_high_score_recommendation_candidates(entity_id)
                candidate_negatives = [
                    cand for cand in recommended_candidates
                    if cand['startup_id'] not in positive_item_ids
                ]

                # If not enough viewed startups, sample from all startups
                if len(viewed_item_ids) < (len(positive_item_ids) * neg_ratio):
                    all_startups = Startup.objects.exclude(
                        id__in=positive_item_ids
                    ).values_list('id', flat=True)[:100]  # Limit for performance
                    
                    all_startup_ids = [str(sid) for sid in all_startups]
                    viewed_item_ids.extend(all_startup_ids)

            # Sample negative items
            num_negatives = min(
                len(positive_item_ids) * neg_ratio,
                len(viewed_item_ids) + len(candidate_negatives)
            )

            if num_negatives > 0:
                hard_negative_count = min(num_negatives, len(candidate_negatives))
                for idx in range(hard_negative_count):
                    if is_reverse:
                        negative_samples.append({
                            'user_id': entity_id,
                            'startup_id': candidate_negatives[idx]['user_id'],
                            'label': 0,
                            'recommendation_rank': candidate_negatives[idx].get('rank'),
                            'recommendation_score': candidate_negatives[idx].get('score'),
                            'recommendation_method': candidate_negatives[idx].get('method'),
                        })
                    else:
                        negative_samples.append({
                            'user_id': entity_id,
                            'startup_id': candidate_negatives[idx]['startup_id'],
                            'label': 0,
                            'recommendation_rank': candidate_negatives[idx].get('rank'),
                            'recommendation_score': candidate_negatives[idx].get('score'),
                            'recommendation_method': candidate_negatives[idx].get('method'),
                        })
                
                remaining = num_negatives - hard_negative_count
                if remaining > 0:
                    sampled_negatives = random.sample(viewed_item_ids, min(remaining, len(viewed_item_ids)))
                    for neg_item_id in sampled_negatives:
                        if neg_item_id in positive_item_ids:
                            continue
                        negative_samples.append({
                            'user_id': entity_id,
                            'startup_id': neg_item_id,
                            'label': 0,
                            'recommendation_rank': None,
                            'recommendation_score': None,
                            'recommendation_method': None,
                        })

        return negative_samples

    def get_high_score_recommendation_candidates_reverse(self, startup_id, use_case, session_limit=10, score_threshold=0.7):
        """Get high-score recommendation candidates for reverse use cases (startup → user)"""
        candidates = []
        
        # Get recommendation sessions for this startup and use case
        sessions = RecommendationSession.objects.filter(
            startup_id=startup_id,
            use_case=use_case
        ).order_by('-created_at')[:session_limit]
        
        for session in sessions:
            recommendations = session.recommendations_shown or []
            for rec in recommendations:
                if isinstance(rec, dict):
                    user_id = rec.get('user_id')
                    score = rec.get('score', 0.0)
                    rank = rec.get('rank', 999)
                    method = rec.get('method', 'unknown')
                    
                    if score >= score_threshold and user_id:
                        candidates.append({
                            'user_id': user_id,
                            'rank': rank,
                            'score': score,
                            'method': method
                        })
        
        # Remove duplicates, keep highest score
        seen = {}
        for cand in candidates:
            user_id = cand['user_id']
            if user_id not in seen or seen[user_id]['score'] < cand['score']:
                seen[user_id] = cand
        
        return list(seen.values())
    
    def get_high_score_recommendation_candidates(self, user_id, session_limit=10, score_threshold=0.7):
        """
        Fetch high-score recommendation exposures for a user to use as hard negatives
        
        Returns startups that were:
        1. Shown in recommendation sessions
        2. Had high scores (> threshold)
        3. User did NOT interact positively with
        """
        sessions = RecommendationSession.objects.filter(
            user_id=user_id,
            use_case__in=['developer_startup', 'investor_startup']
        ).order_by('-created_at')[:session_limit]
        
        # Get user's positive interactions to exclude
        positive_startup_ids = set(
            UserInteraction.objects.filter(
                user_id=user_id,
                interaction_type__in=['like', 'favorite', 'apply', 'interest']
            ).values_list('startup_id', flat=True)
        )
        positive_startup_ids = {str(sid) for sid in positive_startup_ids}
        
        candidates = []
        seen_startups = set()
        
        for session in sessions:
            recommendations = session.recommendations_shown or []
            for rec in recommendations:
                startup_id = str(rec.get('startup_id'))
                score = rec.get('score')
                
                if not startup_id or startup_id in seen_startups:
                    continue
                
                # Skip if user already interacted positively
                if startup_id in positive_startup_ids:
                    continue
                
                # Prioritize high-score recommendations (hard negatives)
                if score is not None and score > score_threshold:
                    candidates.append({
                        'startup_id': startup_id,
                        'rank': rec.get('rank'),
                        'score': score,
                        'method': rec.get('method') or session.recommendation_method,
                    })
                    seen_startups.add(startup_id)
        
        # Sort by score descending (highest scores first - these are the hardest negatives)
        candidates.sort(key=lambda c: (c['score'] is not None, c['score'] or 0), reverse=True)
        return candidates

    def create_dataset(self, positive_interactions, negative_samples, is_reverse=False):
        """
        Create full dataset with features
        
        Features:
        - model_score: For now, use interaction weight as proxy (0.0 for negatives)
        - recency_score: How recent the item is (startup for forward, user for reverse)
        - popularity_score: Views and interaction count
        - diversity_score: Set to neutral 0.5 for training (calculated at inference)
        - label: 1 for positive, 0 for negative
        """
        dataset = []

        # Get item metadata for feature extraction
        # For forward: items are startups, for reverse: items are users
        item_ids = set()
        for interaction in positive_interactions:
            item_ids.add(str(interaction['startup_id']))
        for sample in negative_samples:
            item_ids.add(sample['startup_id'])

        if is_reverse:
            # For reverse: extract user features (startup_id field contains user_ids)
            # Query user data
            users = User.objects.filter(id__in=item_ids).values(
                'id', 'date_joined', 'last_login'
            )
            item_data = {str(u['id']): u for u in users}
            # Note: For production, may need more user features
        else:
            # Query startup data
            startups = Startup.objects.filter(id__in=item_ids).values(
                'id', 'created_at', 'updated_at', 'views'
            )
            item_data = {str(s['id']): s for s in startups}

        # Process positive interactions (only from recommendations)
        for interaction in positive_interactions:
            item_id = str(interaction['startup_id'])
            item = item_data.get(item_id)
            
            if not item:
                continue

            # Use recommendation_score as model_score (teacher signal)
            recommendation_score = interaction.get('recommendation_score')
            if is_reverse:
                # For reverse: extract user features (simplified for now)
                features = self.extract_user_features(
                    item,
                    interaction_weight=interaction['weight'],
                    recommendation_score=recommendation_score
                )
            else:
                features = self.extract_features(
                    item,
                    interaction_weight=interaction['weight'],
                    recommendation_score=recommendation_score
                )
            exposure_weight = self.compute_exposure_weight(interaction.get('recommendation_rank'))
            
            # original_score is the teacher signal - what the base model predicted
            original_score = recommendation_score if recommendation_score is not None else features['model_score']
            
            dataset.append({
                'user_id': str(interaction['user_id']),
                'startup_id': item_id,
                'model_score': features['model_score'],
                'recency_score': features['recency_score'],
                'popularity_score': features['popularity_score'],
                'diversity_score': 0.5,  # Neutral for training
                'label': 1,
                'recommendation_rank': interaction.get('recommendation_rank'),
                'recommendation_score': recommendation_score,
                'original_score': original_score,  # Teacher signal for learning
                'recommendation_method': interaction.get('recommendation_method'),
                'exposure_weight': exposure_weight,
            })

        # Process negative samples
        for sample in negative_samples:
            item_id = sample['startup_id']
            item = item_data.get(item_id)
            
            if not item:
                continue

            recommendation_score = sample.get('recommendation_score')
            if is_reverse:
                features = self.extract_user_features(
                    item,
                    interaction_weight=0.0,
                    recommendation_score=recommendation_score
                )
            else:
                features = self.extract_features(
                    item,
                    interaction_weight=0.0,
                    recommendation_score=recommendation_score
                )
            exposure_weight = self.compute_exposure_weight(sample.get('recommendation_rank'))
            
            # For hard negatives, original_score is what the model predicted (high but wrong)
            # For random negatives, original_score is None/0
            original_score = recommendation_score if recommendation_score is not None else 0.0
            
            dataset.append({
                'user_id': sample['user_id'],
                'startup_id': item_id,
                'model_score': features['model_score'],
                'recency_score': features['recency_score'],
                'popularity_score': features['popularity_score'],
                'diversity_score': 0.5,
                'label': 0,
                'recommendation_rank': sample.get('recommendation_rank'),
                'recommendation_score': recommendation_score,
                'original_score': original_score,  # Teacher signal (high score but no interaction)
                'recommendation_method': sample.get('recommendation_method'),
                'exposure_weight': exposure_weight,
            })

        return dataset

    def extract_features(self, startup, interaction_weight=0.0, recommendation_score=None):
        """Extract ranking features from startup"""
        import math
        
        # Model score (prefer recommendation_score when available, fallback to interaction weight)
        if recommendation_score is not None:
            normalized_model_score = float(recommendation_score)
        else:
            normalized_model_score = min(1.0, interaction_weight / 3.5)  # 3.5 is max weight
        
        # Recency score (exponential decay)
        try:
            updated_at = startup.get('updated_at') or startup.get('created_at')
            if updated_at:
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                age_days = (now - updated_at).total_seconds() / 86400.0
                recency_score = math.exp(-age_days / 30.0)  # 30-day decay
            else:
                recency_score = 0.5
        except:
            recency_score = 0.5
        
        # Popularity score (log-scaled views)
        views = startup.get('views', 0)
        popularity_score = math.log1p(views) / math.log1p(10000)  # Normalize to max 10k views
        popularity_score = min(1.0, popularity_score)
        
        return {
            'model_score': normalized_model_score,
            'recency_score': recency_score,
            'popularity_score': popularity_score
        }

    def extract_user_features(self, user, interaction_weight=0.0, recommendation_score=None):
        """Extract ranking features from user (for reverse use cases)"""
        import math
        
        # Model score (prefer recommendation_score when available, fallback to interaction weight)
        if recommendation_score is not None:
            normalized_model_score = float(recommendation_score)
        else:
            normalized_model_score = min(1.0, interaction_weight / 3.0)  # 3.0 is max weight for reverse
        
        # Recency score (based on user activity - last login or account creation)
        try:
            activity_date = user.get('last_login') or user.get('date_joined')
            if activity_date:
                if activity_date.tzinfo is None:
                    activity_date = activity_date.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                age_days = (now - activity_date).total_seconds() / 86400.0
                recency_score = math.exp(-age_days / 30.0)  # 30-day decay
            else:
                recency_score = 0.5
        except:
            recency_score = 0.5
        
        # Popularity score (for users, use a neutral value or could use profile completeness)
        # For now, use neutral value since user popularity is less relevant for reverse recommendations
        popularity_score = 0.5
        
        return {
            'model_score': normalized_model_score,
            'recency_score': recency_score,
            'popularity_score': popularity_score
        }

    def compute_exposure_weight(self, rank: Optional[int]) -> float:
        """Calculate exposure bias weight based on rank (higher rank => lower weight)"""
        if rank and rank > 0:
            return 1.0 / math.log2(rank + 1)
        return 1.0

    def save_dataset(self, dataset, output_path):
        """Save dataset to CSV"""
        df = pd.DataFrame(dataset)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        df.to_csv(output_path, index=False)
        
        self.stdout.write(self.style.SUCCESS(f'Saved {len(df)} rows'))
        self.stdout.write(f'  Positive: {(df["label"] == 1).sum()}')
        self.stdout.write(f'  Negative: {(df["label"] == 0).sum()}')

