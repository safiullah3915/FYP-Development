"""
Django management command to generate ranker training dataset
Creates positive and negative pairs from explicit user feedback
"""
import os
import pandas as pd
from datetime import datetime, timezone
from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from api.recommendation_models import UserInteraction, RecommendationSession
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

    def handle(self, *args, **options):
        output_path = options['output']
        neg_ratio = options['neg_ratio']

        self.stdout.write(self.style.SUCCESS('\n=== Ranker Dataset Generation ===\n'))

        # Step 1: Load explicit positive interactions
        self.stdout.write('Loading positive interactions...')
        positive_interactions = self.load_positive_interactions()
        
        if not positive_interactions:
            self.stdout.write(self.style.ERROR('No positive interactions found!'))
            self.stdout.write('Make sure users have liked/favorited/applied to startups')
            return

        self.stdout.write(self.style.SUCCESS(f'Found {len(positive_interactions)} positive interactions'))

        # Step 2: Generate negative samples
        self.stdout.write(f'Generating negative samples (ratio: {neg_ratio}:1)...')
        negative_samples = self.generate_negative_samples(
            positive_interactions,
            neg_ratio=neg_ratio
        )
        
        self.stdout.write(self.style.SUCCESS(f'Generated {len(negative_samples)} negative samples'))

        # Step 3: Combine and extract features
        self.stdout.write('Extracting features...')
        dataset = self.create_dataset(positive_interactions, negative_samples)
        
        self.stdout.write(self.style.SUCCESS(f'Total samples: {len(dataset)}'))

        # Step 4: Save to CSV
        self.stdout.write('Saving dataset...')
        self.save_dataset(dataset, output_path)
        
        self.stdout.write(self.style.SUCCESS(f'\n=== Dataset saved to: {output_path} ===\n'))

    def load_positive_interactions(self):
        """Load explicit positive feedback (like, favorite, apply, interest)"""
        interactions = UserInteraction.objects.filter(
            interaction_type__in=['like', 'favorite', 'apply', 'interest']
        ).select_related('user', 'startup').values(
            'user_id',
            'startup_id',
            'interaction_type',
            'weight',
            'created_at',
            'recommendation_rank',
            'recommendation_score',
            'recommendation_source',
            'recommendation_method'
        )

        return list(interactions)

    def generate_negative_samples(self, positive_interactions, neg_ratio=2):
        """
        Generate negative samples (users who didn't interact with startups)
        
        Strategy:
        - For each user with positive interactions
        - Sample startups they DIDN'T interact with
        - Preferably startups they may have seen (views) but didn't engage
        """
        negative_samples = []
        
        # Group positive interactions by user
        user_positive_startups = {}
        for interaction in positive_interactions:
            user_id = str(interaction['user_id'])
            startup_id = str(interaction['startup_id'])
            
            if user_id not in user_positive_startups:
                user_positive_startups[user_id] = set()
            user_positive_startups[user_id].add(startup_id)

        # For each user, generate negative samples
        for user_id, positive_startup_ids in user_positive_startups.items():
            # Get user's viewed startups (potential negatives)
            viewed_startups = UserInteraction.objects.filter(
                user_id=user_id,
                interaction_type='view'
            ).exclude(
                startup_id__in=positive_startup_ids
            ).values_list('startup_id', flat=True)

            viewed_startup_ids = [str(sid) for sid in viewed_startups]

            # High-score recommendations without interaction (hard negatives)
            candidate_negatives = []
            seen_candidates = set()
            recommended_candidates = self.get_high_score_recommendation_candidates(user_id)
            for cand in recommended_candidates:
                sid = cand['startup_id']
                if sid in positive_startup_ids or sid in seen_candidates:
                    continue
                candidate_negatives.append(cand)
                seen_candidates.add(sid)

            # If not enough viewed startups, sample from all startups
            if len(viewed_startup_ids) < (len(positive_startup_ids) * neg_ratio):
                all_startups = Startup.objects.exclude(
                    id__in=positive_startup_ids
                ).values_list('id', flat=True)[:100]  # Limit for performance
                
                all_startup_ids = [str(sid) for sid in all_startups]
                viewed_startup_ids.extend(all_startup_ids)

            # Sample negative startups
            num_negatives = min(
                len(positive_startup_ids) * neg_ratio,
                len(viewed_startup_ids) + len(candidate_negatives)
            )

            if num_negatives > 0:
                hard_negative_count = min(num_negatives, len(candidate_negatives))
                for idx in range(hard_negative_count):
                    negative_samples.append({
                        'user_id': user_id,
                        'startup_id': candidate_negatives[idx]['startup_id'],
                        'label': 0,
                        'recommendation_rank': candidate_negatives[idx].get('rank'),
                        'recommendation_score': candidate_negatives[idx].get('score'),
                        'recommendation_method': candidate_negatives[idx].get('method'),
                    })
                
                remaining = num_negatives - hard_negative_count
                if remaining > 0:
                    sampled_negatives = random.sample(viewed_startup_ids, min(remaining, len(viewed_startup_ids)))
                    for neg_startup_id in sampled_negatives:
                        if neg_startup_id in positive_startup_ids:
                            continue
                        negative_samples.append({
                            'user_id': user_id,
                            'startup_id': neg_startup_id,
                            'label': 0,
                            'recommendation_rank': None,
                            'recommendation_score': None,
                            'recommendation_method': None,
                        })

        return negative_samples

    def get_high_score_recommendation_candidates(self, user_id, session_limit=5):
        """Fetch high-score recommendation exposures for a user to use as hard negatives"""
        sessions = RecommendationSession.objects.filter(
            user_id=user_id
        ).order_by('-created_at')[:session_limit]
        
        candidates = []
        for session in sessions:
            recommendations = session.recommendations_shown or []
            for rec in recommendations:
                startup_id = rec.get('startup_id')
                if not startup_id:
                    continue
                candidates.append({
                    'startup_id': str(startup_id),
                    'rank': rec.get('rank'),
                    'score': rec.get('score'),
                    'method': rec.get('method') or session.recommendation_method,
                })
        
        candidates.sort(key=lambda c: (c['score'] is not None, c['score']), reverse=True)
        return candidates

    def create_dataset(self, positive_interactions, negative_samples):
        """
        Create full dataset with features
        
        Features:
        - model_score: For now, use interaction weight as proxy (0.0 for negatives)
        - recency_score: How recent the startup is
        - popularity_score: Views and interaction count
        - diversity_score: Set to neutral 0.5 for training (calculated at inference)
        - label: 1 for positive, 0 for negative
        """
        dataset = []

        # Get startup metadata for feature extraction
        startup_ids = set()
        for interaction in positive_interactions:
            startup_ids.add(str(interaction['startup_id']))
        for sample in negative_samples:
            startup_ids.add(sample['startup_id'])

        # Query startup data
        startups = Startup.objects.filter(id__in=startup_ids).values(
            'id', 'created_at', 'updated_at', 'views'
        )
        startup_data = {str(s['id']): s for s in startups}

        # Process positive interactions
        for interaction in positive_interactions:
            startup_id = str(interaction['startup_id'])
            startup = startup_data.get(startup_id)
            
            if not startup:
                continue

            features = self.extract_features(
                startup,
                interaction_weight=interaction['weight'],
                recommendation_score=interaction.get('recommendation_score')
            )
            exposure_weight = self.compute_exposure_weight(interaction.get('recommendation_rank'))
            
            dataset.append({
                'user_id': str(interaction['user_id']),
                'startup_id': startup_id,
                'model_score': features['model_score'],
                'recency_score': features['recency_score'],
                'popularity_score': features['popularity_score'],
                'diversity_score': 0.5,  # Neutral for training
                'label': 1,
                'recommendation_rank': interaction.get('recommendation_rank'),
                'recommendation_score': interaction.get('recommendation_score'),
                'original_score': interaction.get('recommendation_score'),
                'recommendation_method': interaction.get('recommendation_method'),
                'exposure_weight': exposure_weight,
            })

        # Process negative samples
        for sample in negative_samples:
            startup_id = sample['startup_id']
            startup = startup_data.get(startup_id)
            
            if not startup:
                continue

            features = self.extract_features(
                startup,
                interaction_weight=0.0,
                recommendation_score=sample.get('recommendation_score')
            )
            exposure_weight = self.compute_exposure_weight(sample.get('recommendation_rank'))
            
            dataset.append({
                'user_id': sample['user_id'],
                'startup_id': startup_id,
                'model_score': features['model_score'],
                'recency_score': features['recency_score'],
                'popularity_score': features['popularity_score'],
                'diversity_score': 0.5,
                'label': 0,
                'recommendation_rank': sample.get('recommendation_rank'),
                'recommendation_score': sample.get('recommendation_score'),
                'original_score': sample.get('recommendation_score'),
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

