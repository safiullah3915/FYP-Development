"""
Django management command to generate ALS training dataset
Creates sparse user-item interaction matrix for collaborative filtering
"""
import os
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from django.core.management.base import BaseCommand
from django.db.models import Count
from api.recommendation_models import UserInteraction, StartupInteraction
from api.models import User, Startup


class Command(BaseCommand):
    help = 'Generate ALS training dataset from UserInteraction table'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            default='../recommendation_service/data',
            help='Output directory for dataset files'
        )
        parser.add_argument(
            '--min-interactions',
            type=int,
            default=1,
            help='Minimum interactions required for user/startup inclusion'
        )
        parser.add_argument(
            '--use-case',
            type=str,
            default='developer_startup',
            choices=['developer_startup', 'investor_startup', 'startup_developer', 'startup_investor'],
            help='Use case to generate dataset for (default: developer_startup)'
        )

    def handle(self, *args, **options):
        output_dir = options['output_dir']
        min_interactions = options['min_interactions']
        use_case = options['use_case']

        self.stdout.write(self.style.SUCCESS('\n=== ALS Dataset Generation ===\n'))
        self.stdout.write(f'Use case: {use_case}')

        # Determine if this is a reverse use case
        reverse_use_cases = ['startup_developer', 'startup_investor']
        is_reverse = use_case in reverse_use_cases

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load interactions
        self.stdout.write('Loading interactions from database...')
        if is_reverse:
            interactions = self.load_startup_interactions(min_interactions, use_case)
        else:
            interactions = self.load_interactions(min_interactions, use_case)
        
        if interactions.empty:
            self.stdout.write(self.style.ERROR('No interactions found!'))
            return

        self.stdout.write(self.style.SUCCESS(f'Loaded {len(interactions)} interactions'))

        # Step 2: Create user and item mappings
        self.stdout.write('Creating user and item mappings...')
        user_mapping, item_mapping = self.create_mappings(interactions)
        
        self.stdout.write(self.style.SUCCESS(
            f'Users: {len(user_mapping)}, Startups: {len(item_mapping)}'
        ))

        # Step 3: Build sparse matrix
        self.stdout.write('Building sparse interaction matrix...')
        sparse_matrix = self.build_sparse_matrix(
            interactions, user_mapping, item_mapping
        )
        
        self.stdout.write(self.style.SUCCESS(
            f'Matrix shape: {sparse_matrix.shape}, '
            f'Non-zero entries: {sparse_matrix.nnz}, '
            f'Density: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]) * 100:.4f}%'
        ))

        # Step 4: Build reverse sparse matrix (Startups × Users)
        self.stdout.write('Building reverse interaction matrix (Startups × Users)...')
        sparse_matrix_reverse = self.build_reverse_sparse_matrix(
            interactions, user_mapping, item_mapping
        )
        
        self.stdout.write(self.style.SUCCESS(
            f'Reverse matrix shape: {sparse_matrix_reverse.shape}, '
            f'Non-zero entries: {sparse_matrix_reverse.nnz}, '
            f'Density: {sparse_matrix_reverse.nnz / (sparse_matrix_reverse.shape[0] * sparse_matrix_reverse.shape[1]) * 100:.4f}%'
        ))

        # Step 5: Save outputs
        self.stdout.write('Saving dataset files...')
        self.save_dataset(sparse_matrix, user_mapping, item_mapping, output_dir)
        self.save_reverse_dataset(sparse_matrix_reverse, user_mapping, item_mapping, output_dir)
        
        self.stdout.write(self.style.SUCCESS('\n=== Dataset Generation Complete! ===\n'))
        self.stdout.write('\nForward Matrix (Users × Startups):')
        self.stdout.write(f'  Matrix: {os.path.join(output_dir, "als_interactions.npz")}')
        self.stdout.write(f'  User mapping: {os.path.join(output_dir, "als_user_mapping.json")}')
        self.stdout.write(f'  Item mapping: {os.path.join(output_dir, "als_item_mapping.json")}')
        self.stdout.write('\nReverse Matrix (Startups × Users):')
        self.stdout.write(f'  Matrix: {os.path.join(output_dir, "als_interactions_reverse.npz")}')
        self.stdout.write(f'  User mapping: {os.path.join(output_dir, "als_reverse_user_mapping.json")}')
        self.stdout.write(f'  Item mapping: {os.path.join(output_dir, "als_reverse_item_mapping.json")}')

    def load_interactions(self, min_interactions):
        """Load interactions with minimum threshold filter"""
        import pandas as pd

        # Get users and startups with minimum interactions
        valid_users = UserInteraction.objects.values('user_id').annotate(
            count=Count('id')
        ).filter(count__gte=min_interactions).values_list('user_id', flat=True)

        valid_startups = UserInteraction.objects.values('startup_id').annotate(
            count=Count('id')
        ).filter(count__gte=min_interactions).values_list('startup_id', flat=True)

        # Filter by user roles based on use case
        user_filter = {}
        if use_case == 'developer_startup':
            user_filter['user__role'] = 'student'
        elif use_case == 'investor_startup':
            user_filter['user__role'] = 'investor'
        
        # Load interactions for valid users and startups
        # Filter out dislikes (explicit negatives) - ALS uses implicit feedback (all positive signals)
        # Weighted by interaction type: apply (3.0) > favorite (2.5) > like (2.0) > click (1.0) > view (0.5)
        interactions = UserInteraction.objects.filter(
            user_id__in=valid_users,
            startup_id__in=valid_startups,
            **user_filter
        ).exclude(
            interaction_type='dislike'  # Exclude explicit negatives
        ).select_related('user', 'startup').values(
            'user_id', 'startup_id', 'interaction_type', 'weight', 'created_at'
        )

        df = pd.DataFrame(list(interactions))
        
        if df.empty:
            return df

        # Convert UUIDs to strings
        df['user_id'] = df['user_id'].astype(str)
        df['startup_id'] = df['startup_id'].astype(str)

        return df

    def load_startup_interactions(self, min_interactions, use_case='startup_developer'):
        """Load startup interactions for reverse use cases (startup → user)"""
        import pandas as pd
        
        # Determine target user roles
        if use_case == 'startup_developer':
            target_user_roles = ['student']
        elif use_case == 'startup_investor':
            target_user_roles = ['investor']
        else:
            target_user_roles = ['student', 'investor']
        
        # Get startups and users with minimum interactions
        valid_startups = StartupInteraction.objects.values('startup_id').annotate(
            count=Count('id')
        ).filter(count__gte=min_interactions).values_list('startup_id', flat=True)

        valid_users = StartupInteraction.objects.values('target_user_id').annotate(
            count=Count('id')
        ).filter(
            target_user__role__in=target_user_roles
        ).filter(count__gte=min_interactions).values_list('target_user_id', flat=True)

        # Load startup interactions
        # Weighted by interaction type: apply_received (3.0) > contact (2.0) > click (1.0) > view (0.5)
        interactions = StartupInteraction.objects.filter(
            startup_id__in=valid_startups,
            target_user_id__in=valid_users
        ).select_related('startup', 'target_user').values(
            'startup_id', 'target_user_id', 'interaction_type', 'weight', 'created_at'
        )

        df = pd.DataFrame(list(interactions))
        
        if df.empty:
            return df

        # Convert UUIDs to strings and rename columns for consistency
        # For reverse: startup_id becomes user_id, target_user_id becomes startup_id
        df['user_id'] = df['startup_id'].astype(str)
        df['startup_id'] = df['target_user_id'].astype(str)
        df = df.drop(columns=['target_user_id'])

        return df

    def create_mappings(self, interactions):
        """Create bidirectional ID mappings"""
        # Get unique users and items
        unique_users = sorted(interactions['user_id'].unique())
        unique_items = sorted(interactions['startup_id'].unique())

        # Create mappings: UUID -> index
        user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}

        # Also create reverse mappings: index -> UUID
        user_mapping['_reverse'] = {idx: user_id for user_id, idx in user_mapping.items() if user_id != '_reverse'}
        item_mapping['_reverse'] = {idx: item_id for item_id, idx in item_mapping.items() if item_id != '_reverse'}

        return user_mapping, item_mapping

    def build_sparse_matrix(self, interactions, user_mapping, item_mapping):
        """
        Build sparse CSR matrix from interactions
        
        Uses weighted interactions based on interaction type:
        - apply: 3.0 (strongest signal)
        - favorite: 2.5
        - like: 2.0
        - click: 1.0
        - view: 0.5 (weakest signal)
        """
        # Map UUIDs to indices
        row_indices = interactions['user_id'].map(lambda x: user_mapping.get(x, -1))
        col_indices = interactions['startup_id'].map(lambda x: item_mapping.get(x, -1))
        
        # Filter out any unmapped entries
        valid_mask = (row_indices != -1) & (col_indices != -1)
        row_indices = row_indices[valid_mask].values
        col_indices = col_indices[valid_mask].values
        # Use interaction weights (already calculated based on interaction type)
        weights = interactions.loc[valid_mask, 'weight'].values

        # Create sparse matrix
        n_users = len([k for k in user_mapping.keys() if k != '_reverse'])
        n_items = len([k for k in item_mapping.keys() if k != '_reverse'])

        sparse_matrix = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        return sparse_matrix

    def build_reverse_sparse_matrix(self, interactions, user_mapping, item_mapping):
        """
        Build reverse sparse CSR matrix (Startups × Users) - just transpose!
        Uses same weighted interactions as forward matrix
        """
        # Map UUIDs to indices (SWAP row and col for reverse)
        row_indices = interactions['startup_id'].map(lambda x: item_mapping.get(x, -1))
        col_indices = interactions['user_id'].map(lambda x: user_mapping.get(x, -1))
        
        # Filter out any unmapped entries
        valid_mask = (row_indices != -1) & (col_indices != -1)
        row_indices = row_indices[valid_mask].values
        col_indices = col_indices[valid_mask].values
        # Use interaction weights (already calculated based on interaction type)
        weights = interactions.loc[valid_mask, 'weight'].values

        # Create sparse matrix (Startups × Users)
        n_items = len([k for k in item_mapping.keys() if k != '_reverse'])
        n_users = len([k for k in user_mapping.keys() if k != '_reverse'])

        sparse_matrix_reverse = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_items, n_users),
            dtype=np.float32
        )

        return sparse_matrix_reverse

    def save_dataset(self, sparse_matrix, user_mapping, item_mapping, output_dir):
        """Save sparse matrix and mappings to disk"""
        # Save sparse matrix
        matrix_path = os.path.join(output_dir, 'als_interactions.npz')
        save_npz(matrix_path, sparse_matrix)

        # Save mappings (exclude reverse mappings from JSON for cleaner files)
        user_mapping_clean = {k: v for k, v in user_mapping.items() if k != '_reverse'}
        item_mapping_clean = {k: v for k, v in item_mapping.items() if k != '_reverse'}

        user_mapping_path = os.path.join(output_dir, 'als_user_mapping.json')
        with open(user_mapping_path, 'w') as f:
            json.dump(user_mapping_clean, f, indent=2)

        item_mapping_path = os.path.join(output_dir, 'als_item_mapping.json')
        with open(item_mapping_path, 'w') as f:
            json.dump(item_mapping_clean, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f'Saved forward sparse matrix: {matrix_path}'))
        self.stdout.write(self.style.SUCCESS(f'Saved forward user mapping: {user_mapping_path}'))
        self.stdout.write(self.style.SUCCESS(f'Saved forward item mapping: {item_mapping_path}'))

    def save_reverse_dataset(self, sparse_matrix_reverse, user_mapping, item_mapping, output_dir):
        """Save reverse sparse matrix and mappings to disk"""
        # Save reverse sparse matrix
        matrix_path = os.path.join(output_dir, 'als_interactions_reverse.npz')
        save_npz(matrix_path, sparse_matrix_reverse)

        # For reverse mappings, startup becomes "user" and user becomes "item"
        # Reverse user mapping: startup_id -> index
        reverse_user_mapping_clean = {k: v for k, v in item_mapping.items() if k != '_reverse'}
        
        # Reverse item mapping: user_id -> index
        reverse_item_mapping_clean = {k: v for k, v in user_mapping.items() if k != '_reverse'}

        reverse_user_mapping_path = os.path.join(output_dir, 'als_reverse_user_mapping.json')
        with open(reverse_user_mapping_path, 'w') as f:
            json.dump(reverse_user_mapping_clean, f, indent=2)

        reverse_item_mapping_path = os.path.join(output_dir, 'als_reverse_item_mapping.json')
        with open(reverse_item_mapping_path, 'w') as f:
            json.dump(reverse_item_mapping_clean, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f'Saved reverse sparse matrix: {matrix_path}'))
        self.stdout.write(self.style.SUCCESS(f'Saved reverse user mapping (startups): {reverse_user_mapping_path}'))
        self.stdout.write(self.style.SUCCESS(f'Saved reverse item mapping (users): {reverse_item_mapping_path}'))


