from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from django.contrib.auth import get_user_model
from api.recommendation_models import UserInteraction
from api.models import Startup
import os
import sys
from pathlib import Path

User = get_user_model()


class Command(BaseCommand):
    help = 'Find users with more than 5 interactions and optionally generate two-tower recommendations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-interactions',
            type=int,
            default=5,
            help='Minimum number of interactions required (default: 5)',
        )
        parser.add_argument(
            '--generate-recommendations',
            action='store_true',
            help='Generate two-tower recommendations for found users',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=10,
            help='Number of recommendations per user (default: 10)',
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path to save results (optional)',
        )

    def handle(self, *args, **options):
        min_interactions = options['min_interactions']
        generate_recommendations = options['generate_recommendations']
        limit = options['limit']
        output_path = options.get('output')

        self.stdout.write(self.style.SUCCESS(
            f'Finding users with more than {min_interactions} interactions...'
        ))

        # Count interactions per user
        user_interaction_counts = (
            UserInteraction.objects
            .values('user_id')
            .annotate(interaction_count=Count('id'))
            .filter(interaction_count__gt=min_interactions)
            .order_by('-interaction_count')
        )

        # Get user IDs with sufficient interactions
        high_interaction_user_ids = [
            item['user_id'] for item in user_interaction_counts
        ]

        if not high_interaction_user_ids:
            self.stdout.write(self.style.WARNING(
                f'No users found with more than {min_interactions} interactions.'
            ))
            return

        # Get users with their interaction counts
        users_with_counts = []
        for item in user_interaction_counts:
            try:
                user = User.objects.get(id=item['user_id'])
                users_with_counts.append({
                    'user': user,
                    'interaction_count': item['interaction_count']
                })
            except User.DoesNotExist:
                continue

        total_users = len(users_with_counts)
        self.stdout.write(self.style.SUCCESS(
            f'\nFound {total_users} users with more than {min_interactions} interactions:'
        ))
        self.stdout.write('=' * 80)

        # Display users
        results = []
        for idx, item in enumerate(users_with_counts, 1):
            user = item['user']
            interaction_count = item['interaction_count']
            
            # Get interaction breakdown
            interaction_types = (
                UserInteraction.objects
                .filter(user=user)
                .values('interaction_type')
                .annotate(count=Count('id'))
                .order_by('-count')
            )
            
            type_breakdown = ', '.join([
                f"{t['interaction_type']}: {t['count']}"
                for t in interaction_types[:5]
            ])
            
            user_info = {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'interaction_count': interaction_count,
                'interaction_types': type_breakdown,
            }
            
            self.stdout.write(
                f"{idx}. {user.username} ({user.email}) - "
                f"Role: {user.role} - "
                f"Interactions: {interaction_count}"
            )
            self.stdout.write(f"   Types: {type_breakdown}")
            
            results.append(user_info)

        # Summary by role
        role_counts = {}
        for item in users_with_counts:
            role = item['user'].role or 'unknown'
            role_counts[role] = role_counts.get(role, 0) + 1

        self.stdout.write('\n' + '=' * 80)
        self.stdout.write('Summary by role:')
        for role, count in sorted(role_counts.items()):
            self.stdout.write(f'  {role}: {count} users')

        # Generate recommendations if requested
        if generate_recommendations:
            self.stdout.write('\n' + '=' * 80)
            self.stdout.write(self.style.SUCCESS('Generating two-tower recommendations...'))
            
            # Try to import two-tower inference
            try:
                # Add recommendation_service to path
                rec_service_path = Path(__file__).parent.parent.parent.parent.parent / 'recommendation_service'
                if rec_service_path.exists():
                    sys.path.insert(0, str(rec_service_path))
                
                from inference_two_tower import TwoTowerInference
                
                # Find model file
                model_path = rec_service_path / 'models' / 'two_tower_v1.pth'
                if not model_path.exists():
                    # Try alternative names
                    model_files = list((rec_service_path / 'models').glob('two_tower*.pth'))
                    if model_files:
                        model_path = model_files[0]
                    else:
                        self.stdout.write(self.style.ERROR(
                            'Two-tower model not found. Please train the model first.'
                        ))
                        return
                
                self.stdout.write(f'Loading model from: {model_path}')
                inference = TwoTowerInference(str(model_path))
                
                # Generate recommendations for each user
                recommendations_generated = 0
                for idx, item in enumerate(users_with_counts[:20], 1):  # Limit to first 20 for display
                    user = item['user']
                    self.stdout.write(f'\n[{idx}/{min(20, total_users)}] Generating recommendations for {user.username}...')
                    
                    try:
                        # Get recommendations
                        rec_results = inference.recommend(
                            str(user.id),
                            limit=limit,
                            filters=None
                        )
                        
                        if rec_results.get('item_ids'):
                            user_info['recommendations'] = {
                                'count': len(rec_results['item_ids']),
                                'top_startups': rec_results['item_ids'][:5],
                                'top_scores': {
                                    sid: rec_results['scores'].get(sid, 0.0)
                                    for sid in rec_results['item_ids'][:5]
                                }
                            }
                            
                            self.stdout.write(f"  ✓ Generated {len(rec_results['item_ids'])} recommendations")
                            self.stdout.write(f"  Top 3 startups: {', '.join(rec_results['item_ids'][:3])}")
                            
                            # Show startup titles
                            startup_ids = rec_results['item_ids'][:3]
                            startups = Startup.objects.filter(id__in=startup_ids)
                            for startup in startups:
                                score = rec_results['scores'].get(str(startup.id), 0.0)
                                self.stdout.write(f"    - {startup.title} (score: {score:.3f})")
                            
                            recommendations_generated += 1
                        else:
                            self.stdout.write(self.style.WARNING(
                                f"  ⚠ No recommendations generated for {user.username}"
                            ))
                            
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(
                            f"  ✗ Error generating recommendations: {str(e)}"
                        ))
                
                self.stdout.write('\n' + '=' * 80)
                self.stdout.write(self.style.SUCCESS(
                    f'Generated recommendations for {recommendations_generated} users'
                ))
                
            except ImportError as e:
                self.stdout.write(self.style.ERROR(
                    f'Could not import two-tower inference module: {str(e)}'
                ))
                self.stdout.write(self.style.WARNING(
                    'Make sure the recommendation_service is properly set up.'
                ))
            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'Error generating recommendations: {str(e)}'
                ))

        # Save results to file if requested
        if output_path:
            import json
            output_data = {
                'total_users': total_users,
                'min_interactions': min_interactions,
                'users': results,
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.stdout.write(self.style.SUCCESS(f'\nResults saved to: {output_path}'))

        self.stdout.write('\n' + '=' * 80)
        self.stdout.write(self.style.SUCCESS('Done!'))

