from django.core.management.base import BaseCommand
from django.db.models import Count
from django.contrib.auth import get_user_model
from api.recommendation_models import UserInteraction

User = get_user_model()


class Command(BaseCommand):
    help = 'Remove users who have less than 5 interactions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-interactions',
            type=int,
            default=5,
            help='Minimum number of interactions required (default: 5)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting users',
        )
        parser.add_argument(
            '--exclude-roles',
            nargs='+',
            default=[],
            help='Roles to exclude from deletion (e.g., --exclude-roles entrepreneur investor)',
        )

    def handle(self, *args, **options):
        min_interactions = options['min_interactions']
        dry_run = options['dry_run']
        exclude_roles = options['exclude_roles'] or []

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No users will be deleted'))

        self.stdout.write(self.style.SUCCESS(
            f'Finding users with less than {min_interactions} interactions...'
        ))

        # Count interactions per user
        user_interaction_counts = (
            UserInteraction.objects
            .values('user_id')
            .annotate(interaction_count=Count('id'))
            .filter(interaction_count__lt=min_interactions)
        )

        # Get user IDs with low interactions
        low_interaction_user_ids = [
            item['user_id'] for item in user_interaction_counts
        ]

        if not low_interaction_user_ids:
            self.stdout.write(self.style.SUCCESS('No users found with low interactions.'))
            return

        # Get users, excluding specified roles
        users_to_delete = User.objects.filter(id__in=low_interaction_user_ids)

        if exclude_roles:
            users_to_delete = users_to_delete.exclude(role__in=exclude_roles)
            self.stdout.write(
                self.style.WARNING(f'Excluding users with roles: {", ".join(exclude_roles)}')
            )

        users_to_delete = users_to_delete.select_related().prefetch_related()

        total_users = users_to_delete.count()

        if total_users == 0:
            self.stdout.write(self.style.SUCCESS('No users to delete (all excluded by role filters).'))
            return

        self.stdout.write(f'\nFound {total_users} users with less than {min_interactions} interactions:')
        self.stdout.write('-' * 80)

        # Show details for first 20 users
        for idx, user in enumerate(users_to_delete[:20], 1):
            interaction_count = UserInteraction.objects.filter(user=user).count()
            self.stdout.write(
                f'{idx}. {user.username} ({user.email}) - Role: {user.role} - '
                f'Interactions: {interaction_count}'
            )

        if total_users > 20:
            self.stdout.write(f'... and {total_users - 20} more users')

        # Show summary by role
        role_counts = {}
        for user in users_to_delete:
            role = user.role or 'unknown'
            role_counts[role] = role_counts.get(role, 0) + 1

        self.stdout.write('\nSummary by role:')
        for role, count in sorted(role_counts.items()):
            self.stdout.write(f'  {role}: {count} users')

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f'\nDRY RUN: Would delete {total_users} users. '
                    'Run without --dry-run to actually delete them.'
                )
            )
            return

        # Confirm deletion
        self.stdout.write(
            self.style.WARNING(
                f'\n⚠️  WARNING: About to delete {total_users} users and all their related data!'
            )
        )
        self.stdout.write('This will delete:')
        self.stdout.write('  - User accounts')
        self.stdout.write('  - All their interactions')
        self.stdout.write('  - All their onboarding preferences')
        self.stdout.write('  - All their startups (if entrepreneur)')
        self.stdout.write('  - All their applications')
        self.stdout.write('  - All their favorites and interests')
        self.stdout.write('  - All their messages and conversations')
        self.stdout.write('  - All other related data (CASCADE)')

        # Actually delete
        deleted_count = 0
        errors = 0

        self.stdout.write('\nDeleting users...')
        for idx, user in enumerate(users_to_delete, 1):
            try:
                username = user.username
                user_id = user.id
                user.delete()  # CASCADE will handle related objects
                deleted_count += 1
                if idx % 10 == 0:
                    self.stdout.write(f'  Deleted {idx}/{total_users} users...')
            except Exception as e:
                errors += 1
                self.stdout.write(
                    self.style.ERROR(f'  Error deleting user {user.username}: {str(e)}')
                )

        self.stdout.write(self.style.SUCCESS(
            f'\n✅ Successfully deleted {deleted_count} users'
        ))

        if errors > 0:
            self.stdout.write(self.style.ERROR(f'❌ {errors} errors occurred during deletion'))

        # Show final stats
        remaining_users = User.objects.count()
        remaining_interactions = UserInteraction.objects.count()
        self.stdout.write(f'\nFinal statistics:')
        self.stdout.write(f'  Remaining users: {remaining_users}')
        self.stdout.write(f'  Remaining interactions: {remaining_interactions}')

