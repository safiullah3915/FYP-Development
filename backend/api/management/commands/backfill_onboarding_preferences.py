from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from api.recommendation_models import UserOnboardingPreferences
from api.messaging_models import UserProfile
import random

User = get_user_model()

# Available categories from Startup model
CATEGORIES = [
    'saas', 'ecommerce', 'agency', 'legal', 'marketplace', 
    'media', 'platform', 'real_estate', 'robotics', 
    'software', 'web3', 'crypto', 'other'
]

# Common fields/industries
FIELDS = [
    'fintech', 'healthcare', 'education', 'ecommerce', 'saas',
    'real_estate', 'media', 'entertainment', 'food', 'travel',
    'fitness', 'fashion', 'logistics', 'agriculture', 'energy',
    'automotive', 'aerospace', 'telecommunications', 'retail', 'consulting'
]

# Common tags
TAGS = [
    'AI', 'Machine Learning', 'blockchain', 'mobile', 'web',
    'cloud', 'IoT', 'cybersecurity', 'data analytics', 'automation',
    'AR/VR', 'fintech', 'healthtech', 'edtech', 'ecommerce',
    'SaaS', 'B2B', 'B2C', 'marketplace', 'platform',
    'API', 'microservices', 'DevOps', 'agile', 'scalable'
]

# Startup stages
STAGES = ['early', 'growth', 'mature', 'scaling', 'established']

# Engagement types
ENGAGEMENT_TYPES = ['full-time', 'part-time', 'equity', 'paid']

# Common skills
SKILLS = [
    'Python', 'JavaScript', 'React', 'Node.js', 'Django', 'Flask',
    'Java', 'Spring Boot', 'C++', 'C#', '.NET', 'Go', 'Rust',
    'TypeScript', 'Vue.js', 'Angular', 'Next.js', 'Express',
    'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Docker', 'Kubernetes',
    'AWS', 'Azure', 'GCP', 'Machine Learning', 'Data Science',
    'TensorFlow', 'PyTorch', 'React Native', 'Flutter', 'Swift',
    'Kotlin', 'GraphQL', 'REST API', 'Microservices', 'DevOps'
]

# Role-based preference templates
ROLE_PREFERENCES = {
    'entrepreneur': {
        'categories': ['saas', 'ecommerce', 'marketplace', 'platform', 'software'],
        'fields': ['fintech', 'healthcare', 'education', 'ecommerce', 'saas'],
        'tags': ['SaaS', 'B2B', 'platform', 'scalable', 'automation'],
        'stages': ['early', 'growth', 'scaling'],
        'engagement_types': ['equity', 'full-time'],
        'skills': ['Python', 'JavaScript', 'React', 'Django', 'Node.js']
    },
    'student': {
        'categories': ['software', 'saas', 'web3', 'platform', 'media'],
        'fields': ['edtech', 'healthtech', 'fintech', 'ecommerce', 'media'],
        'tags': ['AI', 'Machine Learning', 'mobile', 'web', 'cloud'],
        'stages': ['early', 'growth'],
        'engagement_types': ['part-time', 'equity', 'paid', 'full-time'],
        'skills': ['Python', 'React', 'JavaScript', 'Machine Learning', 'Node.js', 'Django']
    },
    'investor': {
        'categories': ['saas', 'fintech', 'healthcare', 'web3', 'platform'],
        'fields': ['fintech', 'healthcare', 'education', 'real_estate', 'energy'],
        'tags': ['scalable', 'B2B', 'platform', 'AI', 'blockchain'],
        'stages': ['growth', 'mature', 'scaling'],
        'engagement_types': ['equity'],
        'skills': []
    }
}


def generate_preferences_for_user(user):
    """Generate synthetic but meaningful preferences based on user role and profile"""
    role = user.role
    
    # Get base preferences for the role
    base_prefs = ROLE_PREFERENCES.get(role, ROLE_PREFERENCES['student'])
    
    # Try to get user profile for additional context
    try:
        profile = user.profile
        existing_skills = profile.skills if hasattr(profile, 'skills') and profile.skills else []
        existing_fields = [profile.location] if hasattr(profile, 'location') and profile.location else []
    except UserProfile.DoesNotExist:
        existing_skills = []
        existing_fields = []
    
    # Generate categories (2-4 categories)
    num_categories = random.randint(2, 4)
    selected_categories = random.sample(base_prefs['categories'], min(num_categories, len(base_prefs['categories'])))
    # Add some randomness
    if random.random() < 0.3:  # 30% chance to add a random category
        available = [c for c in CATEGORIES if c not in selected_categories]
        if available:
            selected_categories.append(random.choice(available))
    
    # Generate fields (2-5 fields)
    num_fields = random.randint(2, 5)
    selected_fields = random.sample(base_prefs['fields'], min(num_fields, len(base_prefs['fields'])))
    # Add user-specific fields if available
    if existing_fields:
        selected_fields.extend(existing_fields[:2])
    # Add some randomness
    if random.random() < 0.4:  # 40% chance to add a random field
        available = [f for f in FIELDS if f not in selected_fields]
        if available:
            selected_fields.append(random.choice(available))
    
    # Generate tags (3-6 tags)
    num_tags = random.randint(3, 6)
    selected_tags = random.sample(base_prefs['tags'], min(num_tags, len(base_prefs['tags'])))
    # Add some randomness
    if random.random() < 0.5:  # 50% chance to add random tags
        available = [t for t in TAGS if t not in selected_tags]
        if available:
            selected_tags.extend(random.sample(available, min(2, len(available))))
    
    # Generate stages (1-3 stages)
    num_stages = random.randint(1, 3)
    preferred_stages = random.sample(base_prefs['stages'], min(num_stages, len(base_prefs['stages'])))
    
    # Generate engagement types (1-3 types, more for students)
    if role == 'student':
        num_engagement = random.randint(2, 4)
    else:
        num_engagement = random.randint(1, 2)
    preferred_engagement = random.sample(
        base_prefs['engagement_types'], 
        min(num_engagement, len(base_prefs['engagement_types']))
    )
    
    # Generate skills (3-8 skills, only for students and entrepreneurs)
    preferred_skills = []
    if role in ['student', 'entrepreneur']:
        num_skills = random.randint(3, 8)
        # Use existing skills if available
        if existing_skills:
            preferred_skills = existing_skills[:min(5, len(existing_skills))]
            num_skills = max(0, num_skills - len(preferred_skills))
        
        # Add base role skills
        available_base_skills = [s for s in base_prefs['skills'] if s not in preferred_skills]
        if available_base_skills:
            preferred_skills.extend(random.sample(available_base_skills, min(3, len(available_base_skills))))
        
        # Add random skills to fill up
        available_skills = [s for s in SKILLS if s not in preferred_skills]
        if available_skills and num_skills > 0:
            preferred_skills.extend(random.sample(available_skills, min(num_skills, len(available_skills))))
    
    # Remove duplicates and limit sizes
    selected_categories = list(set(selected_categories))[:5]
    selected_fields = list(set(selected_fields))[:6]
    selected_tags = list(set(selected_tags))[:8]
    preferred_stages = list(set(preferred_stages))[:4]
    preferred_engagement = list(set(preferred_engagement))[:4]
    preferred_skills = list(set(preferred_skills))[:10]
    
    return {
        'selected_categories': selected_categories,
        'selected_fields': selected_fields,
        'selected_tags': selected_tags,
        'preferred_startup_stages': preferred_stages,
        'preferred_engagement_types': preferred_engagement,
        'preferred_skills': preferred_skills,
        'onboarding_completed': True  # Mark as completed since we're backfilling
    }


class Command(BaseCommand):
    help = 'Backfill UserOnboardingPreferences table with synthetic but meaningful data for existing users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating records',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Update existing preferences (by default, skips users who already have preferences)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write(self.style.SUCCESS('Starting onboarding preferences backfill...'))
        
        # Get all active users
        all_users = User.objects.filter(is_active=True)
        total_users = all_users.count()
        
        self.stdout.write(f'Total active users: {total_users}')
        
        # Get users without preferences
        users_with_prefs = set(
            UserOnboardingPreferences.objects.values_list('user_id', flat=True)
        )
        
        if force:
            users_to_process = all_users
            self.stdout.write(self.style.WARNING('Force mode: Will update existing preferences'))
        else:
            users_to_process = all_users.exclude(id__in=users_with_prefs)
            self.stdout.write(f'Users without preferences: {users_to_process.count()}')
        
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for user in users_to_process:
            try:
                # Check if preferences already exist
                existing_prefs = None
                try:
                    existing_prefs = UserOnboardingPreferences.objects.get(user=user)
                except UserOnboardingPreferences.DoesNotExist:
                    pass
                
                if existing_prefs and not force:
                    skipped_count += 1
                    continue
                
                # Generate preferences
                prefs_data = generate_preferences_for_user(user)
                
                if dry_run:
                    self.stdout.write(f'\n[DRY RUN] Would create/update preferences for: {user.username} ({user.role})')
                    self.stdout.write(f'  Categories: {prefs_data["selected_categories"]}')
                    self.stdout.write(f'  Fields: {prefs_data["selected_fields"]}')
                    self.stdout.write(f'  Tags: {prefs_data["selected_tags"]}')
                    self.stdout.write(f'  Stages: {prefs_data["preferred_startup_stages"]}')
                    self.stdout.write(f'  Engagement: {prefs_data["preferred_engagement_types"]}')
                    self.stdout.write(f'  Skills: {prefs_data["preferred_skills"]}')
                    if existing_prefs:
                        updated_count += 1
                    else:
                        created_count += 1
                else:
                    # Create or update preferences
                    if existing_prefs:
                        for key, value in prefs_data.items():
                            setattr(existing_prefs, key, value)
                        existing_prefs.save()
                        updated_count += 1
                        self.stdout.write(f'✓ Updated preferences for: {user.username} ({user.role})')
                    else:
                        UserOnboardingPreferences.objects.create(
                            user=user,
                            **prefs_data
                        )
                        created_count += 1
                        self.stdout.write(f'✓ Created preferences for: {user.username} ({user.role})')
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Error processing user {user.username}: {str(e)}')
                )
        
        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        if dry_run:
            self.stdout.write(self.style.SUCCESS('DRY RUN SUMMARY:'))
            self.stdout.write(f'  Would create: {created_count}')
            self.stdout.write(f'  Would update: {updated_count}')
            self.stdout.write(f'  Skipped: {skipped_count}')
        else:
            self.stdout.write(self.style.SUCCESS('BACKFILL SUMMARY:'))
            self.stdout.write(f'  Created: {created_count}')
            self.stdout.write(f'  Updated: {updated_count}')
            self.stdout.write(f'  Skipped: {skipped_count}')
            self.stdout.write(f'  Total processed: {created_count + updated_count}')
        self.stdout.write(self.style.SUCCESS('='*50))

