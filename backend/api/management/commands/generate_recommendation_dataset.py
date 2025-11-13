from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction, IntegrityError
from django.contrib.auth import get_user_model
from faker import Faker
import random
from datetime import timedelta
from collections import defaultdict

from api.models import Startup, StartupTag, Position
from api.messaging_models import UserProfile
from api.recommendation_models import UserOnboardingPreferences, UserInteraction

User = get_user_model()
fake = Faker()

# Import constants from backfill_onboarding_preferences
CATEGORIES = [
    'saas', 'ecommerce', 'agency', 'legal', 'marketplace', 
    'media', 'platform', 'real_estate', 'robotics', 
    'software', 'web3', 'crypto', 'other'
]

FIELDS = [
    'fintech', 'healthcare', 'education', 'ecommerce', 'saas',
    'real_estate', 'media', 'entertainment', 'food', 'travel',
    'fitness', 'fashion', 'logistics', 'agriculture', 'energy',
    'automotive', 'aerospace', 'telecommunications', 'retail', 'consulting'
]

TAGS = [
    'AI', 'Machine Learning', 'blockchain', 'mobile', 'web',
    'cloud', 'IoT', 'cybersecurity', 'data analytics', 'automation',
    'AR/VR', 'fintech', 'healthtech', 'edtech', 'ecommerce',
    'SaaS', 'B2B', 'B2C', 'marketplace', 'platform',
    'API', 'microservices', 'DevOps', 'agile', 'scalable',
    'remote-first', 'bootstrapped', 'funded', 'YC-backed'
]

STAGES = ['early', 'growth', 'mature', 'scaling', 'established']

ENGAGEMENT_TYPES = ['full-time', 'part-time', 'equity', 'paid']

SKILLS = [
    'Python', 'JavaScript', 'React', 'Node.js', 'Django', 'Flask',
    'Java', 'Spring Boot', 'C++', 'C#', '.NET', 'Go', 'Rust',
    'TypeScript', 'Vue.js', 'Angular', 'Next.js', 'Express',
    'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Docker', 'Kubernetes',
    'AWS', 'Azure', 'GCP', 'Machine Learning', 'Data Science',
    'TensorFlow', 'PyTorch', 'React Native', 'Flutter', 'Swift',
    'Kotlin', 'GraphQL', 'REST API', 'Microservices', 'DevOps',
    'Marketing', 'Sales', 'Business Development', 'Product Management',
    'Project Management', 'Strategic Planning', 'Market Research',
    'Financial Analysis', 'Fundraising', 'Partnership Development',
    'UI/UX Design', 'Graphic Design', 'Figma', 'Adobe XD', 'Sketch'
]

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


class Command(BaseCommand):
    help = 'Generate realistic synthetic data for recommendation system training'

    def add_arguments(self, parser):
        parser.add_argument(
            '--users',
            type=int,
            default=350,
            help='Number of users to create (default: 350)',
        )
        parser.add_argument(
            '--startups',
            type=int,
            default=250,
            help='Number of startups to create (default: 250)',
        )
        parser.add_argument(
            '--interactions',
            type=int,
            default=7500,
            help='Number of interactions to create (default: 7500)',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing seed data before generating new data',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without actually creating records',
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        num_users = options['users']
        num_startups = options['startups']
        num_interactions = options['interactions']
        clear_data = options['clear']

        if self.dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No records will be created'))

        if clear_data and not self.dry_run:
            if input('Are you sure you want to clear existing seed data? (yes/no): ') != 'yes':
                self.stdout.write(self.style.WARNING('Operation cancelled.'))
                return
            self.clear_seed_data()

        self.stdout.write(self.style.SUCCESS('Starting recommendation dataset generation...'))
        start_time = timezone.now()

        try:
            with transaction.atomic():
                # Generate data in correct order
                entrepreneurs, students, investors = self.create_users(num_users)
                all_users = entrepreneurs + students + investors
                
                self.create_user_profiles(all_users)
                self.create_onboarding_preferences(all_users)
                
                marketplace_startups, collaboration_startups = self.create_startups(
                    num_startups, entrepreneurs
                )
                all_startups = marketplace_startups + collaboration_startups
                
                self.create_startup_tags(all_startups)
                positions = self.create_positions(collaboration_startups)
                
                interactions = self.create_user_interactions(
                    all_users, all_startups, positions, num_interactions
                )
                
                if not self.dry_run:
                    # Run validation
                    validation_results = self.validate_data(all_users, all_startups, positions, interactions)
                    self.print_summary(
                        entrepreneurs, students, investors, all_startups,
                        positions, interactions, validation_results
                    )
                else:
                    # Force rollback for dry-run mode
                    transaction.set_rollback(True)
                    self.print_dry_run_summary(
                        entrepreneurs, students, investors, all_startups,
                        positions, interactions
                    )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error generating dataset: {str(e)}'))
            import traceback
            self.stdout.write(self.style.ERROR(traceback.format_exc()))
            raise

        elapsed = (timezone.now() - start_time).total_seconds()
        self.stdout.write(self.style.SUCCESS(f'\nGeneration completed in {elapsed:.2f} seconds'))

    def clear_seed_data(self):
        """Clear existing seed data in reverse dependency order"""
        self.stdout.write(self.style.WARNING('Clearing existing seed data...'))
        
        UserInteraction.objects.all().delete()
        UserOnboardingPreferences.objects.all().delete()
        Position.objects.all().delete()
        StartupTag.objects.all().delete()
        Startup.objects.all().delete()
        UserProfile.objects.all().delete()
        User.objects.filter(is_superuser=False, is_staff=False).delete()
        
        self.stdout.write(self.style.SUCCESS('Cleared existing seed data.'))

    def create_users(self, num_users):
        """Create users with different roles and timestamp distribution"""
        self.stdout.write('Creating users...')
        
        # Calculate distribution: 40% entrepreneurs, 45% students, 15% investors
        num_entrepreneurs = int(num_users * 0.40)
        num_students = int(num_users * 0.45)
        num_investors = num_users - num_entrepreneurs - num_students
        
        entrepreneurs = []
        students = []
        investors = []
        
        # Generate timestamps distributed across last 12 months
        now = timezone.now()
        base_date = now - timedelta(days=365)
        
        # Create entrepreneurs
        for i in range(num_entrepreneurs):
            try:
                # Distribute creation dates
                days_ago = random.randint(0, 365)
                created_at = base_date + timedelta(days=days_ago)
                
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='entrepreneur',
                    phone_number=fake.phone_number()[:20] if random.random() < 0.7 else '',
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                # Update created_at manually
                User.objects.filter(id=user.id).update(created_at=created_at)
                user.refresh_from_db()
                entrepreneurs.append(user)
                
                if (i + 1) % 50 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_entrepreneurs} entrepreneurs...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating entrepreneur {i + 1}: {str(e)}'))
                continue
        
        # Create students
        for i in range(num_students):
            try:
                days_ago = random.randint(0, 365)
                created_at = base_date + timedelta(days=days_ago)
                
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='student',
                    phone_number=fake.phone_number()[:20] if random.random() < 0.6 else '',
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                User.objects.filter(id=user.id).update(created_at=created_at)
                user.refresh_from_db()
                students.append(user)
                
                if (i + 1) % 50 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_students} students...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating student {i + 1}: {str(e)}'))
                continue
        
        # Create investors
        for i in range(num_investors):
            try:
                days_ago = random.randint(0, 365)
                created_at = base_date + timedelta(days=days_ago)
                
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='investor',
                    phone_number=fake.phone_number()[:20] if random.random() < 0.8 else '',
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                User.objects.filter(id=user.id).update(created_at=created_at)
                user.refresh_from_db()
                investors.append(user)
                
                if (i + 1) % 20 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_investors} investors...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating investor {i + 1}: {str(e)}'))
                continue
        
        self.stdout.write(self.style.SUCCESS(
            f'Created {len(entrepreneurs)} entrepreneurs, {len(students)} students, {len(investors)} investors'
        ))
        return entrepreneurs, students, investors

    def create_user_profiles(self, users):
        """Create user profiles with role-appropriate data"""
        self.stdout.write('Creating user profiles...')
        
        tech_skills = [s for s in SKILLS if s not in ['Marketing', 'Sales', 'Business Development', 
                                                      'Product Management', 'Project Management',
                                                      'Strategic Planning', 'Market Research',
                                                      'Financial Analysis', 'Fundraising',
                                                      'Partnership Development', 'UI/UX Design',
                                                      'Graphic Design', 'Figma', 'Adobe XD', 'Sketch']]
        
        business_skills = ['Marketing', 'Sales', 'Business Development', 'Product Management',
                          'Project Management', 'Strategic Planning', 'Market Research',
                          'Financial Analysis', 'Fundraising', 'Partnership Development']
        
        design_skills = ['UI/UX Design', 'Graphic Design', 'Figma', 'Adobe XD', 'Sketch',
                        'Photoshop', 'Illustrator', 'Prototyping', 'User Research']
        
        work_modes = ['remote', 'hybrid', 'onsite']
        compensation_types = ['paid', 'equity', 'both']
        
        for i, user in enumerate(users):
            # Select skills based on role
            if user.role == 'student':
                skills = random.sample(tech_skills, random.randint(3, 8))
                if random.random() < 0.3:
                    skills.extend(random.sample(design_skills, random.randint(1, 3)))
                preferred_work_modes = random.sample(['remote', 'hybrid'], random.randint(1, 2))
                preferred_compensation = random.sample(['paid', 'part-time', 'equity'], random.randint(1, 3))
            elif user.role == 'entrepreneur':
                skills = random.sample(business_skills, random.randint(2, 5))
                if random.random() < 0.5:
                    skills.extend(random.sample(tech_skills, random.randint(1, 4)))
                preferred_work_modes = random.sample(work_modes, random.randint(1, 2))
                preferred_compensation = ['equity', 'both']
            else:  # investor
                skills = random.sample(business_skills, random.randint(3, 6))
                preferred_work_modes = ['remote', 'hybrid']
                preferred_compensation = ['equity']
            
            # Generate experience
            experience = []
            num_experiences = random.randint(1, 4)
            for _ in range(num_experiences):
                experience.append({
                    'title': fake.job(),
                    'company': fake.company(),
                    'duration': f"{random.randint(1, 5)} years",
                    'description': fake.text(max_nb_chars=200)
                })
            
            # Generate references
            references = []
            if random.random() < 0.6:
                num_refs = random.randint(1, 3)
                for _ in range(num_refs):
                    references.append({
                        'name': fake.name(),
                        'email': fake.email(),
                        'relationship': random.choice(['Former Colleague', 'Manager', 'Client', 'Mentor'])
                    })
            
            profile = UserProfile.objects.create(
                user=user,
                bio=fake.text(max_nb_chars=300),
                location=f"{fake.city()}, {fake.country()}",
                website=fake.url() if random.random() < 0.7 else '',
                is_public=random.choice([True] * 6 + [False] * 4),
                selected_regions=random.sample(['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'], 
                                             random.randint(1, 3)),
                skills=skills,
                experience=experience,
                references=references,
                preferred_work_modes=preferred_work_modes,
                preferred_compensation_types=preferred_compensation,
                onboarding_completed=True
            )
            
            if (i + 1) % 50 == 0:
                self.stdout.write(f'  Created {i + 1}/{len(users)} profiles...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(users)} user profiles'))

    def create_onboarding_preferences(self, users):
        """Create onboarding preferences aligned with user roles"""
        self.stdout.write('Creating onboarding preferences...')
        
        for i, user in enumerate(users):
            try:
                profile = user.profile
                existing_skills = profile.skills if profile.skills else []
            except UserProfile.DoesNotExist:
                existing_skills = []
            
            role = user.role
            base_prefs = ROLE_PREFERENCES.get(role, ROLE_PREFERENCES['student'])
            
            # Generate categories (2-4 categories)
            num_categories = random.randint(2, 4)
            selected_categories = random.sample(base_prefs['categories'], min(num_categories, len(base_prefs['categories'])))
            if random.random() < 0.3:
                available = [c for c in CATEGORIES if c not in selected_categories]
                if available:
                    selected_categories.append(random.choice(available))
            
            # Generate fields (2-5 fields)
            num_fields = random.randint(2, 5)
            selected_fields = random.sample(base_prefs['fields'], min(num_fields, len(base_prefs['fields'])))
            if random.random() < 0.4:
                available = [f for f in FIELDS if f not in selected_fields]
                if available:
                    selected_fields.append(random.choice(available))
            
            # Generate tags (3-6 tags)
            num_tags = random.randint(3, 6)
            selected_tags = random.sample(base_prefs['tags'], min(num_tags, len(base_prefs['tags'])))
            if random.random() < 0.5:
                available = [t for t in TAGS if t not in selected_tags]
                if available:
                    selected_tags.extend(random.sample(available, min(2, len(available))))
            
            # Generate stages (1-3 stages)
            num_stages = random.randint(1, 3)
            preferred_stages = random.sample(base_prefs['stages'], min(num_stages, len(base_prefs['stages'])))
            
            # Generate engagement types
            if role == 'student':
                num_engagement = random.randint(2, 4)
            else:
                num_engagement = random.randint(1, 2)
            preferred_engagement = random.sample(
                base_prefs['engagement_types'], 
                min(num_engagement, len(base_prefs['engagement_types']))
            )
            
            # Generate skills (only for students and entrepreneurs)
            preferred_skills = []
            if role in ['student', 'entrepreneur']:
                num_skills = random.randint(3, 8)
                if existing_skills:
                    preferred_skills = existing_skills[:min(5, len(existing_skills))]
                    num_skills = max(0, num_skills - len(preferred_skills))
                
                available_base_skills = [s for s in base_prefs['skills'] if s not in preferred_skills]
                if available_base_skills:
                    preferred_skills.extend(random.sample(available_base_skills, min(3, len(available_base_skills))))
                
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
            
            UserOnboardingPreferences.objects.create(
                user=user,
                selected_categories=selected_categories,
                selected_fields=selected_fields,
                selected_tags=selected_tags,
                preferred_startup_stages=preferred_stages,
                preferred_engagement_types=preferred_engagement,
                preferred_skills=preferred_skills,
                onboarding_completed=True
            )
            
            if (i + 1) % 50 == 0:
                self.stdout.write(f'  Created {i + 1}/{len(users)} preferences...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(users)} onboarding preferences'))

    def create_startups(self, num_startups, entrepreneurs):
        """Create startups with diversity across categories and types"""
        self.stdout.write('Creating startups...')
        
        num_collaboration = int(num_startups * 0.60)
        num_marketplace = num_startups - num_collaboration
        
        marketplace_startups = []
        collaboration_startups = []
        
        # Generate timestamps distributed across last 18 months
        now = timezone.now()
        base_date = now - timedelta(days=540)
        
        phases = ["Pre-seed", "Seed", "Series A", "Bootstrap"]
        team_sizes = ["Just me", "2-3 people", "4-5 people", "6-10 people"]
        earn_through_options = ["Equity", "Revenue Share", "Equity + Revenue Share", "Salary + Equity"]
        statuses = ['active'] * 9 + ['inactive'] * 1
        
        # Create marketplace startups
        for i in range(num_marketplace):
            try:
                days_ago = random.randint(0, 540)
                created_at = base_date + timedelta(days=days_ago)
                
                stage = random.choice([['early'], ['growth'], ['mature'], ['scaling'], ['early', 'growth']])
                stage_name = stage[0] if isinstance(stage, list) else stage
                
                # Generate financial data based on stage
                if stage_name == "early":
                    monthly_revenue = random.randint(5000, 20000)
                elif stage_name == "growth":
                    monthly_revenue = random.randint(20000, 100000)
                elif stage_name == "mature":
                    monthly_revenue = random.randint(100000, 500000)
                else:  # scaling
                    monthly_revenue = random.randint(500000, 2000000)
                
                profit_margin = random.uniform(0.15, 0.45)
                monthly_profit = int(monthly_revenue * profit_margin)
                annual_revenue = monthly_revenue * 12
                annual_profit = monthly_profit * 12
                asking_price = int(annual_revenue * random.uniform(2, 5))
                
                startup = Startup.objects.create(
                    owner=random.choice(entrepreneurs),
                    title=fake.company() + " " + random.choice(["Platform", "Solutions", "Tech", "App", "Systems", "Labs"]),
                    role_title=random.choice(["Founder & CEO", "Co-founder", "Founder", "CEO"]),
                    description=self.generate_startup_description(random.choice(CATEGORIES)),
                    field=random.choice(FIELDS),
                    website_url=fake.url() if random.random() < 0.8 else '',
                    stages=stage,
                    revenue=f"${monthly_revenue:,}/month",
                    profit=f"${monthly_profit:,}/month",
                    asking_price=f"${asking_price:,}",
                    ttm_revenue=f"${annual_revenue:,}",
                    ttm_profit=f"${annual_profit:,}",
                    last_month_revenue=f"${monthly_revenue:,}",
                    last_month_profit=f"${monthly_profit:,}",
                    type='marketplace',
                    category=random.choice(CATEGORIES),
                    status=random.choice(statuses),
                    views=random.randint(0, 10000),
                    featured=random.random() < 0.10
                )
                Startup.objects.filter(id=startup.id).update(created_at=created_at)
                startup.refresh_from_db()
                marketplace_startups.append(startup)
                
                if (i + 1) % 25 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_marketplace} marketplace startups...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating marketplace startup {i + 1}: {str(e)}'))
                continue
        
        # Create collaboration startups
        for i in range(num_collaboration):
            try:
                days_ago = random.randint(0, 540)
                created_at = base_date + timedelta(days=days_ago)
                
                startup = Startup.objects.create(
                    owner=random.choice(entrepreneurs),
                    title=fake.company() + " " + random.choice(["Platform", "Solutions", "Tech", "App", "Systems", "Labs"]),
                    role_title=random.choice(["Founder", "Co-founder", "Creator", "Builder"]),
                    description=self.generate_startup_description(random.choice(CATEGORIES)),
                    field=random.choice(FIELDS),
                    website_url=fake.url() if random.random() < 0.6 else '',
                    stages=random.choice([['early'], ['growth'], ['early', 'growth']]),
                    phase=random.choice(phases),
                    team_size=random.choice(team_sizes),
                    earn_through=random.choice(earn_through_options),
                    type='collaboration',
                    category=random.choice(CATEGORIES),
                    status=random.choice(statuses),
                    views=random.randint(0, 5000),
                    featured=random.random() < 0.08
                )
                Startup.objects.filter(id=startup.id).update(created_at=created_at)
                startup.refresh_from_db()
                collaboration_startups.append(startup)
                
                if (i + 1) % 25 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_collaboration} collaboration startups...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating collaboration startup {i + 1}: {str(e)}'))
                continue
        
        self.stdout.write(self.style.SUCCESS(
            f'Created {len(marketplace_startups)} marketplace and {len(collaboration_startups)} collaboration startups'
        ))
        return marketplace_startups, collaboration_startups

    def generate_startup_description(self, category):
        """Generate realistic startup description based on category"""
        descriptions = {
            'saas': f"{fake.company()} is a cutting-edge SaaS platform that {fake.bs()}. We help businesses streamline their operations through innovative cloud-based solutions. Our platform offers {random.choice(['real-time analytics', 'automated workflows', 'seamless integrations', 'advanced reporting'])} to help companies scale efficiently. With a growing customer base and proven market demand, we're positioned for rapid growth.",
            'ecommerce': f"{fake.company()} is an innovative e-commerce platform specializing in {random.choice(['sustainable products', 'handcrafted goods', 'tech accessories', 'fashion items'])}. We've built a strong brand presence and customer loyalty through exceptional service and quality products. Our platform features {random.choice(['AI-powered recommendations', 'seamless checkout', 'mobile-first design', 'multi-channel integration'])} and serves customers across multiple regions.",
            'agency': f"{fake.company()} is a full-service digital agency providing {random.choice(['web development', 'marketing services', 'brand design', 'content creation'])} to clients worldwide. We've delivered successful projects for {random.choice(['Fortune 500 companies', 'startups', 'non-profits', 'SMBs'])} and have a proven track record of driving results. Our team combines creativity with data-driven strategies.",
            'legal': f"{fake.company()} is a legal tech platform that {fake.bs()} for {random.choice(['law firms', 'legal professionals', 'businesses', 'individuals'])}. We've developed innovative solutions to {random.choice(['automate document processing', 'streamline case management', 'facilitate legal research', 'connect clients with attorneys'])}. Our platform is trusted by legal professionals nationwide.",
            'marketplace': f"{fake.company()} is a two-sided marketplace connecting {random.choice(['buyers and sellers', 'service providers and clients', 'freelancers and businesses', 'suppliers and retailers'])}. We've created a trusted platform with {random.choice(['secure transactions', 'verified users', 'rating systems', 'seamless payments'])}. Our marketplace has grown significantly and shows strong potential for expansion.",
            'media': f"{fake.company()} is a media platform focused on {random.choice(['video content', 'podcasting', 'digital publishing', 'streaming services'])}. We've built a loyal audience and established partnerships with {random.choice(['content creators', 'advertisers', 'publishers', 'broadcasters'])}. Our platform offers {random.choice(['unique content', 'exclusive access', 'premium features', 'ad-free experience'])}.",
            'platform': f"{fake.company()} is a comprehensive platform that {fake.bs()} for {random.choice(['developers', 'businesses', 'creators', 'professionals'])}. We provide {random.choice(['powerful APIs', 'integrated tools', 'developer resources', 'business solutions'])} that enable users to build and scale their projects. Our platform has a growing ecosystem of users and partners.",
            'real_estate': f"{fake.company()} is a real estate technology platform revolutionizing how people {random.choice(['buy and sell properties', 'find rentals', 'manage properties', 'invest in real estate'])}. We leverage {random.choice(['AI-powered matching', 'virtual tours', 'data analytics', 'blockchain technology'])} to provide a seamless experience. Our platform serves {random.choice(['homebuyers', 'investors', 'property managers', 'real estate agents'])}.",
            'robotics': f"{fake.company()} develops {random.choice(['autonomous robots', 'industrial automation', 'service robots', 'AI-powered robotics'])} for {random.choice(['manufacturing', 'healthcare', 'logistics', 'agriculture'])}. We combine cutting-edge robotics with {random.choice(['machine learning', 'computer vision', 'sensor technology', 'IoT integration'])} to create innovative solutions. Our products are deployed in various industries.",
            'software': f"{fake.company()} creates {random.choice(['enterprise software', 'developer tools', 'productivity apps', 'specialized software'])} that helps businesses {fake.bs()}. Our software solutions feature {random.choice(['intuitive interfaces', 'powerful features', 'seamless integrations', 'scalable architecture'])} and are used by companies of all sizes. We have a strong development roadmap and growing user base.",
            'web3': f"{fake.company()} is a Web3 platform building {random.choice(['decentralized applications', 'NFT marketplaces', 'DeFi protocols', 'blockchain infrastructure'])}. We're at the forefront of {random.choice(['blockchain technology', 'cryptocurrency', 'decentralized finance', 'digital ownership'])} and have a growing community of users and developers. Our platform leverages {random.choice(['smart contracts', 'blockchain networks', 'crypto wallets', 'decentralized storage'])}.",
            'crypto': f"{fake.company()} is a cryptocurrency platform offering {random.choice(['trading services', 'wallet solutions', 'payment processing', 'crypto exchanges'])}. We provide {random.choice(['secure transactions', 'low fees', 'multiple cryptocurrencies', 'advanced trading tools'])} to users worldwide. Our platform is {random.choice(['regulated', 'secure', 'user-friendly', 'feature-rich'])} and serves both beginners and experienced traders.",
            'other': f"{fake.company()} is an innovative company that {fake.bs()} in the {random.choice(['technology', 'services', 'product', 'platform'])} space. We've developed unique solutions that address {random.choice(['market needs', 'customer pain points', 'industry challenges', 'business opportunities'])}. Our company has {random.choice(['strong growth potential', 'proven traction', 'innovative technology', 'experienced team'])} and is positioned for success."
        }
        return descriptions.get(category, descriptions['other'])

    def create_startup_tags(self, startups):
        """Create relevant tags for startups"""
        self.stdout.write('Creating startup tags...')
        
        category_tag_map = {
            'saas': ['SaaS', 'B2B', 'Subscription', 'cloud'],
            'ecommerce': ['E-commerce', 'B2C', 'Marketplace', 'retail'],
            'web3': ['Blockchain', 'Web3', 'Crypto', 'decentralized'],
            'crypto': ['Blockchain', 'Crypto', 'Web3', 'trading'],
            'software': ['Software', 'B2B', 'Enterprise', 'API'],
            'platform': ['Platform', 'API', 'scalable', 'B2B'],
            'media': ['Media', 'Content', 'B2C', 'entertainment'],
            'fintech': ['Fintech', 'Finance', 'B2B', 'payments'],
            'healthtech': ['Healthtech', 'Healthcare', 'B2B', 'AI'],
            'edtech': ['Edtech', 'Education', 'B2C', 'learning']
        }
        
        total_tags = 0
        for i, startup in enumerate(startups):
            num_tags = random.randint(3, 8)
            
            # Start with category-relevant tags
            relevant_tags = category_tag_map.get(startup.category, [])
            selected_tags = random.sample(relevant_tags, min(2, len(relevant_tags))) if relevant_tags else []
            
            # Add field-relevant tags
            if startup.field in ['fintech', 'healthtech', 'edtech']:
                field_tag = startup.field
                if field_tag not in selected_tags and field_tag in TAGS:
                    selected_tags.append(field_tag)
            
            # Fill remaining with random tags
            available_tags = [t for t in TAGS if t not in selected_tags]
            remaining = num_tags - len(selected_tags)
            if available_tags and remaining > 0:
                selected_tags.extend(random.sample(available_tags, min(remaining, len(available_tags))))
            
            # Add stage/status tags
            if random.random() < 0.3:
                if 'bootstrapped' in TAGS and 'bootstrapped' not in selected_tags:
                    selected_tags.append('bootstrapped')
            if random.random() < 0.2:
                if 'funded' in TAGS and 'funded' not in selected_tags:
                    selected_tags.append('funded')
            if random.random() < 0.1:
                if 'remote-first' in TAGS and 'remote-first' not in selected_tags:
                    selected_tags.append('remote-first')
            
            # Limit to max 8 tags
            selected_tags = list(set(selected_tags))[:8]
            
            for tag in selected_tags:
                StartupTag.objects.get_or_create(startup=startup, tag=tag)
                total_tags += 1
            
            if (i + 1) % 50 == 0:
                self.stdout.write(f'  Created tags for {i + 1}/{len(startups)} startups...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {total_tags} startup tags'))

    def create_positions(self, collaboration_startups):
        """Create positions for collaboration startups"""
        self.stdout.write('Creating positions...')
        
        tech_positions = [
            ('Full Stack Developer', 'We are looking for an experienced Full Stack Developer to join our team. You will be responsible for developing and maintaining web applications using modern technologies.', '3+ years of experience with React, Node.js, and databases. Strong problem-solving skills.'),
            ('Backend Engineer', 'Join our backend team to build scalable and efficient server-side applications. You will work on API development and database design.', 'Experience with Python/Django or Node.js, RESTful APIs, database design.'),
            ('Frontend Developer', 'We need a talented Frontend Developer to create beautiful and responsive user interfaces.', 'Proficiency in React, Vue.js, or Angular. Strong CSS/HTML skills.'),
            ('DevOps Engineer', 'Looking for a DevOps Engineer to manage our infrastructure and deployment pipelines.', 'Experience with Docker, Kubernetes, CI/CD pipelines, AWS/Azure/GCP.'),
            ('Mobile Developer', 'We are seeking a Mobile Developer to build native or cross-platform mobile applications.', 'Experience with Swift/Kotlin or React Native/Flutter.'),
            ('Data Engineer', 'Join our data team to build and maintain data pipelines and infrastructure.', 'Experience with Python, SQL, data pipelines, ETL processes.'),
            ('ML Engineer', 'We are looking for a Machine Learning Engineer to develop and deploy ML models.', 'Strong background in machine learning, Python, TensorFlow/PyTorch.'),
        ]
        
        business_positions = [
            ('Marketing Manager', 'We need a Marketing Manager to lead our marketing efforts and drive growth.', '3+ years of marketing experience, strong analytical skills.'),
            ('Sales Lead', 'Join our sales team to drive revenue growth and build relationships with clients.', 'Proven sales track record, excellent communication skills.'),
            ('Business Development Manager', 'We are looking for a Business Development Manager to identify new business opportunities.', 'Strong business acumen, networking skills.'),
            ('Product Manager', 'Join our product team to define and execute product strategy.', 'Experience in product management, strong analytical skills.'),
            ('Growth Hacker', 'We need a Growth Hacker to drive user acquisition and retention.', 'Experience with growth marketing, A/B testing.'),
        ]
        
        design_positions = [
            ('UI/UX Designer', 'We are looking for a talented UI/UX Designer to create intuitive user experiences.', 'Strong portfolio, experience with Figma/Adobe XD.'),
            ('Graphic Designer', 'Join our design team to create visual content and branding materials.', 'Proficiency in Adobe Creative Suite, strong portfolio.'),
        ]
        
        all_positions = tech_positions + business_positions + design_positions
        created_positions = []
        
        for startup in collaboration_startups:
            num_positions = random.randint(1, 5)
            selected_positions = random.sample(all_positions, min(num_positions, len(all_positions)))
            
            for title, description, requirements in selected_positions:
                position = Position.objects.create(
                    startup=startup,
                    title=title,
                    description=description,
                    requirements=requirements,
                    is_active=random.choice([True] * 8 + [False] * 2)
                )
                created_positions.append(position)
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(created_positions)} positions'))
        return created_positions

    def calculate_match_score(self, user, startup):
        """Calculate how well a startup matches user preferences"""
        try:
            prefs = user.onboarding_preferences
        except UserOnboardingPreferences.DoesNotExist:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Category match (weight: 2.0)
        max_score += 2.0
        if startup.category in prefs.selected_categories:
            score += 2.0
        
        # Field match (weight: 1.5)
        max_score += 1.5
        if startup.field in prefs.selected_fields:
            score += 1.5
        
        # Stage match (weight: 1.0)
        max_score += 1.0
        startup_stages = startup.stages if isinstance(startup.stages, list) else []
        if any(stage in prefs.preferred_startup_stages for stage in startup_stages):
            score += 1.0
        
        # Tag match (weight: 0.5 per matching tag, max 2.0)
        max_score += 2.0
        startup_tags = list(StartupTag.objects.filter(startup=startup).values_list('tag', flat=True))
        matching_tags = set(startup_tags) & set(prefs.selected_tags)
        score += min(len(matching_tags) * 0.5, 2.0)
        
        # Role-specific matching
        if user.role == 'student':
            # Check if startup has positions with matching skills
            positions = Position.objects.filter(startup=startup, is_active=True)
            if positions.exists():
                for position in positions:
                    req_lower = position.requirements.lower()
                    for skill in prefs.preferred_skills:
                        if skill.lower() in req_lower:
                            score += 0.5
                            max_score += 0.5
                            break
        
        elif user.role == 'investor':
            # Prefer marketplace startups with revenue
            if startup.type == 'marketplace' and startup.revenue:
                score += 1.0
                max_score += 1.0
            # Prefer funded/bootstrapped tags
            if any(tag in ['funded', 'bootstrapped'] for tag in startup_tags):
                score += 0.5
                max_score += 0.5
        
        # Normalize score
        if max_score > 0:
            return score / max_score
        return 0.0

    def create_user_interactions(self, users, startups, positions, num_interactions):
        """Create user interactions with preference alignment"""
        self.stdout.write('Creating user interactions...')
        
        # Interaction type distribution
        interaction_weights = {
            'view': 0.45,
            'click': 0.22,
            'like': 0.12,
            'favorite': 0.08,
            'apply': 0.08,
            'interest': 0.04,
            'dislike': 0.01
        }
        
        # Generate interaction types based on distribution
        interaction_types = []
        for interaction_type, weight in interaction_weights.items():
            count = int(num_interactions * weight)
            interaction_types.extend([interaction_type] * count)
        
        # Fill remaining to reach exact count
        while len(interaction_types) < num_interactions:
            interaction_types.append(random.choices(
                list(interaction_weights.keys()),
                weights=list(interaction_weights.values())
            )[0])
        
        random.shuffle(interaction_types)
        
        # Timestamp distribution (last 6 months, weighted toward recent)
        now = timezone.now()
        base_date = now - timedelta(days=180)
        
        interactions_created = []
        interaction_count_by_type = defaultdict(int)
        errors = 0
        
        # Track user-startup pairs to enable multiple interactions
        user_startup_pairs = defaultdict(set)
        
        # Pre-calculate match scores for all user-startup pairs
        self.stdout.write('  Calculating match scores...')
        match_scores = {}
        for user in users:
            for startup in startups:
                match_scores[(user.id, startup.id)] = self.calculate_match_score(user, startup)
        
        # Sort startups by match score for each user (for preference alignment)
        user_startup_rankings = {}
        for user in users:
            rankings = []
            for startup in startups:
                score = match_scores[(user.id, startup.id)]
                rankings.append((startup, score))
            rankings.sort(key=lambda x: x[1], reverse=True)
            user_startup_rankings[user.id] = [s[0] for s in rankings]
        
        # Generate interactions
        for idx, interaction_type in enumerate(interaction_types):
            if (idx + 1) % 500 == 0:
                self.stdout.write(f'  Created {idx + 1}/{num_interactions} interactions...')
            
            # Select user based on interaction type
            if interaction_type == 'apply':
                # Only students and entrepreneurs can apply
                eligible_users = [u for u in users if u.role in ['student', 'entrepreneur']]
                if not eligible_users:
                    continue
                user = random.choice(eligible_users)
                # Only collaboration startups with positions
                eligible_startups = [s for s in startups if s.type == 'collaboration' and 
                                   Position.objects.filter(startup=s, is_active=True).exists()]
                if not eligible_startups:
                    continue
            elif interaction_type == 'interest':
                # Mostly investors
                if random.random() < 0.8:
                    eligible_users = [u for u in users if u.role == 'investor']
                else:
                    eligible_users = users
                if not eligible_users:
                    continue
                user = random.choice(eligible_users)
                # Prefer marketplace startups
                eligible_startups = [s for s in startups if s.type == 'marketplace' or random.random() < 0.3]
            elif interaction_type == 'favorite':
                # More common for investors
                if random.random() < 0.6:
                    eligible_users = [u for u in users if u.role == 'investor']
                else:
                    eligible_users = users
                user = random.choice(eligible_users)
                eligible_startups = startups
            else:
                user = random.choice(users)
                eligible_startups = startups
            
            if not eligible_startups:
                continue
            
            # Select startup with preference alignment (70-80% match)
            if random.random() < 0.75:  # 75% preference alignment
                # Select from top 30% of matching startups
                rankings = user_startup_rankings[user.id]
                top_n = max(1, int(len(rankings) * 0.3))
                startup = random.choice(rankings[:top_n])
            else:  # 25% noise
                startup = random.choice(eligible_startups)
            
            # Check unique constraint
            pair_key = (user.id, startup.id, interaction_type)
            if pair_key in user_startup_pairs[user.id]:
                continue
            
            # Generate timestamp (weighted toward recent dates)
            # Use exponential distribution to favor recent dates
            days_ago = int(random.expovariate(1.0 / 60))  # Mean of 60 days
            days_ago = min(days_ago, 180)  # Cap at 180 days
            created_at = base_date + timedelta(days=days_ago)
            
            # For apply interactions, select a position
            position = None
            if interaction_type == 'apply':
                available_positions = list(Position.objects.filter(startup=startup, is_active=True))
                if available_positions:
                    position = random.choice(available_positions)
                else:
                    continue  # Skip if no positions available
            
            try:
                interaction = UserInteraction.objects.create(
                    user=user,
                    startup=startup,
                    interaction_type=interaction_type,
                    position=position,
                    created_at=created_at
                )
                user_startup_pairs[user.id].add(pair_key)
                interactions_created.append(interaction)
                interaction_count_by_type[interaction_type] += 1
            except IntegrityError:
                errors += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 10:  # Only show first 10 errors
                    self.stdout.write(self.style.WARNING(f'  Error creating interaction: {str(e)}'))
                continue
        
        self.stdout.write(self.style.SUCCESS(
            f'Created {len(interactions_created)} interactions (errors: {errors})'
        ))
        self.stdout.write('  Interaction type distribution:')
        for itype, count in sorted(interaction_count_by_type.items()):
            percentage = (count / len(interactions_created) * 100) if interactions_created else 0
            self.stdout.write(f'    {itype}: {count} ({percentage:.1f}%)')
        
        return interactions_created

    def validate_data(self, users, startups, positions, interactions):
        """Validate data quality"""
        self.stdout.write('\nValidating data quality...')
        results = {
            'foreign_keys_valid': True,
            'unique_constraints_valid': True,
            'weights_valid': True,
            'timestamps_valid': True,
            'errors': []
        }
        
        # Validate foreign keys
        try:
            for interaction in interactions[:100]:  # Sample check
                if not User.objects.filter(id=interaction.user_id).exists():
                    results['foreign_keys_valid'] = False
                    results['errors'].append(f'Invalid user_id in interaction {interaction.id}')
                if not Startup.objects.filter(id=interaction.startup_id).exists():
                    results['foreign_keys_valid'] = False
                    results['errors'].append(f'Invalid startup_id in interaction {interaction.id}')
                if interaction.position_id and not Position.objects.filter(id=interaction.position_id).exists():
                    results['foreign_keys_valid'] = False
                    results['errors'].append(f'Invalid position_id in interaction {interaction.id}')
        except Exception as e:
            results['errors'].append(f'Error validating foreign keys: {str(e)}')
        
        # Validate unique constraints (check for duplicates)
        try:
            from django.db.models import Count
            duplicate_interactions = UserInteraction.objects.values(
                'user', 'startup', 'interaction_type'
            ).annotate(count=Count('id')).filter(count__gt=1)
            
            if duplicate_interactions.exists():
                results['unique_constraints_valid'] = False
                results['errors'].append(f'Found {duplicate_interactions.count()} duplicate interactions')
        except Exception as e:
            results['errors'].append(f'Error validating unique constraints: {str(e)}')
        
        # Validate weights
        try:
            weight_mapping = {
                'view': 0.5, 'click': 1.0, 'like': 2.0, 'dislike': -1.0,
                'favorite': 2.5, 'apply': 3.0, 'interest': 3.5
            }
            invalid_weights = []
            for interaction in interactions[:100]:  # Sample check
                expected_weight = weight_mapping.get(interaction.interaction_type, 1.0)
                if abs(interaction.weight - expected_weight) > 0.01:
                    invalid_weights.append(f'Interaction {interaction.id}: expected {expected_weight}, got {interaction.weight}')
            
            if invalid_weights:
                results['weights_valid'] = False
                results['errors'].extend(invalid_weights[:10])  # Limit error messages
        except Exception as e:
            results['errors'].append(f'Error validating weights: {str(e)}')
        
        # Validate timestamps (users created before interactions)
        try:
            now = timezone.now()
            for user in users[:50]:  # Sample check
                user_interactions = UserInteraction.objects.filter(user=user)
                for interaction in user_interactions[:10]:
                    if interaction.created_at < user.created_at:
                        results['timestamps_valid'] = False
                        results['errors'].append(f'Interaction {interaction.id} created before user {user.id}')
                    if interaction.created_at > now:
                        results['timestamps_valid'] = False
                        results['errors'].append(f'Interaction {interaction.id} has future timestamp')
        except Exception as e:
            results['errors'].append(f'Error validating timestamps: {str(e)}')
        
        if results['errors']:
            self.stdout.write(self.style.WARNING(f'  Found {len(results["errors"])} validation issues'))
        else:
            self.stdout.write(self.style.SUCCESS('  All validation checks passed'))
        
        return results

    def print_summary(self, entrepreneurs, students, investors, startups, positions, interactions, validation_results=None):
        """Print generation summary"""
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('RECOMMENDATION DATASET GENERATION SUMMARY'))
        self.stdout.write(self.style.SUCCESS('='*60))
        
        self.stdout.write(f'\nUsers:')
        self.stdout.write(f'  - Entrepreneurs: {len(entrepreneurs)}')
        self.stdout.write(f'  - Students: {len(students)}')
        self.stdout.write(f'  - Investors: {len(investors)}')
        self.stdout.write(f'  - Total: {len(entrepreneurs) + len(students) + len(investors)}')
        
        marketplace_count = len([s for s in startups if s.type == 'marketplace'])
        collaboration_count = len([s for s in startups if s.type == 'collaboration'])
        self.stdout.write(f'\nStartups:')
        self.stdout.write(f'  - Marketplace: {marketplace_count}')
        self.stdout.write(f'  - Collaboration: {collaboration_count}')
        self.stdout.write(f'  - Total: {len(startups)}')
        
        self.stdout.write(f'\nOther Data:')
        self.stdout.write(f'  - User Profiles: {UserProfile.objects.count()}')
        self.stdout.write(f'  - Onboarding Preferences: {UserOnboardingPreferences.objects.count()}')
        self.stdout.write(f'  - Startup Tags: {StartupTag.objects.count()}')
        self.stdout.write(f'  - Positions: {len(positions)}')
        self.stdout.write(f'  - User Interactions: {len(interactions)}')
        
        # Interaction statistics
        if interactions:
            interactions_by_type = defaultdict(int)
            for interaction in interactions:
                interactions_by_type[interaction.interaction_type] += 1
            
            self.stdout.write(f'\nInteraction Distribution:')
            for itype, count in sorted(interactions_by_type.items()):
                percentage = (count / len(interactions) * 100)
                self.stdout.write(f'  - {itype}: {count} ({percentage:.1f}%)')
        
        # Validation results
        if validation_results:
            self.stdout.write(f'\nValidation Results:')
            self.stdout.write(f'  - Foreign Keys Valid: {validation_results["foreign_keys_valid"]}')
            self.stdout.write(f'  - Unique Constraints Valid: {validation_results["unique_constraints_valid"]}')
            self.stdout.write(f'  - Weights Valid: {validation_results["weights_valid"]}')
            self.stdout.write(f'  - Timestamps Valid: {validation_results["timestamps_valid"]}')
            if validation_results['errors']:
                self.stdout.write(self.style.WARNING(f'  - Errors Found: {len(validation_results["errors"])}'))
                if len(validation_results['errors']) <= 5:
                    for error in validation_results['errors']:
                        self.stdout.write(self.style.WARNING(f'    * {error}'))
        
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('Dataset generation completed successfully!'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))

    def print_dry_run_summary(self, entrepreneurs, students, investors, startups, positions, interactions):
        """Print dry run summary"""
        self.stdout.write(self.style.WARNING('\n' + '='*60))
        self.stdout.write(self.style.WARNING('DRY RUN SUMMARY'))
        self.stdout.write(self.style.WARNING('='*60))
        
        self.stdout.write(f'\nWould create:')
        self.stdout.write(f'  - Users: {len(entrepreneurs) + len(students) + len(investors)}')
        self.stdout.write(f'  - Startups: {len(startups)}')
        self.stdout.write(f'  - Positions: {len(positions)}')
        self.stdout.write(f'  - Interactions: {len(interactions)}')
        
        self.stdout.write(self.style.WARNING('='*60 + '\n'))

