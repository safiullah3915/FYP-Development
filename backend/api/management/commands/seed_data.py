from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from faker import Faker
import random
from datetime import timedelta

from api.models import (
    User, Startup, StartupTag, Position, Application,
    Notification, Favorite, Interest
)
from api.messaging_models import (
    UserProfile, Conversation, Message, FileUpload
)

fake = Faker()


class Command(BaseCommand):
    help = 'Generate realistic seed data for all database models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing seed data before generating new data',
        )
        parser.add_argument(
            '--users',
            type=int,
            default=100,
            help='Number of users to create (default: 100)',
        )
        parser.add_argument(
            '--startups',
            type=int,
            default=120,
            help='Number of startups to create (default: 120)',
        )

    def handle(self, *args, **options):
        clear_data = options['clear']
        num_users = options['users']
        num_startups = options['startups']

        if clear_data:
            if input('Are you sure you want to clear existing seed data? (yes/no): ') != 'yes':
                self.stdout.write(self.style.WARNING('Operation cancelled.'))
                return
            self.clear_seed_data()

        self.stdout.write(self.style.SUCCESS('Starting seed data generation...'))
        
        try:
            with transaction.atomic():
                # Generate data in correct order
                entrepreneurs, students, investors = self.create_users(num_users)
                self.create_user_profiles(entrepreneurs + students + investors)
                
                marketplace_startups, collaboration_startups = self.create_startups(
                    num_startups, entrepreneurs
                )
                all_startups = marketplace_startups + collaboration_startups
                
                self.create_startup_tags(all_startups)
                positions = self.create_positions(all_startups)
                
                applications = self.create_applications(positions, students)
                self.create_favorites_and_interests(investors, marketplace_startups)
                self.create_notifications(applications, all_startups, investors)
                
                conversations = self.create_conversations(entrepreneurs + students + investors)
                self.create_messages(conversations)
                # Skip file uploads as they require actual files
                # self.create_file_uploads(entrepreneurs + students + investors, applications)
                
                self.print_summary(
                    entrepreneurs, students, investors, all_startups,
                    positions, applications, conversations
                )
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error generating seed data: {str(e)}'))
            raise

    def clear_seed_data(self):
        """Clear existing seed data in reverse dependency order"""
        self.stdout.write(self.style.WARNING('Clearing existing seed data...'))
        
        # Delete in reverse order to respect foreign key constraints
        FileUpload.objects.all().delete()
        Message.objects.all().delete()
        Conversation.objects.all().delete()
        Notification.objects.all().delete()
        Interest.objects.all().delete()
        Favorite.objects.all().delete()
        Application.objects.all().delete()
        Position.objects.all().delete()
        StartupTag.objects.all().delete()
        Startup.objects.all().delete()
        UserProfile.objects.all().delete()
        # Keep superusers and staff, delete only regular users
        User.objects.filter(is_superuser=False, is_staff=False).delete()
        
        self.stdout.write(self.style.SUCCESS('Cleared existing seed data.'))

    def create_users(self, num_users):
        """Create users with different roles"""
        self.stdout.write('Creating users...')
        
        # Calculate distribution
        num_entrepreneurs = int(num_users * 0.4)
        num_students = int(num_users * 0.4)
        num_investors = num_users - num_entrepreneurs - num_students
        
        entrepreneurs = []
        students = []
        investors = []
        
        # Create entrepreneurs
        for i in range(num_entrepreneurs):
            try:
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='entrepreneur',
                    phone_number=fake.phone_number()[:20],
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                entrepreneurs.append(user)
                if (i + 1) % 10 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_entrepreneurs} entrepreneurs...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating entrepreneur {i + 1}: {str(e)}'))
                continue
        
        # Create students/developers
        for i in range(num_students):
            try:
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='student',
                    phone_number=fake.phone_number()[:20],
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                students.append(user)
                if (i + 1) % 10 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_students} students...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating student {i + 1}: {str(e)}'))
                continue
        
        # Create investors
        for i in range(num_investors):
            try:
                user = User.objects.create_user(
                    username=fake.unique.user_name(),
                    email=fake.unique.email(),
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    role='investor',
                    phone_number=fake.phone_number()[:20],
                    email_verified=random.choice([True] * 9 + [False]),
                    is_active=True,
                )
                investors.append(user)
                if (i + 1) % 10 == 0:
                    self.stdout.write(f'  Created {i + 1}/{num_investors} investors...')
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'  Error creating investor {i + 1}: {str(e)}'))
                continue
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(entrepreneurs)} entrepreneurs, {len(students)} students, {len(investors)} investors'))
        return entrepreneurs, students, investors

    def create_user_profiles(self, users):
        """Create user profiles for all users"""
        self.stdout.write('Creating user profiles...')
        
        tech_skills = ['Python', 'JavaScript', 'React', 'Node.js', 'Django', 'Flask', 
                      'Vue.js', 'Angular', 'TypeScript', 'Java', 'C++', 'Go', 'Rust',
                      'Swift', 'Kotlin', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
                      'PostgreSQL', 'MongoDB', 'Redis', 'GraphQL', 'REST API']
        
        business_skills = ['Marketing', 'Sales', 'Business Development', 'Product Management',
                          'Project Management', 'Strategic Planning', 'Market Research',
                          'Financial Analysis', 'Fundraising', 'Partnership Development']
        
        design_skills = ['UI/UX Design', 'Graphic Design', 'Figma', 'Adobe XD', 'Sketch',
                        'Photoshop', 'Illustrator', 'Prototyping', 'User Research']
        
        for i, user in enumerate(users):
            # Select skills based on role
            if user.role == 'student':
                skills = random.sample(tech_skills, random.randint(3, 8))
                if random.random() < 0.3:
                    skills.extend(random.sample(design_skills, random.randint(1, 3)))
            elif user.role == 'entrepreneur':
                skills = random.sample(business_skills, random.randint(2, 5))
                if random.random() < 0.5:
                    skills.extend(random.sample(tech_skills, random.randint(1, 4)))
            else:  # investor
                skills = random.sample(business_skills, random.randint(3, 6))
            
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
                references=references
            )
            
            if (i + 1) % 20 == 0:
                self.stdout.write(f'  Created {i + 1}/{len(users)} profiles...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(users)} user profiles'))

    def create_startups(self, num_startups, entrepreneurs):
        """Create marketplace and collaboration startups"""
        self.stdout.write('Creating startups...')
        
        num_marketplace = int(num_startups * 0.5)
        num_collaboration = num_startups - num_marketplace
        
        categories = ['saas', 'ecommerce', 'agency', 'legal', 'marketplace', 'media',
                     'platform', 'real_estate', 'robotics', 'software', 'web3', 'crypto', 'other']
        
        marketplace_stages = [["MVP Stage"], ["Product Market Fit"], ["Growth"], ["Exit"], ["Fund raising"]]
        collaboration_stages = [["Idea Stage"], ["Building MVP"], ["MVP Stage"], ["Product Market Fit"]]
        
        phases = ["Pre-seed", "Seed", "Series A", "Bootstrap"]
        team_sizes = ["Just me", "2-3 people", "4-5 people", "6-10 people"]
        earn_through_options = ["Equity", "Revenue Share", "Equity + Revenue Share", "Salary + Equity"]
        
        statuses = ['active'] * 8 + ['inactive'] * 1 + ['paused'] * 1
        
        marketplace_startups = []
        collaboration_startups = []
        
        # Create marketplace startups
        for i in range(num_marketplace):
            stage = random.choice(marketplace_stages)
            stage_name = stage[0]
            
            # Generate financial data based on stage
            if stage_name == "MVP Stage":
                monthly_revenue = random.randint(5000, 20000)
            elif stage_name == "Product Market Fit":
                monthly_revenue = random.randint(20000, 100000)
            elif stage_name == "Growth":
                monthly_revenue = random.randint(100000, 500000)
            else:  # Exit or Fund raising
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
                description=self.generate_startup_description(random.choice(categories)),
                field=random.choice(["Technology", "Healthcare", "Finance", "Education", "E-commerce", 
                                   "Real Estate", "Media", "Legal Services", "Robotics", "Blockchain"]),
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
                category=random.choice(categories),
                status=random.choice(statuses),
                views=random.randint(0, 10000),
                featured=random.random() < 0.15
            )
            marketplace_startups.append(startup)
            if (i + 1) % 10 == 0:
                self.stdout.write(f'  Created {i + 1}/{num_marketplace} marketplace startups...')
        
        # Create collaboration startups
        for i in range(num_collaboration):
            startup = Startup.objects.create(
                owner=random.choice(entrepreneurs),
                title=fake.company() + " " + random.choice(["Platform", "Solutions", "Tech", "App", "Systems", "Labs"]),
                role_title=random.choice(["Founder", "Co-founder", "Creator", "Builder"]),
                description=self.generate_startup_description(random.choice(categories)),
                field=random.choice(["Technology", "Healthcare", "Finance", "Education", "E-commerce", 
                                   "Real Estate", "Media", "Legal Services", "Robotics", "Blockchain"]),
                website_url=fake.url() if random.random() < 0.6 else '',
                stages=random.choice(collaboration_stages),
                phase=random.choice(phases),
                team_size=random.choice(team_sizes),
                earn_through=random.choice(earn_through_options),
                type='collaboration',
                category=random.choice(categories),
                status=random.choice(statuses),
                views=random.randint(0, 5000),
                featured=random.random() < 0.1
            )
            collaboration_startups.append(startup)
            if (i + 1) % 10 == 0:
                self.stdout.write(f'  Created {i + 1}/{num_collaboration} collaboration startups...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(marketplace_startups)} marketplace and {len(collaboration_startups)} collaboration startups'))
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
        """Create tags for startups"""
        self.stdout.write('Creating startup tags...')
        
        tech_tags = ['Python', 'JavaScript', 'React', 'Node.js', 'AI', 'ML', 'Blockchain',
                    'TypeScript', 'Django', 'Flask', 'Vue.js', 'Angular', 'Swift', 'Kotlin',
                    'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP']
        
        business_tags = ['B2B', 'B2C', 'SaaS', 'Subscription', 'E-commerce', 'Marketplace',
                        'Freemium', 'Enterprise']
        
        industry_tags = ['Healthcare', 'Finance', 'Education', 'Real Estate', 'Food & Beverage',
                        'Travel', 'Fitness', 'Gaming', 'Social Media']
        
        stage_tags = ['Early-stage', 'Growth', 'Profitable', 'Scalable', 'Pre-revenue', 'Post-revenue']
        
        all_tags = tech_tags + business_tags + industry_tags + stage_tags
        
        total_tags = 0
        for i, startup in enumerate(startups):
            num_tags = random.randint(3, 6)
            selected_tags = random.sample(all_tags, min(num_tags, len(all_tags)))
            
            # Ensure at least one relevant tag based on category
            category_tag_map = {
                'saas': ['SaaS', 'B2B', 'Subscription'],
                'ecommerce': ['E-commerce', 'B2C', 'Marketplace'],
                'web3': ['Blockchain', 'Web3', 'Crypto'],
                'crypto': ['Blockchain', 'Crypto', 'Web3'],
                'software': ['Software', 'B2B', 'Enterprise'],
            }
            
            if startup.category in category_tag_map:
                relevant_tags = category_tag_map[startup.category]
                for tag in relevant_tags:
                    if tag in all_tags and tag not in selected_tags:
                        selected_tags[0] = tag
                        break
            
            for tag in selected_tags:
                StartupTag.objects.get_or_create(startup=startup, tag=tag)
                total_tags += 1
            
            if (i + 1) % 20 == 0:
                self.stdout.write(f'  Created tags for {i + 1}/{len(startups)} startups...')
        
        self.stdout.write(self.style.SUCCESS(f'Created {total_tags} startup tags'))

    def create_positions(self, startups):
        """Create positions for startups"""
        self.stdout.write('Creating positions...')
        
        tech_positions = [
            ('Full Stack Developer', 'We are looking for an experienced Full Stack Developer to join our team. You will be responsible for developing and maintaining web applications using modern technologies. Strong experience with frontend and backend frameworks required.', '3+ years of experience with React, Node.js, and databases. Strong problem-solving skills and ability to work in a fast-paced environment.'),
            ('Backend Engineer', 'Join our backend team to build scalable and efficient server-side applications. You will work on API development, database design, and system architecture.', 'Experience with Python/Django or Node.js, RESTful APIs, database design, and cloud services. Strong understanding of system architecture.'),
            ('Frontend Developer', 'We need a talented Frontend Developer to create beautiful and responsive user interfaces. You will work closely with designers and backend developers to deliver exceptional user experiences.', 'Proficiency in React, Vue.js, or Angular. Strong CSS/HTML skills, experience with state management, and understanding of UX principles.'),
            ('DevOps Engineer', 'Looking for a DevOps Engineer to manage our infrastructure and deployment pipelines. You will ensure our systems are scalable, secure, and reliable.', 'Experience with Docker, Kubernetes, CI/CD pipelines, AWS/Azure/GCP, and monitoring tools. Strong scripting skills.'),
            ('Mobile Developer', 'We are seeking a Mobile Developer to build native or cross-platform mobile applications. You will work on iOS and/or Android development.', 'Experience with Swift/Kotlin or React Native/Flutter. Strong understanding of mobile UI/UX and app store guidelines.'),
            ('Data Engineer', 'Join our data team to build and maintain data pipelines and infrastructure. You will work with large datasets and help drive data-driven decisions.', 'Experience with Python, SQL, data pipelines, ETL processes, and big data technologies. Strong analytical skills.'),
            ('ML Engineer', 'We are looking for a Machine Learning Engineer to develop and deploy ML models. You will work on AI-powered features and improve our algorithms.', 'Strong background in machine learning, Python, TensorFlow/PyTorch, and data science. Experience with model deployment and MLOps.'),
        ]
        
        business_positions = [
            ('Marketing Manager', 'We need a Marketing Manager to lead our marketing efforts and drive growth. You will develop and execute marketing strategies across multiple channels.', '3+ years of marketing experience, strong analytical skills, experience with digital marketing tools, and excellent communication skills.'),
            ('Sales Lead', 'Join our sales team to drive revenue growth and build relationships with clients. You will identify opportunities and close deals.', 'Proven sales track record, excellent communication skills, ability to build relationships, and experience with CRM tools.'),
            ('Business Development Manager', 'We are looking for a Business Development Manager to identify and pursue new business opportunities. You will build partnerships and expand our market presence.', 'Strong business acumen, networking skills, experience with partnerships, and ability to negotiate deals.'),
            ('Product Manager', 'Join our product team to define and execute product strategy. You will work with engineering and design to deliver great products.', 'Experience in product management, strong analytical skills, ability to prioritize, and excellent communication skills.'),
            ('Growth Hacker', 'We need a Growth Hacker to drive user acquisition and retention. You will experiment with different growth strategies and optimize our funnel.', 'Experience with growth marketing, A/B testing, analytics tools, and creative problem-solving skills.'),
        ]
        
        design_positions = [
            ('UI/UX Designer', 'We are looking for a talented UI/UX Designer to create intuitive and beautiful user experiences. You will work on user research, wireframing, and design.', 'Strong portfolio, experience with Figma/Adobe XD, understanding of UX principles, and ability to work with developers.'),
            ('Graphic Designer', 'Join our design team to create visual content and branding materials. You will work on marketing materials, social media, and brand identity.', 'Proficiency in Adobe Creative Suite, strong portfolio, creativity, and attention to detail.'),
        ]
        
        other_positions = [
            ('Content Writer', 'We need a Content Writer to create engaging content for our blog, website, and marketing materials. You will help tell our story and engage our audience.', 'Excellent writing skills, experience with content creation, SEO knowledge, and ability to adapt tone and style.'),
            ('Data Analyst', 'Join our analytics team to analyze data and provide insights. You will work with various data sources and help drive data-driven decisions.', 'Strong analytical skills, experience with SQL, Excel, and data visualization tools. Ability to communicate insights effectively.'),
            ('Operations Manager', 'We are looking for an Operations Manager to streamline our operations and improve efficiency. You will work on process improvement and team coordination.', 'Experience in operations, strong organizational skills, ability to manage multiple projects, and problem-solving skills.'),
        ]
        
        all_positions = tech_positions + business_positions + design_positions + other_positions
        created_positions = []
        
        for startup in startups:
            # More positions for collaboration startups
            if startup.type == 'collaboration':
                num_positions = random.randint(2, 3)
            else:
                num_positions = random.randint(1, 2)
            
            selected_positions = random.sample(all_positions, min(num_positions, len(all_positions)))
            
            for title, description, requirements in selected_positions:
                position = Position.objects.create(
                    startup=startup,
                    title=title,
                    description=description,
                    requirements=requirements,
                    is_active=random.choice([True] * 9 + [False])
                )
                created_positions.append(position)
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(created_positions)} positions'))
        return created_positions

    def create_applications(self, positions, students):
        """Create applications from students to positions"""
        self.stdout.write('Creating applications...')
        
        statuses = ['pending'] * 60 + ['approved'] * 25 + ['rejected'] * 10 + ['withdrawn'] * 5
        
        applications = []
        applied_combinations = set()  # Track (startup, applicant) to respect unique constraint
        
        # Target 80-150 applications
        target_applications = random.randint(80, min(150, len(positions) * 2))
        
        for _ in range(target_applications):
            position = random.choice(positions)
            applicant = random.choice(students)
            
            # Check unique constraint
            if (position.startup.id, applicant.id) in applied_combinations:
                continue
            
            applied_combinations.add((position.startup.id, applicant.id))
            
            status = random.choice(statuses)
            
            cover_letter = f"""Dear {position.startup.title} Team,

I am writing to express my strong interest in the {position.title} position. I have been following {position.startup.title} and am impressed by your work in {position.startup.field}.

With my background in {', '.join(random.sample(['software development', 'web development', 'data analysis', 'design', 'marketing'], 2))}, I believe I can contribute significantly to your team. I am particularly excited about the opportunity to work on {random.choice(['innovative projects', 'cutting-edge technology', 'meaningful products', 'scalable solutions'])}.

I would welcome the opportunity to discuss how my skills and experience align with your needs.

Best regards,
{applicant.first_name} {applicant.last_name}"""
            
            experience_text = f"""I have {random.randint(1, 5)} years of experience in {random.choice(['software development', 'web development', 'data analysis', 'design', 'marketing'])}. 

Previous roles:
- {fake.job()} at {fake.company()} ({random.randint(1, 3)} years)
- {fake.job()} at {fake.company()} ({random.randint(1, 2)} years)

Key achievements:
- {fake.sentence()}
- {fake.sentence()}
- {fake.sentence()}"""
            
            notes = ''
            if status == 'approved':
                notes = 'Strong candidate with relevant experience. Recommended for next round.'
            elif status == 'rejected':
                notes = 'Candidate does not meet the required qualifications.'
            
            application = Application.objects.create(
                startup=position.startup,
                position=position,
                applicant=applicant,
                cover_letter=cover_letter,
                experience=experience_text,
                portfolio_url=fake.url() if random.random() < 0.7 else '',
                resume_url=fake.url() if random.random() < 0.5 else '',
                status=status,
                notes=notes
            )
            applications.append(application)
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(applications)} applications'))
        return applications

    def create_favorites_and_interests(self, investors, marketplace_startups):
        """Create favorites and interests for investors"""
        self.stdout.write('Creating favorites and interests...')
        
        # Create favorites (40-80)
        num_favorites = random.randint(40, min(80, len(investors) * len(marketplace_startups) // 10))
        favorites_created = 0
        
        for _ in range(num_favorites * 2):  # Try more to account for unique constraints
            if favorites_created >= num_favorites:
                break
            
            investor = random.choice(investors)
            startup = random.choice(marketplace_startups)
            
            favorite, created = Favorite.objects.get_or_create(
                user=investor,
                startup=startup
            )
            if created:
                favorites_created += 1
        
        # Create interests (30-60)
        num_interests = random.randint(30, min(60, len(investors) * len(marketplace_startups) // 15))
        interests_created = 0
        
        interest_messages = [
            "I'm interested in learning more about your startup and potential investment opportunities.",
            "Your startup shows great potential. I'd like to discuss investment possibilities.",
            "I'm impressed by your growth and would like to explore how I can support your journey.",
            "Your business model is compelling. Let's schedule a call to discuss investment.",
            "I see strong potential in your market. Interested in discussing funding opportunities.",
        ]
        
        # Focus on startups with better financials
        high_value_startups = [s for s in marketplace_startups if s.status == 'active' and random.random() < 0.6]
        if not high_value_startups:
            high_value_startups = marketplace_startups
        
        for _ in range(num_interests * 2):
            if interests_created >= num_interests:
                break
            
            investor = random.choice(investors)
            startup = random.choice(high_value_startups)
            
            interest, created = Interest.objects.get_or_create(
                user=investor,
                startup=startup,
                defaults={'message': random.choice(interest_messages)}
            )
            if created:
                interests_created += 1
        
        self.stdout.write(self.style.SUCCESS(f'Created {favorites_created} favorites and {interests_created} interests'))

    def create_notifications(self, applications, startups, investors):
        """Create notifications"""
        self.stdout.write('Creating notifications...')
        
        notifications = []
        
        # Application status notifications (40%)
        num_app_status = int(len(applications) * 0.4)
        for application in random.sample(applications, min(num_app_status, len(applications))):
            if application.status == 'approved':
                title = f"Application Approved - {application.startup.title}"
                message = f"Congratulations! Your application for {application.position.title} at {application.startup.title} has been approved."
            elif application.status == 'rejected':
                title = f"Application Update - {application.startup.title}"
                message = f"Thank you for your interest. Unfortunately, your application for {application.position.title} at {application.startup.title} was not selected at this time."
            else:
                continue
            
            notification = Notification.objects.create(
                user=application.applicant,
                type='application_status',
                title=title,
                message=message,
                data={'application_id': str(application.id), 'startup_id': str(application.startup.id)},
                is_read=random.choice([True] * 6 + [False] * 4)
            )
            notifications.append(notification)
        
        # New application notifications (30%)
        num_new_app = int(len(applications) * 0.3)
        for application in random.sample(applications, min(num_new_app, len(applications))):
            notification = Notification.objects.create(
                user=application.startup.owner,
                type='new_application',
                title=f"New Application - {application.position.title}",
                message=f"{application.applicant.first_name} {application.applicant.last_name} has applied for the {application.position.title} position.",
                data={'application_id': str(application.id), 'applicant_id': str(application.applicant.id)},
                is_read=random.choice([True] * 5 + [False] * 5)
            )
            notifications.append(notification)
        
        # Pitch notifications (10%)
        num_pitch = random.randint(10, 20)
        for _ in range(num_pitch):
            investor = random.choice(investors)
            startup = random.choice(startups)
            notification = Notification.objects.create(
                user=investor,
                type='pitch',
                title=f"New Business Pitch - {startup.title}",
                message=f"{startup.title} is looking for investors. Check out their pitch!",
                data={'startup_id': str(startup.id)},
                is_read=random.choice([True] * 4 + [False] * 6)
            )
            notifications.append(notification)
        
        # General notifications (20%)
        num_general = random.randint(20, 40)
        general_messages = [
            "Welcome to the platform! Explore startups and opportunities.",
            "New featured startups are available. Check them out!",
            "Your profile has been viewed by potential collaborators.",
            "Weekly digest: New opportunities matching your interests.",
            "Platform update: New features are now available.",
        ]
        all_users = list(set([app.applicant for app in applications] + [s.owner for s in startups] + investors))
        for _ in range(num_general):
            user = random.choice(all_users)
            notification = Notification.objects.create(
                user=user,
                type='general',
                title=random.choice(["Platform Update", "New Opportunities", "Welcome", "Weekly Digest"]),
                message=random.choice(general_messages),
                data={},
                is_read=random.choice([True] * 6 + [False] * 4)
            )
            notifications.append(notification)
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(notifications)} notifications'))

    def create_conversations(self, users):
        """Create conversations between users"""
        self.stdout.write('Creating conversations...')
        
        num_conversations = random.randint(30, min(60, len(users) * 2))
        conversations = []
        used_pairs = set()
        
        conversation_titles = [
            "Investment Discussion",
            "Job Opportunity",
            "Collaboration Inquiry",
            "Partnership Discussion",
            "Startup Inquiry",
            "Networking",
        ]
        
        for _ in range(num_conversations * 3):  # Try more to get enough unique pairs
            if len(conversations) >= num_conversations:
                break
            
            user1 = random.choice(users)
            user2 = random.choice(users)
            
            if user1.id == user2.id:
                continue
            
            pair = tuple(sorted([user1.id, user2.id]))
            if pair in used_pairs:
                continue
            
            used_pairs.add(pair)
            
            conversation = Conversation.objects.create(
                title=random.choice(conversation_titles) if random.random() < 0.5 else '',
                is_active=random.choice([True] * 9 + [False])
            )
            conversation.participants.add(user1, user2)
            conversations.append(conversation)
        
        self.stdout.write(self.style.SUCCESS(f'Created {len(conversations)} conversations'))
        return conversations

    def create_messages(self, conversations):
        """Create messages in conversations"""
        self.stdout.write('Creating messages...')
        
        total_messages = 0
        
        message_templates = [
            "Hi! I'm interested in discussing {topic}.",
            "Thanks for reaching out. I'd love to learn more about {topic}.",
            "That sounds great! When would be a good time to connect?",
            "I've reviewed your proposal and I'm interested in moving forward.",
            "Could you provide more details about {topic}?",
            "I think there's a great opportunity here. Let's schedule a call.",
            "Thanks for the information. I'll get back to you soon.",
            "I'm excited about the potential collaboration. What are the next steps?",
        ]
        
        for conversation in conversations:
            participants = list(conversation.participants.all())
            if len(participants) < 2:
                continue
            
            num_messages = random.randint(3, 10)
            topic = random.choice(["the opportunity", "your startup", "investment", "collaboration", "the project"])
            
            for i in range(num_messages):
                sender = participants[i % 2]  # Alternate between participants
                content = random.choice(message_templates).format(topic=topic)
                
                # Make messages more natural
                if i > 0:
                    content = fake.sentence() + " " + content
                
                message = Message.objects.create(
                    conversation=conversation,
                    sender=sender,
                    content=content,
                    message_type=random.choice(['text'] * 9 + ['image', 'file']),
                    is_read=random.choice([True] * 7 + [False] * 3) if i < num_messages - 1 else False
                )
                total_messages += 1
        
        self.stdout.write(self.style.SUCCESS(f'Created {total_messages} messages'))

    def create_file_uploads(self, users, applications):
        """Create file uploads (skipped - file field is required and we cannot create actual files)"""
        self.stdout.write('Skipping file uploads (file field is required, actual files cannot be created in seed data)...')
        self.stdout.write(self.style.WARNING('FileUpload records require actual file uploads. Skipping this step.'))
        # Note: FileUpload model requires an actual file, so we skip creating these records
        # In production, files would be uploaded through the API

    def print_summary(self, entrepreneurs, students, investors, startups, positions, applications, conversations):
        """Print summary statistics"""
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        self.stdout.write(self.style.SUCCESS('SEED DATA GENERATION SUMMARY'))
        self.stdout.write(self.style.SUCCESS('='*50))
        
        self.stdout.write(f'\nUsers:')
        self.stdout.write(f'  - Entrepreneurs: {len(entrepreneurs)}')
        self.stdout.write(f'  - Students/Developers: {len(students)}')
        self.stdout.write(f'  - Investors: {len(investors)}')
        self.stdout.write(f'  - Total: {len(entrepreneurs) + len(students) + len(investors)}')
        
        self.stdout.write(f'\nStartups:')
        marketplace_count = len([s for s in startups if s.type == 'marketplace'])
        collaboration_count = len([s for s in startups if s.type == 'collaboration'])
        self.stdout.write(f'  - Marketplace: {marketplace_count}')
        self.stdout.write(f'  - Collaboration: {collaboration_count}')
        self.stdout.write(f'  - Total: {len(startups)}')
        
        self.stdout.write(f'\nOther Data:')
        self.stdout.write(f'  - User Profiles: {UserProfile.objects.count()}')
        self.stdout.write(f'  - Startup Tags: {StartupTag.objects.count()}')
        self.stdout.write(f'  - Positions: {len(positions)}')
        self.stdout.write(f'  - Applications: {len(applications)}')
        self.stdout.write(f'  - Favorites: {Favorite.objects.count()}')
        self.stdout.write(f'  - Interests: {Interest.objects.count()}')
        self.stdout.write(f'  - Notifications: {Notification.objects.count()}')
        self.stdout.write(f'  - Conversations: {len(conversations)}')
        self.stdout.write(f'  - Messages: {Message.objects.count()}')
        self.stdout.write(f'  - File Uploads: {FileUpload.objects.count()} (skipped - requires actual files)')
        
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        self.stdout.write(self.style.SUCCESS('Seed data generation completed successfully!'))
        self.stdout.write(self.style.SUCCESS('='*50 + '\n'))

