import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinLengthValidator, URLValidator
from django.utils import timezone


class User(AbstractUser):
	"""Custom User model with additional fields"""
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	email = models.EmailField(unique=True)
	is_active = models.BooleanField(default=True)
	email_verified = models.BooleanField(default=False)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	# New fields
	ROLE_CHOICES = [
		('entrepreneur', 'Entrepreneur'),
		('student', 'Student/Professional'),
		('investor', 'Investor'),
	]
	role = models.CharField(max_length=32, choices=ROLE_CHOICES, default='entrepreneur')
	phone_number = models.CharField(max_length=20, blank=True)
	
	# Embedding fields for recommendation system
	profile_embedding = models.TextField(null=True, blank=True)  # JSON string of embedding vector
	embedding_model = models.CharField(max_length=50, default='all-MiniLM-L6-v2', blank=True)
	embedding_version = models.IntegerField(default=1)
	embedding_updated_at = models.DateTimeField(null=True, blank=True)
	
	# Fix related_name conflicts with default User model
	groups = models.ManyToManyField(
		'auth.Group',
		verbose_name='groups',
		blank=True,
		help_text='The groups this user belongs to.',
		related_name='api_user_set',
		related_query_name='api_user',
	)
	user_permissions = models.ManyToManyField(
		'auth.Permission',
		verbose_name='user permissions',
		blank=True,
		help_text='Specific permissions for this user.',
		related_name='api_user_set',
		related_query_name='api_user',
	)
	
	USERNAME_FIELD = 'email'
	REQUIRED_FIELDS = ['username']
	
	class Meta:
		db_table = 'users'
		indexes = [
			models.Index(fields=['email']),
			models.Index(fields=['username']),
			models.Index(fields=['role']),
		]


class Startup(models.Model):
	"""Startup model for both marketplace and collaboration listings"""
	TYPE_CHOICES = [
		('marketplace', 'Marketplace'),
		('collaboration', 'Collaboration'),
	]
	
	CATEGORY_CHOICES = [
		('saas', 'SaaS'),
		('ecommerce', 'E-commerce'),
		('agency', 'Agency'),
		('legal', 'Legal'),
		('marketplace', 'Marketplace'),
		('media', 'Media'),
		('platform', 'Platform'),
		('real_estate', 'Real Estate'),
		('robotics', 'Robotics'),
		('software', 'Software'),
		('web3', 'Web3'),
		('crypto', 'Crypto'),
		('other', 'Other'),
	]
	
	STATUS_CHOICES = [
		('active', 'Active'),
		('inactive', 'Inactive'),
		('sold', 'Sold'),
		('paused', 'Paused'),
	]
	
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='startups')
	title = models.CharField(max_length=200, validators=[MinLengthValidator(5)])
	role_title = models.CharField(max_length=100, blank=True)
	description = models.TextField(validators=[MinLengthValidator(20)])
	field = models.CharField(max_length=100)
	website_url = models.URLField(blank=True, validators=[URLValidator()])
	stages = models.JSONField(default=list, blank=True)
	revenue = models.CharField(max_length=50, blank=True)
	profit = models.CharField(max_length=50, blank=True)
	asking_price = models.CharField(max_length=50, blank=True)
	ttm_revenue = models.CharField(max_length=50, blank=True)
	ttm_profit = models.CharField(max_length=50, blank=True)
	last_month_revenue = models.CharField(max_length=50, blank=True)
	last_month_profit = models.CharField(max_length=50, blank=True)
	earn_through = models.CharField(max_length=50, blank=True)  # For collaborations
	phase = models.CharField(max_length=50, blank=True)  # For collaborations
	team_size = models.CharField(max_length=50, blank=True)  # For collaborations
	type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='marketplace')
	category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='other')
	status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
	views = models.IntegerField(default=0)
	featured = models.BooleanField(default=False)
	
	# Embedding fields for recommendation system
	profile_embedding = models.TextField(null=True, blank=True)  # JSON string of embedding vector
	embedding_model = models.CharField(max_length=50, default='all-MiniLM-L6-v2', blank=True)
	embedding_version = models.IntegerField(default=1)
	embedding_updated_at = models.DateTimeField(null=True, blank=True)
	
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	
	class Meta:
		db_table = 'startups'
		indexes = [
			models.Index(fields=['owner']),
			models.Index(fields=['type']),
			models.Index(fields=['category']),
			models.Index(fields=['status']),
			models.Index(fields=['created_at']),
		]
	
	def __str__(self):
		return self.title


class StartupTag(models.Model):
	"""Tags for startups"""
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	startup = models.ForeignKey(Startup, on_delete=models.CASCADE, related_name='tags')
	tag = models.CharField(max_length=100)
	
	class Meta:
		db_table = 'startup_tags'
		indexes = [
			models.Index(fields=['startup', 'tag']),
		]
		unique_together = ['startup', 'tag']
	
	def __str__(self):
		return f"{self.startup.title} - {self.tag}"


class Position(models.Model):
	"""Available positions in startups"""
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	startup = models.ForeignKey(Startup, on_delete=models.CASCADE, related_name='positions')
	title = models.CharField(max_length=100)
	description = models.TextField(blank=True)
	requirements = models.TextField(blank=True)
	is_active = models.BooleanField(default=True)
	created_at = models.DateTimeField(auto_now_add=True)
	
	class Meta:
		db_table = 'positions'
		indexes = [
			models.Index(fields=['startup']),
		]
	
	def __str__(self):
		return f"{self.startup.title} - {self.title}"


class Application(models.Model):
	"""Applications for startup positions"""
	STATUS_CHOICES = [
		('pending', 'Pending'),
		('approved', 'Approved'),
		('rejected', 'Rejected'),
		('withdrawn', 'Withdrawn'),
	]
	
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	startup = models.ForeignKey(Startup, on_delete=models.CASCADE, related_name='applications')
	position = models.ForeignKey(Position, on_delete=models.CASCADE, related_name='applications')
	applicant = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')
	cover_letter = models.TextField(blank=True)
	experience = models.TextField(blank=True)
	portfolio_url = models.URLField(blank=True)
	resume_url = models.URLField(blank=True)  # CV/Resume file URL
	status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
	notes = models.TextField(blank=True)  # Admin/owner notes
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	
	class Meta:
		db_table = 'applications'
		unique_together = ['startup', 'applicant']  # User can only apply to one position per startup
		indexes = [
			models.Index(fields=['startup']),
			models.Index(fields=['applicant']),
			models.Index(fields=['status']),
		]
	
	def __str__(self):
		return f"{self.applicant.username} - {self.startup.title} - {self.position.title}"




class Notification(models.Model):
    """In-app notifications sent to users"""
    TYPE_CHOICES = [
        ('application_status', 'Application Status'),
        ('new_application', 'New Application'),
        ('pitch', 'Business Pitch'),
        ('general', 'General Notification'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    type = models.CharField(max_length=50, choices=TYPE_CHOICES)
    title = models.CharField(max_length=200)
    message = models.TextField(blank=True)
    data = models.JSONField(default=dict, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        indexes = [
            models.Index(fields=['user', 'is_read']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.type} - {self.title}"


class Favorite(models.Model):
    """User saves a startup (investor engagement)"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='favorites')
    startup = models.ForeignKey(Startup, on_delete=models.CASCADE, related_name='favorited_by')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'favorites'
        unique_together = ['user', 'startup']
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['startup']),
        ]

    def __str__(self):
        return f"{self.user.username} -> {self.startup.title}"


class Interest(models.Model):
    """Investor expresses interest in a startup"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='interests')
    startup = models.ForeignKey(Startup, on_delete=models.CASCADE, related_name='interests')
    message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'interests'
        unique_together = ['user', 'startup']
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['startup']),
        ]

    def __str__(self):
        return f"{self.user.username} -> {self.startup.title}"
