from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import bcrypt
from .models import Startup, StartupTag, Position, Application, Notification, Favorite, Interest
from .messaging_models import Conversation, Message, UserProfile, FileUpload
from .recommendation_models import UserInteraction, UserOnboardingPreferences, StartupTrendingMetrics, RecommendationModel

User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
	"""Serializer for user registration"""
	password = serializers.CharField(write_only=True, min_length=8)
	role = serializers.ChoiceField(choices=User.ROLE_CHOICES, default='entrepreneur')
	phone_number = serializers.CharField(required=False, allow_blank=True)
	
	class Meta:
		model = User
		fields = ('username', 'email', 'password', 'role', 'phone_number')
	
	def validate_password(self, value):
		try:
			validate_password(value)
		except ValidationError as e:
			raise serializers.ValidationError(e.messages)
		return value
	
	def validate_email(self, value):
		if User.objects.filter(email=value).exists():
			raise serializers.ValidationError("User already exists")
		return value
	
	def create(self, validated_data):
		password = validated_data.pop('password')
		user = User(**validated_data)
		# Hash password with bcrypt
		hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
		user.password = hashed_password.decode('utf-8')
		user.save()
		return user


class UserLoginSerializer(serializers.Serializer):
	"""Serializer for user login"""
	email = serializers.EmailField()
	password = serializers.CharField()


class UserSerializer(serializers.ModelSerializer):
	"""Serializer for user data"""
	stats = serializers.SerializerMethodField()
	
	class Meta:
		model = User
		fields = ('id', 'username', 'email', 'created_at', 'stats', 'role', 'email_verified')
	
	def get_stats(self, obj):
		return {
			'startupsCreated': obj.startups.count(),
			'applicationsSubmitted': obj.applications.count(),
			'collaborations': obj.applications.filter(status='approved').count()
		}


class StartupTagSerializer(serializers.ModelSerializer):
	"""Serializer for startup tags"""
	class Meta:
		model = StartupTag
		fields = ('tag',)


class PositionSerializer(serializers.ModelSerializer):
	"""Serializer for positions"""
	applications_count = serializers.SerializerMethodField()
	startup = serializers.SerializerMethodField()

	class Meta:
		model = Position
		fields = ('id', 'title', 'description', 'requirements', 'is_active', 'applications_count', 'created_at', 'startup')

	def get_applications_count(self, obj):
		return obj.applications.count()

	def get_startup(self, obj):
		# Provide minimal startup info needed by JobCard
		# Handle missing fields gracefully for older startups
		return {
			'id': str(obj.startup.id),
			'title': obj.startup.title or '',
			'category': obj.startup.category or '',
			'earn_through': getattr(obj.startup, 'earn_through', None) or '',
			'team_size': getattr(obj.startup, 'team_size', None) or '',
			'phase': getattr(obj.startup, 'phase', None) or '',
		}


class StartupListSerializer(serializers.ModelSerializer):
	"""Serializer for startup list view"""
	owner = UserSerializer(read_only=True)
	tag = serializers.SerializerMethodField()
	name = serializers.CharField(source='title', read_only=True)
	
	class Meta:
		model = Startup
		fields = (
			'id', 'name', 'title', 'description', 'revenue', 'profit', 
			'asking_price', 'tag', 'type', 'field', 'category', 'created_at', 
			'owner', 'earn_through', 'phase', 'team_size'
		)
	
	def get_tag(self, obj):
		# Return first tag or a default based on type
		first_tag = obj.tags.first()
		if first_tag:
			return first_tag.tag
		return "Fund Raising" if obj.type == 'marketplace' else "Open to Collaborate"


class StartupDetailSerializer(serializers.ModelSerializer):
	"""Serializer for startup detail view"""
	owner = UserSerializer(read_only=True)
	tags = serializers.SerializerMethodField()
	positions = PositionSerializer(many=True, read_only=True)
	performance = serializers.SerializerMethodField()
	name = serializers.CharField(source='title', read_only=True)
	
	class Meta:
		model = Startup
		fields = (
			'id', 'name', 'title', 'description', 'tags', 'performance', 
			'positions', 'owner', 'type', 'field', 'category', 
			'revenue', 'profit', 'asking_price',
			'ttm_revenue', 'ttm_profit', 'last_month_revenue', 'last_month_profit',
			'earn_through', 'phase', 'team_size', 'website_url',
			'created_at', 'updated_at'
		)
	
	def get_tags(self, obj):
		# Handle missing tags gracefully
		try:
			return [tag.tag for tag in obj.tags.all() if tag.tag]
		except Exception as e:
			print(f"⚠️ Warning: Error getting tags for startup {obj.id}: {str(e)}")
			return []
	
	def get_performance(self, obj):
		# Handle missing fields gracefully for older startups
		return {
			'ttmRevenue': getattr(obj, 'ttm_revenue', None) or '$0',
			'ttmProfit': getattr(obj, 'ttm_profit', None) or '$0',
			'lastMonthRevenue': getattr(obj, 'last_month_revenue', None) or '$0',
			'lastMonthProfit': getattr(obj, 'last_month_profit', None) or '$0',
		}


class StartupCreateSerializer(serializers.ModelSerializer):
	"""Serializer for creating startups"""
	stages = serializers.ListField(child=serializers.CharField(), required=False)
	
	class Meta:
		model = Startup
		fields = (
			'title', 'role_title', 'description', 'field', 'website_url',
			'stages', 'revenue', 'profit', 'asking_price', 'ttm_revenue',
			'ttm_profit', 'last_month_revenue', 'last_month_profit', 'type',
			'earn_through', 'phase', 'team_size', 'category'
		)
	
	def validate(self, data):
		print(f"📋 StartupCreateSerializer: Received data:")
		for key, value in data.items():
			print(f"- {key}: {value} (type: {type(value).__name__})")
		return data
	
	def create(self, validated_data):
		print(f"✅ StartupCreateSerializer: Creating startup with validated data")
		# Respect owner provided by the view (serializer.save(owner=user))
		owner = validated_data.get('owner')
		if owner is None:
			# Fallback to authenticated request.user only if it's a real user
			req_user = self.context['request'].user if 'request' in self.context else None
			if getattr(req_user, 'is_authenticated', False):
				validated_data['owner'] = req_user
			else:
				raise serializers.ValidationError({'owner': 'Authentication required'})
		return super().create(validated_data)


class ApplicationSerializer(serializers.ModelSerializer):
	"""Serializer for applications"""
	startup = StartupListSerializer(read_only=True)
	position = PositionSerializer(read_only=True)
	applicant = UserSerializer(read_only=True)
	
	class Meta:
		model = Application
		fields = (
			'id', 'startup', 'position', 'applicant', 'cover_letter',
			'experience', 'portfolio_url', 'resume_url', 'status', 'created_at'
		)


class ApplicationCreateSerializer(serializers.ModelSerializer):
	"""Serializer for creating applications"""
	position_id = serializers.UUIDField(write_only=True)
	resume_url = serializers.CharField(required=True, allow_blank=False)
	
	class Meta:
		model = Application
		fields = ('position_id', 'cover_letter', 'experience', 'portfolio_url', 'resume_url')
	
	def validate_resume_url(self, value):
		"""Convert relative URLs to absolute URLs"""
		if not value or value.strip() == '':
			raise serializers.ValidationError("Resume URL is required")

		value = value.strip()
		print(f"🔍 Validating resume_url: {value}")
		
		# If it's already a full URL, return as is
		if value.startswith('http://') or value.startswith('https://'):
			print(f"✅ resume_url is already a full URL")
			return value
		
		# If it's a relative path, convert to absolute URL
		if value.startswith('/'):
			# Get the request from context to build absolute URL
			request = self.context.get('request')
			if request:
				# Build absolute URL using request host and scheme
				scheme = request.scheme
				host = request.get_host()
				full_url = f"{scheme}://{host}{value}"
				print(f"✅ Converted relative URL to absolute: {full_url}")
				return full_url
			else:
				# Fallback: use Django settings
				from django.conf import settings
				# Try to get site domain from settings or use default
				domain = getattr(settings, 'ALLOWED_HOSTS', ['localhost'])[0] if hasattr(settings, 'ALLOWED_HOSTS') and settings.ALLOWED_HOSTS else 'localhost'
				full_url = f"http://{domain}{value}"
				print(f"✅ Converted relative URL to absolute (fallback): {full_url}")
				return full_url
		
		# If it doesn't start with /, it might be invalid
		print(f"⚠️ resume_url doesn't start with / or http, treating as invalid")
		raise serializers.ValidationError("Resume URL must be a valid URL or path starting with /")
	
	def validate(self, attrs):
		"""Validate that user hasn't already applied to this startup"""
		print(f"\n🔍 ApplicationCreateSerializer.validate() called")
		print(f"🔍 Attrs received: {attrs}")
		print(f"🔍 Attrs keys: {list(attrs.keys())}")
		
		user = self.context.get('user') or self.context['request'].user
		if not user or user.is_anonymous:
			print(f"❌ No authenticated user in context")
			raise serializers.ValidationError("Authentication required")
		
		print(f"✅ User found: {user.username}")
		
		position_id = attrs.get('position_id')
		print(f"🔍 Position ID: {position_id}")
		
		if not position_id:
			print(f"❌ Position ID is missing")
			raise serializers.ValidationError({"position_id": "Position ID is required"})
		
		try:
			position = Position.objects.get(id=position_id)
			print(f"✅ Position found: {position.title}")
		except Position.DoesNotExist:
			print(f"❌ Position not found: {position_id}")
			raise serializers.ValidationError({"position_id": "Position not found"})
		
		startup = position.startup
		print(f"✅ Startup: {startup.title}")

		if position.startup.owner_id == user.id:
			print(f"❌ User attempted to apply to their own startup")
			raise serializers.ValidationError({
				"error": "You cannot apply to your own startup"
			})
		
		# Check if user has already applied to any position in this startup
		existing_application = Application.objects.filter(
			startup=startup,
			applicant=user
		).first()
		
		if existing_application:
			print(f"❌ User has already applied to this startup")
			raise serializers.ValidationError({
				"error": f"You have already applied to a position at {startup.title}. You can only apply to one position per startup."
			})
		
		if not attrs.get('resume_url'):
			print(f"❌ No resume URL provided")
			raise serializers.ValidationError({"resume_url": "Resume is required"})

		print(f"✅ Validation passed")
		return attrs
	
	def create(self, validated_data):
		print(f"\n🔍 ApplicationCreateSerializer.create() called")
		print(f"🔍 Validated data: {validated_data}")
		
		position_id = validated_data.pop('position_id')
		position = Position.objects.get(id=position_id)
		validated_data['startup'] = position.startup
		validated_data['position'] = position
		# Get user from context (passed from view) or fallback to request.user
		user = self.context.get('user') or self.context['request'].user
		if not user or user.is_anonymous:
			raise serializers.ValidationError("Authentication required")
		validated_data['applicant'] = user
		
		# Ensure resume_url is properly formatted (it should already be converted in validate_resume_url)
		resume_url = validated_data.get('resume_url')
		if resume_url:
			print(f"✅ Saving resume_url: {resume_url}")
		
		return super().create(validated_data)


class UserStartupSerializer(serializers.ModelSerializer):
	"""Serializer for user's startups"""
	applications = serializers.SerializerMethodField()
	
	class Meta:
		model = Startup
		fields = (
			'id', 'title', 'description', 'type', 'category', 'status', 
			'revenue', 'profit', 'asking_price', 'ttm_revenue', 'ttm_profit',
			'last_month_revenue', 'last_month_profit', 'earn_through', 
			'phase', 'team_size', 'field', 'website_url', 'applications', 
			'views', 'created_at'
		)
	
	def get_applications(self, obj):
		return obj.applications.count()


class SearchResultSerializer(serializers.ModelSerializer):
	"""Serializer for search results"""
	relevance_score = serializers.SerializerMethodField()
	
	class Meta:
		model = Startup
		fields = ('id', 'title', 'description', 'type', 'relevance_score')
	
	def get_relevance_score(self, obj):
		# Simple relevance scoring - can be enhanced
		return 1.0


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for in-app notifications"""
    class Meta:
        model = Notification
        fields = ('id', 'type', 'title', 'message', 'data', 'is_read', 'created_at')


class UserMiniSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')


class FavoriteSerializer(serializers.ModelSerializer):
    """Serializer for saved startups (favorites)"""
    startup = StartupListSerializer(read_only=True)

    class Meta:
        model = Favorite
        fields = ('id', 'startup', 'created_at')


class InterestSerializer(serializers.ModelSerializer):
    """Serializer for investor interests"""
    startup = StartupListSerializer(read_only=True)
    user = UserMiniSerializer(read_only=True)

    class Meta:
        model = Interest
        fields = ('id', 'startup', 'user', 'message', 'created_at')


# Messaging Serializers
class MessageSerializer(serializers.ModelSerializer):
    """Serializer for messages"""
    sender = UserMiniSerializer(read_only=True)
    
    class Meta:
        model = Message
        fields = ('id', 'sender', 'content', 'message_type', 'attachment', 'is_read', 'created_at')


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for conversations"""
    participants = UserMiniSerializer(many=True, read_only=True)
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = ('id', 'participants', 'title', 'is_active', 'last_message', 'unread_count', 'created_at', 'updated_at')
    
    def get_last_message(self, obj):
        last_msg = obj.messages.last()
        if last_msg:
            return MessageSerializer(last_msg).data
        return None
    
    def get_unread_count(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.messages.filter(is_read=False).exclude(sender=request.user).count()
        return 0


class ConversationCreateSerializer(serializers.ModelSerializer):
	"""Serializer for creating conversations"""
	participant_ids = serializers.ListField(
		child=serializers.UUIDField(),
		write_only=True
	)

	class Meta:
		model = Conversation
		fields = ('title', 'participant_ids')

	def create(self, validated_data):
		participant_ids = set(str(pid) for pid in validated_data.pop('participant_ids', []))
		request = self.context.get('request')
		current_user = self.context.get('current_user')
		if not current_user and request is not None:
			candidate = getattr(request, 'user', None)
			if getattr(candidate, 'is_authenticated', False):
				current_user = candidate
		if not current_user or not getattr(current_user, 'is_authenticated', False):
			raise serializers.ValidationError({'participant_ids': 'Authentication required'})
		participant_ids.add(str(current_user.id))
		if len(participant_ids) <= 1:
			raise serializers.ValidationError({'participant_ids': 'Please include at least one other participant'})
		participants = User.objects.filter(id__in=participant_ids, is_active=True)
		if participants.count() != len(participant_ids):
			raise serializers.ValidationError({'participant_ids': 'One or more participants are invalid'})
		conversation = Conversation.objects.create(**validated_data)
		conversation.participants.set(participants)
		conversation.save(update_fields=['updated_at'])
		return conversation


class MessageCreateSerializer(serializers.ModelSerializer):
	"""Serializer for creating messages"""

	class Meta:
		model = Message
		fields = ('content', 'message_type', 'attachment')

	def create(self, validated_data):
		conversation = validated_data.get('conversation') or self.context.get('conversation')
		if conversation is None:
			raise serializers.ValidationError({'conversation': 'Conversation is required'})
		sender = validated_data.get('sender')
		if sender is None:
			current_user = self.context.get('current_user')
			if not current_user:
				request = self.context.get('request')
				candidate = getattr(request, 'user', None) if request is not None else None
				if getattr(candidate, 'is_authenticated', False):
					current_user = candidate
			if not current_user or not getattr(current_user, 'is_authenticated', False):
				raise serializers.ValidationError({'sender': 'Authentication required'})
			sender = current_user
		validated_data['sender'] = sender
		validated_data['conversation'] = conversation
		return super().create(validated_data)


# User Profile Serializers
class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile"""
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = (
            'id', 'user', 'bio', 'location', 'website', 'profile_picture',
            'is_public', 'selected_regions', 'skills', 'experience',
            'references', 'created_at', 'updated_at'
        )


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile"""
    
    class Meta:
        model = UserProfile
        fields = (
            'bio', 'location', 'website', 'profile_picture', 'is_public',
            'selected_regions', 'skills', 'experience', 'references'
        )
    
    def update(self, instance, validated_data):
        # Handle file upload for profile picture
        if 'profile_picture' in validated_data:
            if instance.profile_picture:
                instance.profile_picture.delete(save=False)
        return super().update(instance, validated_data)


# File Upload Serializers
class FileUploadSerializer(serializers.ModelSerializer):
    """Serializer for file uploads"""
    user = UserMiniSerializer(read_only=True)
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = FileUpload
        fields = (
            'id', 'user', 'file', 'file_type', 'original_name',
            'file_size', 'mime_type', 'file_url', 'created_at'
        )
    
    def get_file_url(self, obj):
        if obj.file:
            return obj.file.url
        return None


class FileUploadCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating file uploads"""
    
    class Meta:
        model = FileUpload
        fields = ('file', 'file_type')
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        validated_data['original_name'] = validated_data['file'].name
        validated_data['file_size'] = validated_data['file'].size
        validated_data['mime_type'] = validated_data['file'].content_type
        return super().create(validated_data)


# Recommendation System Serializers
class UserOnboardingPreferencesSerializer(serializers.ModelSerializer):
    """Serializer for user onboarding preferences with investor profile support"""
    
    investor_preferences = serializers.DictField(write_only=True, required=False)
    investor_profile = serializers.DictField(read_only=True)
    
    class Meta:
        model = UserOnboardingPreferences
        fields = (
            'id', 'selected_categories', 'selected_fields', 'selected_tags',
            'preferred_startup_stages', 'preferred_engagement_types',
            'preferred_skills', 'onboarding_completed', 'created_at', 'updated_at',
            'investor_profile', 'investor_preferences'
        )
        read_only_fields = ('id', 'created_at', 'updated_at', 'investor_profile')
    
    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_string(v) for v in value if v not in (None, '')]
        return [self._normalize_string(value)]
    
    def _normalize_string(self, value):
        if value is None:
            return ''
        return str(value).strip()
    
    def _normalize_slug(self, value):
        return self._normalize_string(value).lower().replace(' ', '-')
    
    def _dedup(self, seq):
        seen = set()
        result = []
        for item in seq:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
    
    def _sanitize_investor_payload(self, payload):
        if not isinstance(payload, dict):
            return {}
        
        sanitized = {}
        sanitized['thesis_summary'] = self._normalize_string(payload.get('thesis_summary'))
        sanitized['sectors'] = [self._normalize_slug(v) for v in self._as_list(payload.get('sectors'))]
        sanitized['stages'] = [self._normalize_slug(v) for v in self._as_list(payload.get('stages'))]
        sanitized['round_types'] = [self._normalize_slug(v) for v in self._as_list(payload.get('round_types'))]
        sanitized['instruments'] = [self._normalize_slug(v) for v in self._as_list(payload.get('instruments'))]
        sanitized['business_models'] = [self._normalize_slug(v) for v in self._as_list(payload.get('business_models'))]
        sanitized['geographies'] = [self._normalize_slug(v) for v in self._as_list(payload.get('geographies'))]
        sanitized['support_preferences'] = [self._normalize_slug(v) for v in self._as_list(payload.get('support_preferences'))]
        sanitized['co_investor_profile'] = [self._normalize_slug(v) for v in self._as_list(payload.get('co_investor_profile'))]
        
        sanitized['collaboration_style'] = self._normalize_slug(payload.get('collaboration_style'))
        sanitized['lead_preference'] = self._normalize_slug(payload.get('lead_preference'))
        sanitized['decision_speed'] = self._normalize_slug(payload.get('decision_speed') or 'standard')
        
        def clamp_numeric(value):
            if value in (None, ''):
                return ''
            return str(value).strip()
        
        check = payload.get('check_size') or {}
        sanitized['check_size'] = {
            'min': clamp_numeric(check.get('min')),
            'max': clamp_numeric(check.get('max')),
            'currency': (check.get('currency') or 'USD').upper()
        }
        
        ownership = payload.get('target_ownership') or {}
        sanitized['target_ownership'] = {
            'min_pct': clamp_numeric(ownership.get('min_pct')),
            'max_pct': clamp_numeric(ownership.get('max_pct')),
        }
        
        valuation = payload.get('valuation_caps') or {}
        sanitized['valuation_caps'] = {
            key: clamp_numeric(val)
            for key, val in valuation.items()
            if val not in (None, '')
        }
        
        traction = payload.get('traction') or {}
        sanitized['traction'] = {
            key: clamp_numeric(val)
            for key, val in traction.items()
            if val not in (None, '')
        }
        
        sanitized['instruments'] = self._dedup(sanitized['instruments'])
        sanitized['round_types'] = self._dedup(sanitized['round_types'])
        sanitized['sectors'] = self._dedup(sanitized['sectors'])
        sanitized['stages'] = self._dedup(sanitized['stages'])
        sanitized['business_models'] = self._dedup(sanitized['business_models'])
        sanitized['geographies'] = self._dedup(sanitized['geographies'])
        sanitized['support_preferences'] = self._dedup(sanitized['support_preferences'])
        sanitized['co_investor_profile'] = self._dedup(sanitized['co_investor_profile'])
        
        return sanitized
    
    def _merge_investor_preferences(self, attrs, investor_payload):
        if not investor_payload:
            return attrs
        
        selected_categories = list(attrs.get('selected_categories') or [])
        selected_fields = list(attrs.get('selected_fields') or [])
        selected_tags = list(attrs.get('selected_tags') or [])
        preferred_startup_stages = list(attrs.get('preferred_startup_stages') or [])
        preferred_engagement_types = list(attrs.get('preferred_engagement_types') or [])
        
        selected_categories.extend(investor_payload.get('sectors', []))
        preferred_startup_stages.extend(investor_payload.get('stages', []))
        
        for bm in investor_payload.get('business_models', []):
            selected_tags.append(f"bm:{bm}")
        
        for geo in investor_payload.get('geographies', []):
            selected_tags.append(f"geo:{geo}")
        
        for support in investor_payload.get('support_preferences', []):
            selected_tags.append(f"support:{support}")
        
        collaboration_style = investor_payload.get('collaboration_style')
        if collaboration_style:
            selected_tags.append(f"collab:{collaboration_style}")
        
        decision_speed = investor_payload.get('decision_speed')
        if decision_speed:
            selected_tags.append(f"decision:{decision_speed}")
        
        for r in investor_payload.get('round_types', []):
            preferred_engagement_types.append(f"round:{r}")
        for instr in investor_payload.get('instruments', []):
            preferred_engagement_types.append(f"instr:{instr}")
        
        lead_pref = investor_payload.get('lead_preference')
        if lead_pref:
            selected_tags.append(f"leadpref:{lead_pref}")
        
        coinvest = investor_payload.get('co_investor_profile', [])
        for partner in coinvest:
            selected_tags.append(f"coinvest:{partner}")
        
        check = investor_payload.get('check_size') or {}
        if check.get('min'):
            selected_tags.append(f"check_min:{check['min']}")
        if check.get('max'):
            selected_tags.append(f"check_max:{check['max']}")
        if check.get('currency'):
            selected_tags.append(f"ccy:{check['currency']}")
        
        ownership = investor_payload.get('target_ownership') or {}
        if ownership.get('min_pct'):
            selected_tags.append(f"own_min:{ownership['min_pct']}")
        if ownership.get('max_pct'):
            selected_tags.append(f"own_max:{ownership['max_pct']}")
        
        val_caps = investor_payload.get('valuation_caps') or {}
        for key, value in val_caps.items():
            selected_tags.append(f"valcap_{key}:{value}")
        
        traction = investor_payload.get('traction') or {}
        for key, value in traction.items():
            selected_tags.append(f"tr_{key}:{value}")
        
        attrs['selected_categories'] = self._dedup([c for c in selected_categories if c])
        attrs['selected_fields'] = self._dedup([f for f in selected_fields if f])
        attrs['selected_tags'] = self._dedup([t for t in selected_tags if t])
        attrs['preferred_startup_stages'] = self._dedup([s for s in preferred_startup_stages if s])
        attrs['preferred_engagement_types'] = self._dedup([e for e in preferred_engagement_types if e])
        
        return attrs
    
    def validate(self, attrs):
        attrs = dict(attrs)
        investor_payload = attrs.pop('investor_preferences', None) or attrs.get('investor_profile')
        if investor_payload:
            sanitized = self._sanitize_investor_payload(investor_payload)
            attrs['investor_profile'] = sanitized
            attrs = self._merge_investor_preferences(attrs, sanitized)
        return attrs
    
    def update(self, instance, validated_data):
        return super().update(instance, validated_data)


class UserInteractionSerializer(serializers.ModelSerializer):
    """Serializer for creating user interactions"""
    
    class Meta:
        model = UserInteraction
        fields = ('id', 'startup', 'interaction_type', 'position', 'metadata', 'created_at')
        read_only_fields = ('id', 'created_at')


class StartupInteractionStatusSerializer(serializers.Serializer):
    """Serializer for startup interaction status response"""
    has_like = serializers.BooleanField()
    has_dislike = serializers.BooleanField()
    has_favorite = serializers.BooleanField()
    has_interest = serializers.BooleanField()
    has_application = serializers.BooleanField()
