import jwt
import hashlib
import random
from datetime import datetime, timedelta
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.mail import send_mail
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import UserSession

User = get_user_model()


class JWTAuthentication(BaseAuthentication):
	"""Custom JWT authentication class"""
	
	def authenticate(self, request):
		token = self.get_token_from_request(request)
		if not token:
			print("❌ No token provided, returning None (unauthenticated)")
			return None
		
		print(f"🔐 Attempting to authenticate token: {token[:20]}...")
		
		try:
			payload = jwt.decode(
				token, 
				settings.JWT_SECRET_KEY, 
				algorithms=[settings.JWT_ALGORITHM]
			)
			
			# Check if it's an access token
			if payload.get('token_type') != 'access':
				raise AuthenticationFailed('Invalid token type')
			
			user_id = payload.get('user_id')
			if not user_id:
				raise AuthenticationFailed('Invalid token payload')
			
			user = User.objects.get(id=user_id, is_active=True)
			
			# Check if session exists and is not expired
			token_hash = hashlib.sha256(token.encode()).hexdigest()
			session = UserSession.objects.filter(
				user=user,
				token_hash=token_hash,
				token_type='access',
				expires_at__gt=timezone.now()
			).first()
			
			if not session:
				raise AuthenticationFailed('Session expired or invalid')
			
			return (user, token)
			
		except jwt.ExpiredSignatureError:
			raise AuthenticationFailed('Token has expired')
		except jwt.InvalidTokenError:
			raise AuthenticationFailed('Invalid token')
		except User.DoesNotExist:
			raise AuthenticationFailed('User not found')
	
	def get_token_from_request(self, request):
		"""Extract JWT token from cookie or Authorization header"""
		print(f"🔍 Looking for JWT token in request to {request.path}")
		print(f"Available cookies: {list(request.COOKIES.keys())}")
		
		# First try to get from cookie
		token = request.COOKIES.get(settings.JWT_COOKIE_NAME)
		if token:
			print(f"✅ Found token in cookie '{settings.JWT_COOKIE_NAME}': {token[:20]}...")
			return token
		
		# Fallback to Authorization header
		auth_header = request.META.get('HTTP_AUTHORIZATION')
		if auth_header and auth_header.startswith('Bearer '):
			token = auth_header.split(' ')[1]
			print(f"✅ Found token in Authorization header: {token[:20]}...")
			return token
		
		print("❌ No JWT token found in cookies or Authorization header")
		return None


def create_jwt_token(user):
	"""Create JWT access token for user"""
	now = timezone.now()
	expiration = now + timedelta(seconds=settings.JWT_ACCESS_TOKEN_DELTA)
	
	payload = {
		'user_id': str(user.id),
		'username': user.username,
		'email': user.email,
		'token_type': 'access',
		'exp': int(expiration.timestamp()),
		'iat': int(now.timestamp())
	}
	
	token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
	
	# Create session record for access token
	token_hash = hashlib.sha256(token.encode()).hexdigest()
	
	UserSession.objects.create(
		user=user,
		token_hash=token_hash,
		token_type='access',
		expires_at=expiration
	)
	
	return token


def create_refresh_token(user):
	"""Create JWT refresh token for user"""
	now = timezone.now()
	expiration = now + timedelta(seconds=settings.JWT_REFRESH_TOKEN_DELTA)
	
	payload = {
		'user_id': str(user.id),
		'token_type': 'refresh',
		'exp': int(expiration.timestamp()),
		'iat': int(now.timestamp())
	}
	
	refresh_token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
	
	# Create session record for refresh token
	token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
	
	UserSession.objects.create(
		user=user,
		token_hash=token_hash,
		expires_at=expiration,
		token_type='refresh'
	)
	
	return refresh_token


def create_token_pair(user):
	"""Create both access and refresh tokens for user"""
	access_token = create_jwt_token(user)
	refresh_token = create_refresh_token(user)
	return {
		'access_token': access_token,
		'refresh_token': refresh_token,
		'token_type': 'Bearer',
		'expires_in': settings.JWT_ACCESS_TOKEN_DELTA
	}


def verify_refresh_token(refresh_token):
	"""Verify refresh token and return user if valid"""
	try:
		payload = jwt.decode(
			refresh_token, 
			settings.JWT_SECRET_KEY, 
			algorithms=[settings.JWT_ALGORITHM]
		)
		
		# Check if it's a refresh token
		if payload.get('token_type') != 'refresh':
			raise AuthenticationFailed('Invalid token type')
		
		user_id = payload.get('user_id')
		if not user_id:
			raise AuthenticationFailed('Invalid token payload')
		
		user = User.objects.get(id=user_id, is_active=True)
		print(f"✅ JWT: Found user - {user.username} (ID: {user.id})")
		print(f"- Role: {user.role}")
		print(f"- Email Verified: {user.email_verified}")
		
		# Check if refresh token session exists and is not expired
		token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
		session = UserSession.objects.filter(
			user=user,
			token_hash=token_hash,
			token_type='refresh',
			expires_at__gt=timezone.now()
		).first()
		
		if not session:
			print(f"❌ JWT: No valid refresh session found for token")
			raise AuthenticationFailed('Session expired or invalid')
		
		print(f"✅ JWT: Refresh token validation successful for {user.username}")
		return user
		
	except jwt.ExpiredSignatureError:
		raise AuthenticationFailed('Refresh token has expired')
	except jwt.InvalidTokenError:
		raise AuthenticationFailed('Invalid refresh token')
	except User.DoesNotExist:
		raise AuthenticationFailed('User not found')


def refresh_access_token(refresh_token):
	"""Generate new access token using refresh token"""
	user = verify_refresh_token(refresh_token)
	
	# Invalidate old access tokens for this user
	UserSession.objects.filter(
		user=user,
		token_type='access'
	).delete()
	
	# Create new access token
	access_token = create_jwt_token(user)
	
	return {
		'access_token': access_token,
		'token_type': 'Bearer',
		'expires_in': settings.JWT_ACCESS_TOKEN_DELTA
	}


def invalidate_user_sessions(user):
	"""Invalidate all user sessions"""
	UserSession.objects.filter(user=user).delete()


def cleanup_expired_sessions():
	"""Clean up expired sessions"""
	UserSession.objects.filter(expires_at__lt=timezone.now()).delete()


def create_email_verification_token(user):
	"""Create a short-lived email verification token"""
	payload = {
		'user_id': str(user.id),
		'purpose': 'verify_email',
		'exp': timezone.now() + timedelta(hours=24),
		'iat': timezone.now()
	}
	return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_email_verification_token(token):
	"""Decode and validate an email verification token"""
	payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
	if payload.get('purpose') != 'verify_email':
		raise AuthenticationFailed('Invalid verification token')
	return payload


def generate_verification_code():
	"""Generate a 6-digit verification code"""
	return str(random.randint(100000, 999999))


def send_verification_code(user):
	"""Generate and send verification code to user's email"""
	from .models import EmailVerificationCode
	
	# Clean up old unused codes for this user
	EmailVerificationCode.objects.filter(
		user=user,
		is_used=False
	).delete()
	
	# Generate new code
	code = generate_verification_code()
	expires_at = timezone.now() + timedelta(minutes=15)  # Code expires in 15 minutes
	
	# Save code to database
	verification_code = EmailVerificationCode.objects.create(
		user=user,
		code=code,
		expires_at=expires_at
	)
	
	# Send email (you can customize this based on your email settings)
	try:
		subject = 'Your Email Verification Code - Startup Platform'
		message = f'''
Hi {user.username},

Welcome to Startup Platform! To complete your account setup, please verify your email address.

Your verification code is: {code}

This code will expire in 15 minutes for security reasons.

If you didn't create an account with us, please ignore this email.

Best regards,
The Startup Platform Team

---
This is an automated email. Please do not reply to this message.
		'''
		
		send_mail(
			subject,
			message,
			settings.DEFAULT_FROM_EMAIL,
			[user.email],
			fail_silently=False,
		)
		print(f"✅ Verification email sent successfully to {user.email}")
		return True
	except Exception as e:
		print(f"❌ Failed to send email to {user.email}: {e}")
		if "authentication failed" in str(e).lower():
			print("💡 Tip: Check Gmail App Password configuration")
		print(f"Email settings - HOST: smtp.gmail.com, PORT: 587, TLS: True")
		return False


def verify_email_code(user, code):
	"""Verify the email verification code"""
	from .models import EmailVerificationCode
	
	try:
		verification_code = EmailVerificationCode.objects.get(
			user=user,
			code=code,
			is_used=False,
			expires_at__gt=timezone.now()
		)
		
		# Mark code as used
		verification_code.is_used = True
		verification_code.save()
		
		# Mark user email as verified
		user.email_verified = True
		user.save(update_fields=['email_verified'])
		
		return True
		
	except EmailVerificationCode.DoesNotExist:
		return False
