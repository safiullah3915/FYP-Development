"""
Custom authentication backend for bcrypt password verification
"""
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
import bcrypt

User = get_user_model()


class BcryptAuthBackend(ModelBackend):
    """
    Custom authentication backend that handles bcrypt password hashing
    """
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        """
        Authenticate user with email and bcrypt password
        """
        print(f"\n{'='*80}")
        print(f"🔐 BCRYPT AUTH BACKEND - AUTHENTICATION ATTEMPT")
        print(f"{'='*80}")
        print(f"📧 Username/Email provided: {username}")
        print(f"🔑 Password provided: {'Yes' if password else 'No'}")
        print(f"📋 Additional kwargs: {kwargs}")
        
        # Get email from username parameter (Django admin passes it as username)
        email = username or kwargs.get('email')
        
        if not email or not password:
            print(f"❌ Missing email or password")
            print(f"{'='*80}\n")
            return None
        
        try:
            # Try to get user by email (since USERNAME_FIELD = 'email')
            print(f"🔍 Looking for user with email: {email}")
            user = User.objects.get(email=email)
            print(f"✅ User found: {user.username} (ID: {user.id})")
            print(f"📊 User details:")
            print(f"  - is_active: {user.is_active}")
            print(f"  - is_staff: {user.is_staff}")
            print(f"  - is_superuser: {user.is_superuser}")
            print(f"  - email_verified: {user.email_verified}")
            print(f"  - role: {user.role}")
        except User.DoesNotExist:
            print(f"❌ No user found with email: {email}")
            # Try by username as fallback
            try:
                print(f"🔍 Trying to find user by username: {email}")
                user = User.objects.get(username=email)
                print(f"✅ User found by username: {user.username} (ID: {user.id})")
            except User.DoesNotExist:
                print(f"❌ No user found with username: {email}")
                print(f"{'='*80}\n")
                return None
        
        # Verify password with bcrypt
        try:
            print(f"🔒 Verifying password with bcrypt...")
            print(f"🔒 Stored password (first 20 chars): {user.password[:20]}...")
            
            # Check if password is bcrypt hashed
            if user.password.startswith('$2b$') or user.password.startswith('$2a$'):
                print(f"✅ Password is bcrypt hashed")
                password_match = bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8'))
                print(f"🔑 Password match result: {password_match}")
                
                if password_match:
                    print(f"✅ Authentication successful for user: {user.username}")
                    print(f"{'='*80}\n")
                    return user
                else:
                    print(f"❌ Password verification failed - incorrect password")
                    print(f"{'='*80}\n")
                    return None
            else:
                print(f"⚠️ Password is NOT bcrypt hashed (might be Django default)")
                print(f"🔄 Falling back to Django's default password checker...")
                # Fall back to Django's default password check
                if user.check_password(password):
                    print(f"✅ Authentication successful with Django default hasher")
                    print(f"{'='*80}\n")
                    return user
                else:
                    print(f"❌ Password verification failed with Django hasher too")
                    print(f"{'='*80}\n")
                    return None
        except Exception as e:
            print(f"❌ Password verification error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            return None
    
    def get_user(self, user_id):
        """
        Get user by ID
        """
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None


