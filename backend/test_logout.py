#!/usr/bin/env python3
"""
Quick logout test to verify the fix
"""
import os
import django
import requests

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup_platform.settings')
django.setup()

from django.contrib.auth import get_user_model
from api.authentication import create_token_pair

User = get_user_model()

BASE_URL = "http://localhost:8000"

def test_logout():
    """Test logout functionality"""
    
    # Get or create a test user
    test_user, created = User.objects.get_or_create(
        email='logout_test@example.com',
        defaults={
            'username': 'logoutuser',
            'role': 'entrepreneur',
            'email_verified': True,
            'is_active': True,
            'password': '$2b$12$kGXhyFtcgHvTnJyO9UaV4OQ5xYq4zLfMpJjGVnJFLKQqgVyL8LS7u'
        }
    )
    
    # Generate tokens
    tokens = create_token_pair(test_user)
    access_token = tokens['access_token']
    
    print("🚪 Testing Logout Endpoint")
    print("=" * 30)
    
    headers = {'Authorization': f'Bearer {access_token}'}
    
    # Test logout
    try:
        response = requests.post(f"{BASE_URL}/auth/logout", headers=headers)
        print(f"Logout Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Logout working properly!")
            
            # Test if token is invalidated
            profile_response = requests.get(f"{BASE_URL}/api/users/profile-data", headers=headers)
            if profile_response.status_code in [401, 403]:
                print("✅ Token properly invalidated after logout")
            else:
                print("⚠️ Token may still be valid after logout")
        else:
            print(f"❌ Logout failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_logout()