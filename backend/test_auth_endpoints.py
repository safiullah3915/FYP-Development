#!/usr/bin/env python3
"""
Quick test script to verify authentication and profile endpoints work properly
Run this after starting the Django development server to test the 403 fixes.
"""
import os
import django
import requests
import json
import sys

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup_platform.settings')
django.setup()

from django.contrib.auth import get_user_model
from api.messaging_models import UserProfile
from api.authentication import create_token_pair

User = get_user_model()

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test the key authentication and profile endpoints"""
    
    print("🧪 Testing Authentication and Profile Endpoints")
    print("=" * 50)
    
    # Test 1: Home endpoint (should work without authentication)
    print("\n1️⃣  Testing home endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Home endpoint working")
        else:
            print(f"   ❌ Home endpoint failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Create a test user if needed and get tokens
    print("\n2️⃣  Setting up test user and tokens...")
    try:
        # Try to get or create a verified test user
        test_user, created = User.objects.get_or_create(
            email='test@example.com',
            defaults={
                'username': 'testuser',
                'role': 'entrepreneur',
                'email_verified': True,
                'is_active': True,
                'password': '$2b$12$kGXhyFtcgHvTnJyO9UaV4OQ5xYq4zLfMpJjGVnJFLKQqgVyL8LS7u'  # bcrypt hash of 'testpass123'
            }
        )
        
        if created:
            print("   ✅ Created new test user")
        else:
            print("   ✅ Using existing test user")
        
        # Generate JWT tokens for the user
        tokens = create_token_pair(test_user)
        access_token = tokens['access_token']
        
        print(f"   ✅ Generated tokens successfully")
        print(f"   Access token: {access_token[:20]}...")
        
    except Exception as e:
        print(f"   ❌ Error setting up test user: {e}")
        return
    
    # Test 3: Profile data endpoint with authentication
    print("\n3️⃣  Testing profile data endpoint...")
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        response = requests.get(f"{BASE_URL}/api/users/profile-data", headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Profile data endpoint working")
            data = response.json()
            print(f"   User profile loaded: {data.get('profile', {}).get('role', 'No role found')}")
        else:
            print(f"   ❌ Profile data endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Profile data PATCH endpoint
    print("\n4️⃣  Testing profile data PATCH endpoint...")
    patch_data = {
        'skills': 'Python, Django, React',
        'summary': 'Updated test summary'
    }
    
    try:
        response = requests.patch(
            f"{BASE_URL}/api/users/profile-data", 
            headers=headers,
            json=patch_data
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Profile data PATCH endpoint working")
            data = response.json()
            print(f"   Updated skills: {data.get('skills', 'No skills found')}")
        else:
            print(f"   ❌ Profile data PATCH failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Test without authentication (should fail properly)
    print("\n5️⃣  Testing profile endpoint without authentication...")
    try:
        response = requests.get(f"{BASE_URL}/api/users/profile-data")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 401 or response.status_code == 403:
            print("   ✅ Properly rejected unauthenticated request")
        else:
            print(f"   ❌ Should have rejected unauthenticated request: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - If all tests show ✅, the 403 authentication issues are fixed")
    print("   - If any tests show ❌, there may still be authentication problems")
    print("   - Make sure Django development server is running on localhost:8000")
    

if __name__ == "__main__":
    test_endpoints()