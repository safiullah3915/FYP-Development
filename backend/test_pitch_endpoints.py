#!/usr/bin/env python3
"""
Test script to verify PitchIdea related endpoints work properly
"""
import os
import django
import requests
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup_platform.settings')
django.setup()

from django.contrib.auth import get_user_model
from api.authentication import create_token_pair

User = get_user_model()

BASE_URL = "http://localhost:8000"

def test_pitch_related_endpoints():
    """Test endpoints used by PitchIdea component"""
    
    print("🎯 Testing PitchIdea Related Endpoints")
    print("=" * 50)
    
    # Create test users
    entrepreneur, _ = User.objects.get_or_create(
        email='entrepreneur@test.com',
        defaults={
            'username': 'entrepreneur_test',
            'role': 'entrepreneur',
            'email_verified': True,
            'is_active': True,
            'password': '$2b$12$kGXhyFtcgHvTnJyO9UaV4OQ5xYq4zLfMpJjGVnJFLKQqgVyL8LS7u'
        }
    )
    
    investor, _ = User.objects.get_or_create(
        email='investor@test.com',
        defaults={
            'username': 'investor_test',
            'role': 'investor',
            'email_verified': True,
            'is_active': True,
            'password': '$2b$12$kGXhyFtcgHvTnJyO9UaV4OQ5xYq4zLfMpJjGVnJFLKQqgVyL8LS7u'
        }
    )
    
    # Generate tokens for entrepreneur
    tokens = create_token_pair(entrepreneur)
    headers = {'Authorization': f'Bearer {tokens["access_token"]}'}
    
    # Test 1: Online users endpoint (used to get investors)
    print("\\n1️⃣  Testing online users endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/messages/users/online", headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Online users endpoint working")
            users = response.json()
            investors = [u for u in users if u.get('role') == 'investor']
            print(f"   Found {len(investors)} investors")
        else:
            print(f"   ❌ Online users failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Startup creation (used in pitch submission)
    print("\\n2️⃣  Testing startup creation...")
    startup_data = {
        "title": "Test Pitch Startup",
        "description": "Test startup for pitch",
        "field": "Technology",
        "type": "marketplace",
        "category": "other",
        "asking_price": "50000",
        "website_url": "https://test.com"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/startups", json=startup_data, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 201:
            print("   ✅ Startup creation working")
            startup = response.json()
            startup_id = startup.get('id')
            print(f"   Created startup ID: {startup_id}")
        else:
            print(f"   ❌ Startup creation failed: {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 3: Notification creation (used to notify investors)
    print("\\n3️⃣  Testing notification creation...")
    notification_data = {
        "user_id": str(investor.id),
        "type": "pitch",
        "title": "New Business Pitch: Test Pitch Startup",
        "message": f"{entrepreneur.username} has pitched their business idea to you.",
        "data": {
            "startup_id": startup_id,
            "entrepreneur_id": str(entrepreneur.id)
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/notifications", json=notification_data, headers=headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 201:
            print("   ✅ Notification creation working")
            notification = response.json()
            print(f"   Created notification: {notification.get('title')}")
        else:
            print(f"   ❌ Notification creation failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Get notifications (to verify they were created)
    print("\\n4️⃣  Testing notification retrieval...")
    investor_tokens = create_token_pair(investor)
    investor_headers = {'Authorization': f'Bearer {investor_tokens["access_token"]}'}
    
    try:
        response = requests.get(f"{BASE_URL}/api/notifications", headers=investor_headers)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Notification retrieval working")
            notifications = response.json()
            pitch_notifications = [n for n in notifications if n.get('type') == 'pitch']
            print(f"   Found {len(pitch_notifications)} pitch notifications")
        else:
            print(f"   ❌ Notification retrieval failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\\n" + "=" * 50)
    print("🎯 PitchIdea Endpoints Test Summary:")
    print("   If all tests show ✅, the PitchIdea functionality should work")
    print("   If any tests show ❌, there are still issues to resolve")

if __name__ == "__main__":
    test_pitch_related_endpoints()