#!/usr/bin/env python3
"""
Quick test for notification POST endpoint
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

# Get test user
test_user, _ = User.objects.get_or_create(
    email='notif_test@example.com',
    defaults={
        'username': 'notif_test',
        'role': 'entrepreneur',
        'email_verified': True,
        'is_active': True,
        'password': '$2b$12$kGXhyFtcgHvTnJyO9UaV4OQ5xYq4zLfMpJjGVnJFLKQqgVyL8LS7u'
    }
)

# Generate token
tokens = create_token_pair(test_user)
headers = {'Authorization': f'Bearer {tokens["access_token"]}'}

print("🔔 Testing Notification POST")
print("=" * 30)

# Test notification creation
notification_data = {
    "user_id": str(test_user.id),
    "type": "pitch",
    "title": "Test Notification",
    "message": "This is a test notification"
}

try:
    response = requests.post(f"{BASE_URL}/api/notifications", json=notification_data, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 201:
        print("✅ Notification creation working!")
    else:
        print(f"❌ Failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")