#!/usr/bin/env python3

import requests
import json

def test_notification_api():
    """Test the notification POST endpoint"""
    
    # First, let's get a valid user token by logging in
    login_url = "http://127.0.0.1:8000/auth/login"
    login_data = {
        "email": "apitest@example.com",  # Test user with known password
        "password": "testpass123"     # Known password
    }
    
    print("🔐 Attempting login...")
    try:
        login_response = requests.post(login_url, json=login_data)
        print(f"Login status: {login_response.status_code}")
        print(f"Login response: {login_response.text}")
        
        if login_response.status_code == 200:
            login_data = login_response.json()
            access_token = login_data.get('access_token')
            print(f"✅ Login successful, got token: {access_token[:30]}...")
        else:
            print("❌ Login failed, trying with different credentials...")
            # Try with another user
            login_data = {
                "email": "dev-codeloom@example.com",
                "password": "testpass123"
            }
            login_response = requests.post(login_url, json=login_data)
            print(f"Retry login status: {login_response.status_code}")
            if login_response.status_code == 200:
                login_data = login_response.json()
                access_token = login_data.get('access_token')
                print(f"✅ Login successful with second attempt, got token: {access_token[:30]}...")
            else:
                print("❌ Could not login with any credentials")
                print("Please make sure you have a user with email verification enabled")
                return
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Login request failed: {e}")
        return
    
    # Now test the notification endpoint
    notification_url = "http://127.0.0.1:8000/api/notifications"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Test data for notification
    notification_data = {
        'user_id': 'd6f147fb-aa84-480e-94b5-7b7941595307',  # Test user ID
        'type': 'pitch',
        'title': 'New Business Pitch',
        'message': 'A new startup has been pitched to you',
        'data': {
            'startup_id': 'test-startup-123',
            'pitch_type': 'marketplace'
        }
    }
    
    print(f"\n🚀 Testing notification POST to {notification_url}")
    print(f"📝 Data: {json.dumps(notification_data, indent=2)}")
    print(f"🎫 Headers: {headers}")
    
    try:
        response = requests.post(notification_url, json=notification_data, headers=headers)
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        print(f"📄 Response Body: {response.text}")
        
        if response.status_code == 201:
            print("✅ SUCCESS: Notification created successfully!")
        elif response.status_code == 403:
            print("❌ 403 Forbidden - Authentication issue")
        elif response.status_code == 500:
            print("❌ 500 Internal Server Error - Server issue")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_notification_api()