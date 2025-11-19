#!/usr/bin/env python
"""Test script to check interaction count for a specific user"""
import sys
import os
import django

# Setup Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup_platform.settings')
django.setup()

from api.recommendation_models import UserInteraction
from django.contrib.auth import get_user_model

User = get_user_model()

# Test user ID from logs
user_id = '3ef204fd-d6b3-4a7e-a376-27758efc383d'

print(f"Testing user ID: {user_id}")
print("=" * 80)

# Check Django ORM
user = User.objects.filter(id=user_id).first()
if user:
    print(f"Django - User found: {user.username} ({user.email})")
    django_count = UserInteraction.objects.filter(user=user).count()
    print(f"Django - Interaction count: {django_count}")
    
    # Show some interactions
    interactions = UserInteraction.objects.filter(user=user)[:5]
    print(f"\nDjango - Sample interactions:")
    for i, interaction in enumerate(interactions, 1):
        print(f"  {i}. {interaction.interaction_type} - Startup: {interaction.startup_id}")
else:
    print(f"Django - User NOT found!")

print("\n" + "=" * 80)

# Check Flask SQLAlchemy
try:
    sys.path.insert(0, '../recommendation_service')
    from database.connection import SessionLocal
    from database.models import UserInteraction as FlaskUserInteraction, User as FlaskUser
    
    db = SessionLocal()
    try:
        flask_user = db.query(FlaskUser).filter(FlaskUser.id == user_id).first()
        if flask_user:
            print(f"Flask - User found: {flask_user.username} ({flask_user.email})")
            flask_count = db.query(FlaskUserInteraction).filter(
                FlaskUserInteraction.user_id == user_id
            ).count()
            print(f"Flask - Interaction count: {flask_count}")
            
            # Show some interactions
            interactions = db.query(FlaskUserInteraction).filter(
                FlaskUserInteraction.user_id == user_id
            ).limit(5).all()
            print(f"\nFlask - Sample interactions:")
            for i, interaction in enumerate(interactions, 1):
                print(f"  {i}. {interaction.interaction_type} - Startup: {interaction.startup_id}")
        else:
            print(f"Flask - User NOT found!")
            # Check if user exists with different format
            all_users = db.query(FlaskUser).limit(5).all()
            print(f"\nFlask - Sample user IDs:")
            for u in all_users:
                print(f"  - {u.id} ({u.username})")
    finally:
        db.close()
except Exception as e:
    print(f"Flask check failed: {e}")
    import traceback
    traceback.print_exc()

