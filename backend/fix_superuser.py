#!/usr/bin/env python
"""
Script to fix superuser permissions for Django admin access
"""
import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'startup_platform.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

def fix_superuser():
    """Fix superuser to have proper admin access"""
    email = input("Enter your superuser email: ").strip()
    
    try:
        user = User.objects.get(email=email)
        print(f"\n✅ Found user: {user.username} ({user.email})")
        print(f"Current flags:")
        print(f"  - is_staff: {user.is_staff}")
        print(f"  - is_superuser: {user.is_superuser}")
        print(f"  - is_active: {user.is_active}")
        
        # Update user to have admin access
        user.is_staff = True
        user.is_superuser = True
        user.is_active = True
        user.save()
        
        print(f"\n✅ User updated successfully!")
        print(f"New flags:")
        print(f"  - is_staff: {user.is_staff}")
        print(f"  - is_superuser: {user.is_superuser}")
        print(f"  - is_active: {user.is_active}")
        print(f"\n🎉 You can now access Django admin at http://localhost:8000/admin")
        
    except User.DoesNotExist:
        print(f"\n❌ No user found with email: {email}")
        print("Available users:")
        for u in User.objects.all()[:5]:
            print(f"  - {u.username} ({u.email})")

if __name__ == '__main__':
    fix_superuser()




