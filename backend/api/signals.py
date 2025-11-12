from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from .models import Application, Favorite, Interest, User
from .messaging_models import UserProfile
from .recommendation_models import UserInteraction, UserOnboardingPreferences


@receiver(post_save, sender=Application)
def create_apply_interaction(sender, instance, created, **kwargs):
    """Create UserInteraction when Application is created"""
    if created:
        UserInteraction.objects.create(
            user=instance.applicant,
            startup=instance.startup,
            position=instance.position,
            interaction_type='apply',
            weight=3.0
        )


@receiver(post_save, sender=Favorite)
def create_favorite_interaction(sender, instance, created, **kwargs):
    """Create UserInteraction when Favorite is created"""
    if created:
        UserInteraction.objects.create(
            user=instance.user,
            startup=instance.startup,
            interaction_type='favorite',
            weight=2.5
        )


@receiver(post_save, sender=Interest)
def create_interest_interaction(sender, instance, created, **kwargs):
    """Create UserInteraction when Interest is created"""
    if created:
        UserInteraction.objects.create(
            user=instance.user,
            startup=instance.startup,
            interaction_type='interest',
            weight=3.5
        )


# Embedding update signals
# Store old role value before save to detect changes
_user_old_roles = {}


@receiver(pre_save, sender=User)
def capture_user_role_before_save(sender, instance, **kwargs):
    """Capture the old role value before save to detect changes"""
    if instance.pk:
        try:
            old_instance = User.objects.get(pk=instance.pk)
            _user_old_roles[instance.pk] = old_instance.role
        except User.DoesNotExist:
            pass


@receiver(post_save, sender=User)
def mark_user_embedding_for_update_on_role_change(sender, instance, created, **kwargs):
    """Mark user embedding for update when role changes"""
    if not created and instance.pk:
        # Check if role was updated
        update_fields = kwargs.get('update_fields', None)
        if update_fields is None or 'role' in update_fields:
            # Check if role actually changed by comparing with captured old value
            old_role = _user_old_roles.pop(instance.pk, None)
            if old_role is not None and old_role != instance.role:
                User.objects.filter(pk=instance.pk).update(embedding_needs_update=True)


@receiver(post_save, sender=UserProfile)
def mark_user_embedding_for_update_on_profile_change(sender, instance, created, **kwargs):
    """Mark user embedding for update when profile fields change"""
    # Fields that affect embedding generation
    embedding_relevant_fields = ['bio', 'skills', 'experience', 'location']
    
    update_fields = kwargs.get('update_fields', None)
    
    # If update_fields is specified, only mark if relevant fields changed
    if update_fields:
        if any(field in embedding_relevant_fields for field in update_fields):
            User.objects.filter(pk=instance.user_id).update(embedding_needs_update=True)
    else:
        # If update_fields is None, mark for update (safer approach)
        User.objects.filter(pk=instance.user_id).update(embedding_needs_update=True)


@receiver(post_save, sender=UserOnboardingPreferences)
def mark_user_embedding_for_update_on_preferences_change(sender, instance, created, **kwargs):
    """Mark user embedding for update when onboarding preferences change"""
    # Fields that affect embedding generation
    embedding_relevant_fields = ['selected_categories', 'selected_fields', 'preferred_skills']
    
    update_fields = kwargs.get('update_fields', None)
    
    # If update_fields is specified, only mark if relevant fields changed
    if update_fields:
        if any(field in embedding_relevant_fields for field in update_fields):
            User.objects.filter(pk=instance.user_id).update(embedding_needs_update=True)
    else:
        # If update_fields is None, mark for update (safer approach)
        User.objects.filter(pk=instance.user_id).update(embedding_needs_update=True)

