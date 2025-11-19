from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver
from .models import Application, Favorite, Interest, User, Startup, StartupTag
from .messaging_models import UserProfile
from .recommendation_models import UserInteraction, UserOnboardingPreferences
import logging

logger = logging.getLogger(__name__)


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


@receiver(post_save, sender=UserInteraction)
def update_trending_metrics_realtime(sender, instance, created, **kwargs):
    """Update trending metrics in real-time when UserInteraction is created"""
    if created:
        try:
            from api.services.trending_metrics_service import TrendingMetricsService
            
            # Update metrics asynchronously (or synchronously for now)
            service = TrendingMetricsService()
            service.increment_interaction(
                startup_id=str(instance.startup.id),
                interaction_type=instance.interaction_type
            )
            
            logger.info(f"✅ [Signal] Updated trending metrics for startup {instance.startup.id} - {instance.interaction_type}")
        except Exception as e:
            # Don't let metrics update failure break the interaction creation
            logger.error(f"⚠️ [Signal] Failed to update trending metrics: {str(e)}", exc_info=True)


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
def generate_or_mark_user_embedding(sender, instance, created, **kwargs):
    """Generate embedding synchronously for new users, mark for update on changes"""
    # Skip if this is an embedding update itself (avoid infinite loop)
    update_fields = kwargs.get('update_fields', None)
    if update_fields and 'profile_embedding' in update_fields:
        return
    
    if created:
        # New user - generate embedding synchronously
        try:
            from api.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            success = embedding_service.generate_user_embedding(instance)
            if success:
                logger.info(f"Successfully generated embedding for new user {instance.id}")
            else:
                logger.warning(f"Failed to generate embedding for new user {instance.id}")
        except Exception as e:
            logger.error(f"Error generating embedding for new user {instance.id}: {e}", exc_info=True)
    elif instance.pk:
        # Existing user - check if role was updated
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


# Startup embedding signals
# Store old startup field values before save to detect changes
_startup_old_values = {}


@receiver(pre_save, sender=Startup)
def capture_startup_fields_before_save(sender, instance, **kwargs):
    """Capture old startup field values before save to detect changes"""
    if instance.pk:
        try:
            old_instance = Startup.objects.get(pk=instance.pk)
            _startup_old_values[instance.pk] = {
                'title': old_instance.title,
                'description': old_instance.description,
                'field': old_instance.field,
                'category': old_instance.category,
                'type': old_instance.type,
                'stages': old_instance.stages,
            }
        except Startup.DoesNotExist:
            pass


@receiver(post_save, sender=Startup)
def generate_or_mark_startup_embedding(sender, instance, created, **kwargs):
    """Generate embedding synchronously for new startups, mark for update on changes"""
    # Skip if this is an embedding update itself (avoid infinite loop)
    update_fields = kwargs.get('update_fields', None)
    if update_fields and 'profile_embedding' in update_fields:
        return
    
    if created:
        # New startup - generate embedding synchronously
        try:
            from api.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            success = embedding_service.generate_startup_embedding(instance)
            if success:
                logger.info(f"Successfully generated embedding for new startup {instance.id}")
            else:
                logger.warning(f"Failed to generate embedding for new startup {instance.id}")
        except Exception as e:
            logger.error(f"Error generating embedding for new startup {instance.id}: {e}", exc_info=True)
    elif instance.pk:
        # Existing startup - check if relevant fields changed
        old_values = _startup_old_values.pop(instance.pk, None)
        if old_values:
            # Check if any relevant field changed
            fields_changed = (
                old_values['title'] != instance.title or
                old_values['description'] != instance.description or
                old_values['field'] != instance.field or
                old_values['category'] != instance.category or
                old_values['type'] != instance.type or
                old_values['stages'] != instance.stages
            )
            if fields_changed:
                # Mark for asynchronous update
                Startup.objects.filter(pk=instance.pk).update(embedding_needs_update=True)
                logger.info(f"Marked startup {instance.id} for embedding update due to field changes")


@receiver(post_save, sender=StartupTag)
def mark_startup_embedding_for_update_on_tag_add(sender, instance, created, **kwargs):
    """Mark startup for embedding update when a tag is added"""
    if created:
        try:
            # Mark parent startup for embedding update
            Startup.objects.filter(pk=instance.startup_id).update(embedding_needs_update=True)
            logger.info(f"Marked startup {instance.startup_id} for embedding update due to new tag")
        except Exception as e:
            logger.error(f"Error marking startup {instance.startup_id} for update: {e}")


@receiver(post_delete, sender=StartupTag)
def mark_startup_embedding_for_update_on_tag_delete(sender, instance, **kwargs):
    """Mark startup for embedding update when a tag is removed"""
    try:
        # Mark parent startup for embedding update
        Startup.objects.filter(pk=instance.startup_id).update(embedding_needs_update=True)
        logger.info(f"Marked startup {instance.startup_id} for embedding update due to tag removal")
    except Exception as e:
        logger.error(f"Error marking startup {instance.startup_id} for update: {e}")

