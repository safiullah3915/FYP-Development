from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Application, Favorite, Interest
from .recommendation_models import UserInteraction


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

