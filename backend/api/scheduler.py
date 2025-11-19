"""
APScheduler configuration for periodic background tasks
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django_apscheduler.models import DjangoJobExecution
from django.conf import settings
from django.utils import timezone
from django.db import models

logger = logging.getLogger(__name__)


def update_pending_user_embeddings():
    """
    Periodic task to update embeddings for users whose profile data has changed.
    This function is called by APScheduler at scheduled intervals.
    """
    from api.models import User
    from api.services.embedding_service import EmbeddingService
    
    logger.info("Starting periodic user embedding update task...")
    
    # Query users that need embedding updates
    users_query = User.objects.filter(is_active=True).filter(
        models.Q(embedding_needs_update=True) | 
        models.Q(embedding_updated_at__isnull=True) |
        models.Q(profile_embedding__isnull=True) |
        models.Q(profile_embedding='')
    )
    
    users = list(users_query)
    total = len(users)
    
    if total == 0:
        logger.info("No users need embedding updates.")
        return
    
    logger.info(f"Found {total} users needing embedding updates. Processing in batches...")
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Process in batches
    batch_size = 50
    results = embedding_service.generate_embeddings_batch(users, batch_size=batch_size)
    
    # Log results
    logger.info(
        f"User embedding update task completed: {results['success']} succeeded, "
        f"{results['failed']} failed out of {total} users"
    )
    
    if results['errors']:
        logger.warning(f"Encountered {len(results['errors'])} errors during user embedding generation")
        for error in results['errors'][:5]:  # Log first 5 errors
            logger.error(f"Embedding error: {error}")


def update_pending_startup_embeddings():
    """
    Periodic task to update embeddings for startups whose data has changed.
    This function is called by APScheduler at scheduled intervals.
    """
    from api.models import Startup
    from api.services.embedding_service import EmbeddingService
    
    logger.info("Starting periodic startup embedding update task...")
    
    # Query startups that need embedding updates (only active startups)
    startups_query = Startup.objects.filter(status='active').filter(
        models.Q(embedding_needs_update=True) |
        models.Q(embedding_updated_at__isnull=True) |
        models.Q(profile_embedding__isnull=True) |
        models.Q(profile_embedding='')
    )
    
    startups = list(startups_query)
    total = len(startups)
    
    if total == 0:
        logger.info("No startups need embedding updates.")
        return
    
    logger.info(f"Found {total} startups needing embedding updates. Processing in batches...")
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Process in batches
    batch_size = 50
    results = embedding_service.generate_startup_embeddings_batch(startups, batch_size=batch_size)
    
    # Log results
    logger.info(
        f"Startup embedding update task completed: {results['success']} succeeded, "
        f"{results['failed']} failed out of {total} startups"
    )
    
    if results['errors']:
        logger.warning(f"Encountered {len(results['errors'])} errors during startup embedding generation")
        for error in results['errors'][:5]:  # Log first 5 errors
            logger.error(f"Embedding error: {error}")


def compute_trending_metrics():
    """
    Periodic task to compute trending metrics for all active startups.
    This function is called by APScheduler at scheduled intervals.
    Calculates popularity_score, trending_score, velocity_score, and various view/application counts.
    """
    from django.core.management import call_command
    
    logger.info("Starting periodic trending metrics computation task...")
    
    try:
        # Call the management command to compute trending metrics
        call_command('compute_trending_metrics', verbosity=0)
        logger.info("Trending metrics computation task completed successfully")
    except Exception as e:
        logger.error(f"Error computing trending metrics: {e}", exc_info=True)


# Global scheduler instance to prevent multiple instances
_scheduler = None


def start_scheduler():
    """
    Start the APScheduler background scheduler.
    This should be called when Django starts (e.g., in apps.py ready() method).
    """
    global _scheduler
    
    # Check if scheduler already exists and is running
    if _scheduler is not None:
        # Check scheduler state: 0 = STATE_STOPPED, 1 = STATE_RUNNING, 2 = STATE_PAUSED
        if _scheduler.state == 1:  # Already running
            logger.info("APScheduler is already running, skipping start")
            return
        elif _scheduler.state == 2:  # Paused, resume it
            logger.info("APScheduler was paused, resuming...")
            _scheduler.resume()
            return
    
    # Create new scheduler instance
    _scheduler = BackgroundScheduler()
    _scheduler.add_jobstore(DjangoJobStore(), "default")
    
    # Schedule the user embedding update task to run every 6 hours
    _scheduler.add_job(
        update_pending_user_embeddings,
        trigger=CronTrigger(hour="*/6"),  # Every 6 hours
        id="update_user_embeddings",
        name="Update user embeddings for changed profiles",
        replace_existing=True,
    )
    
    # Schedule the startup embedding update task to run every 6 hours
    _scheduler.add_job(
        update_pending_startup_embeddings,
        trigger=CronTrigger(hour="*/6"),  # Every 6 hours
        id="update_startup_embeddings",
        name="Update startup embeddings for changed data",
        replace_existing=True,
    )
    
    # Schedule the trending metrics computation task to run every 1 minute
    # This runs periodically to ensure all metrics are recalculated with global max values
    # Real-time updates happen via signals when interactions occur
    _scheduler.add_job(
        compute_trending_metrics,
        trigger=IntervalTrigger(minutes=1),  # Every 1 minute
        id="compute_trending_metrics",
        name="Compute trending metrics for all active startups",
        replace_existing=True,
    )
    
    # Register events to clean up old job executions
    register_events(_scheduler)
    
    try:
        # Check state before starting (0 = STATE_STOPPED)
        if _scheduler.state == 0:
            logger.info("Starting APScheduler...")
            _scheduler.start()
            logger.info("APScheduler started successfully")
        else:
            logger.warning(f"APScheduler is in unexpected state: {_scheduler.state}")
    except Exception as e:
        logger.error(f"Error starting APScheduler: {e}", exc_info=True)
        _scheduler = None  # Reset on error
        raise

