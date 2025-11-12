from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    def ready(self):
        """Called when Django starts"""
        # Import signals
        import api.signals  # noqa
        
        # Start scheduler only if not in a management command or migration
        import sys
        if 'runserver' in sys.argv or 'uwsgi' in sys.argv or 'gunicorn' in sys.argv:
            try:
                from api.scheduler import start_scheduler
                start_scheduler()
            except Exception as e:
                logger.warning(f"Could not start scheduler: {e}")