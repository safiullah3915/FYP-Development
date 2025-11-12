from django.core.management.base import BaseCommand
from api.authentication import cleanup_expired_sessions


class Command(BaseCommand):
    help = 'Clean up expired user sessions'
    
    def handle(self, *args, **options):
        cleanup_expired_sessions()
        self.stdout.write(
            self.style.SUCCESS('Successfully cleaned up expired sessions')
        )
