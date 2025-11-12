from django.utils.deprecation import MiddlewareMixin
from .authentication import JWTAuthentication


class JWTAuthenticationMiddleware(MiddlewareMixin):
    """Middleware to handle JWT authentication for all requests"""
    
    def process_request(self, request):
        """Process request to add user authentication"""
        auth = JWTAuthentication()
        try:
            user_auth_tuple = auth.authenticate(request)
            if user_auth_tuple is not None:
                request.user, request.auth = user_auth_tuple
        except Exception:
            # If authentication fails, let the view handle it
            pass
        
        return None
