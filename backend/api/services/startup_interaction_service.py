"""
Startup Interaction Service
Handles startup → user interactions for reverse recommendations

Integration Points:
- When entrepreneur views developer/investor profile from recommendations:
  Call create_startup_interaction() with interaction_type='view'
  
- When entrepreneur clicks on developer/investor profile from recommendations:
  Call create_startup_interaction() with interaction_type='click'
  
- When entrepreneur contacts developer/investor from recommendations:
  Call create_startup_interaction() with interaction_type='contact'
  
- When developer applies to startup position (reverse of apply):
  Call create_startup_interaction() with interaction_type='apply_received'

Example usage:
  from api.services.startup_interaction_service import StartupInteractionService
  from api.models import Startup, User
  
  startup = Startup.objects.get(id=startup_id)
  target_user = User.objects.get(id=user_id)
  session = RecommendationSession.objects.get(id=session_id)
  
  StartupInteractionService.create_startup_interaction(
      startup=startup,
      target_user=target_user,
      interaction_type='view',
      recommendation_session=session,
      recommendation_source='recommendation',
      recommendation_rank=1,
      recommendation_score=0.95
  )
"""
from typing import Optional, Dict, Any
from api.models import Startup
from api.recommendation_models import StartupInteraction, RecommendationSession


class StartupInteractionService:
    """Service for creating and managing startup interactions"""
    
    @staticmethod
    def create_startup_interaction(
        startup: Startup,
        target_user,
        interaction_type: str,
        recommendation_session: Optional[RecommendationSession] = None,
        metadata: Optional[Dict[str, Any]] = None,
        recommendation_source: Optional[str] = None,
        recommendation_rank: Optional[int] = None,
        recommendation_score: Optional[float] = None,
        recommendation_method: Optional[str] = None
    ) -> StartupInteraction:
        """
        Create a startup interaction record
        
        Args:
            startup: Startup instance
            target_user: User instance (developer or investor)
            interaction_type: Type of interaction ('view', 'click', 'contact', 'apply_received')
            recommendation_session: Optional RecommendationSession instance
            metadata: Optional metadata dict
            recommendation_source: Optional source ('organic' or 'recommendation')
            recommendation_rank: Optional rank in recommendation list
            recommendation_score: Optional recommendation score
            recommendation_method: Optional recommendation method used
            
        Returns:
            StartupInteraction instance
        """
        # Build metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add recommendation context to metadata if available
        if recommendation_session:
            metadata['recommendation_session_id'] = str(recommendation_session.id)
            if recommendation_source is None:
                recommendation_source = 'recommendation'
        
        # Create or update interaction (unique constraint on startup, target_user, interaction_type)
        interaction, created = StartupInteraction.objects.update_or_create(
            startup=startup,
            target_user=target_user,
            interaction_type=interaction_type,
            defaults={
                'recommendation_session': recommendation_session,
                'recommendation_source': recommendation_source or 'organic',
                'recommendation_rank': recommendation_rank,
                'recommendation_score': recommendation_score,
                'recommendation_method': recommendation_method,
                'metadata': metadata
            }
        )
        
        return interaction
    
    @staticmethod
    def build_interaction_metadata(
        startup: Startup,
        target_user,
        recommendation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build metadata dict with startup context
        
        Args:
            startup: Startup instance
            target_user: User instance
            recommendation_context: Optional context dict with session info
            
        Returns:
            Metadata dict
        """
        metadata = {
            'startup_id': str(startup.id),
            'startup_title': startup.title,
            'target_user_id': str(target_user.id),
            'target_user_username': target_user.username,
        }
        
        if recommendation_context:
            metadata.update({
                'session_id': recommendation_context.get('session_id'),
                'rank': recommendation_context.get('rank'),
                'score': recommendation_context.get('score'),
                'method': recommendation_context.get('method'),
                'source': 'recommendation'
            })
        else:
            metadata['source'] = 'organic'
        
        return metadata
    
    @staticmethod
    def get_weight_for_interaction_type(interaction_type: str) -> float:
        """
        Get weight for interaction type
        
        Args:
            interaction_type: Type of interaction
            
        Returns:
            Weight value
        """
        weight_mapping = {
            'view': 0.5,
            'click': 1.0,
            'contact': 2.0,
            'apply_received': 3.0,
        }
        return weight_mapping.get(interaction_type, 1.0)
    
    @staticmethod
    def validate_interaction_type(interaction_type: str) -> bool:
        """
        Validate interaction type
        
        Args:
            interaction_type: Type of interaction to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_types = ['view', 'click', 'contact', 'apply_received']
        return interaction_type in valid_types

