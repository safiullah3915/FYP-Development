from api.recommendation_models import UserInteraction, RecommendationSession
from django.utils import timezone
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class InteractionService:
    """Service for handling user interactions with recommendation context"""
    
    @staticmethod
    def build_interaction_metadata(
        recommendation_session_id: Optional[str] = None,
        recommendation_rank: Optional[int] = None,
        user_id: Optional[str] = None,
        **additional_context
    ) -> Dict:
        """
        Build standardized metadata for interaction
        This ensures consistent structure for ETL extraction
        """
        metadata = {}
        
        if recommendation_session_id and user_id:
            # Verify session exists and is valid
            try:
                session = RecommendationSession.objects.get(
                    id=recommendation_session_id,
                    user_id=user_id
                )
                
                # Check if session is still valid
                if session.expires_at and session.expires_at < timezone.now():
                    metadata['source'] = 'organic'
                    metadata['expired_session_id'] = recommendation_session_id
                else:
                    # Valid recommendation session
                    metadata['source'] = 'recommendation'
                    metadata['recommendation_session_id'] = recommendation_session_id
                    metadata['recommendation_use_case'] = session.use_case
                    metadata['recommendation_method'] = session.recommendation_method
                    metadata['model_version'] = session.model_version or ''
                    
                    if recommendation_rank:
                        metadata['recommendation_rank'] = int(recommendation_rank)
                    
                    # Get recommendation score from session if available
                    recommendations = session.recommendations_shown
                    if recommendations and recommendation_rank:
                        rec = next((r for r in recommendations if r.get('rank') == recommendation_rank), None)
                        if rec:
                            metadata['recommendation_score'] = rec.get('score', 0.0)
            except RecommendationSession.DoesNotExist:
                metadata['source'] = 'organic'
                metadata['invalid_session_id'] = recommendation_session_id
        else:
            # No recommendation context - organic interaction
            metadata['source'] = 'organic'
        
        # Add additional context (device, referrer, etc.)
        metadata.update(additional_context)
        
        return metadata
    
    @staticmethod
    def create_interaction(
        user,
        startup,
        interaction_type: str,
        metadata: Dict,
        position=None
    ) -> Tuple[UserInteraction, bool]:
        """Create interaction with standardized metadata"""
        from django.db import IntegrityError
        
        try:
            interaction, created = UserInteraction.objects.get_or_create(
                user=user,
                startup=startup,
                interaction_type=interaction_type,
                defaults={
                    'metadata': metadata,
                    'position': position
                }
            )
            
            # Update metadata if interaction already existed
            if not created:
                interaction.metadata = metadata
                if position:
                    interaction.position = position
                interaction.save(update_fields=['metadata', 'position'])
            
            return interaction, created
        except IntegrityError:
            # Race condition: fetch existing
            interaction = UserInteraction.objects.get(
                user=user,
                startup=startup,
                interaction_type=interaction_type
            )
            interaction.metadata = metadata
            if position:
                interaction.position = position
            interaction.save(update_fields=['metadata', 'position'])
            return interaction, False

