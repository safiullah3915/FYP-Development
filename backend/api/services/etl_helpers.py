from api.recommendation_models import UserInteraction, RecommendationSession
from django.db.models import Q
from django.utils import timezone
from datetime import datetime
from typing import List


class ETLQueryHelpers:
    """Helper methods for ETL data extraction - ready for future use"""
    
    @staticmethod
    def get_recommendation_interactions(
        date_from: datetime = None,
        date_to: datetime = None,
        use_case: str = None
    ):
        """
        Get all interactions from recommendations
        Optimized query for ETL extraction
        """
        query = UserInteraction.objects.filter(
            metadata__source='recommendation'
        )
        
        if date_from:
            query = query.filter(created_at__gte=date_from)
        if date_to:
            query = query.filter(created_at__lte=date_to)
        
        # Filter by use_case if provided (requires session join)
        if use_case:
            # Get session IDs for this use_case
            session_ids = RecommendationSession.objects.filter(
                use_case=use_case
            ).values_list('id', flat=True)
            
            # Filter interactions by session IDs
            query = query.filter(
                metadata__recommendation_session_id__in=[str(sid) for sid in session_ids]
            )
        
        return query.select_related('user', 'startup', 'position')
    
    @staticmethod
    def get_organic_interactions(date_from: datetime = None, date_to: datetime = None):
        """Get all organic interactions"""
        query = UserInteraction.objects.filter(
            metadata__source='organic'
        )
        
        if date_from:
            query = query.filter(created_at__gte=date_from)
        if date_to:
            query = query.filter(created_at__lte=date_to)
        
        return query.select_related('user', 'startup')
    
    @staticmethod
    def get_unprocessed_sessions(use_case: str = None, limit: int = 1000):
        """Get sessions not yet processed by ETL"""
        query = RecommendationSession.objects.filter(etl_processed=False)
        
        if use_case:
            query = query.filter(use_case=use_case)
        
        return query[:limit]
    
    @staticmethod
    def mark_sessions_processed(session_ids: List[str]):
        """Mark sessions as processed by ETL"""
        RecommendationSession.objects.filter(id__in=session_ids).update(
            etl_processed=True,
            etl_processed_at=timezone.now()
        )

