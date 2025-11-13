import { useState, useCallback } from 'react';
import { recommendationAPI } from '../utils/apiServices';

/**
 * Hook for tracking recommendation sessions
 * Can be used in any component showing recommendations
 */
export const useRecommendationContext = () => {
  const [currentSession, setCurrentSession] = useState(null);
  
  const storeSession = useCallback(async (sessionData) => {
    try {
      await recommendationAPI.storeRecommendationSession(sessionData);
      setCurrentSession({
        sessionId: sessionData.recommendation_session_id,
        recommendations: sessionData.recommendations || []
      });
    } catch (error) {
      console.error('Failed to store recommendation session:', error);
    }
  }, []);
  
  const getRecommendationContext = useCallback((startupId) => {
    if (!currentSession) return null;
    
    const recommendation = currentSession.recommendations.find(
      r => r.startup_id === startupId
    );
    
    if (!recommendation) return null;
    
    return {
      sessionId: currentSession.sessionId,
      rank: recommendation.rank
    };
  }, [currentSession]);
  
  return {
    currentSession,
    storeSession,
    getRecommendationContext
  };
};

