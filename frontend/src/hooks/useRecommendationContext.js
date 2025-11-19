import { useState, useCallback } from 'react';
import { recommendationAPI } from '../utils/apiServices';

// Simple UUID v4 generator (fallback if crypto.randomUUID is not available)
const generateUUID = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback UUID v4 generator
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

/**
 * Hook for tracking recommendation sessions
 * Can be used in any component showing recommendations
 * 
 * Stores recommendation sessions and provides context for interactions
 */
export const useRecommendationContext = () => {
  const [currentSession, setCurrentSession] = useState(null);
  
  /**
   * Store a recommendation session
   * @param {Object} params - Session parameters
   * @param {Array} params.recommendations - Array of recommendations with startup_id/user_id, rank, score
   * @param {string} params.useCase - Use case (e.g., 'developer_startup', 'startup_developer', 'startup_investor')
   * @param {string} params.method - Recommendation method (e.g., 'two_tower', 'als')
   * @param {string} params.modelVersion - Model version (e.g., 'two_tower_v1.0')
   * @param {string} params.startupId - Optional startup_id for reverse use cases
   */
  const storeSession = useCallback(async ({ recommendations, useCase, method, modelVersion, startupId }) => {
    try {
      // Generate session ID if not provided
      const sessionId = generateUUID();
      
      // Determine if this is a reverse use case
      const reverseUseCases = ['startup_developer', 'startup_investor'];
      const isReverse = reverseUseCases.includes(useCase);
      
      // Prepare session data according to API spec
      const sessionData = {
        recommendation_session_id: sessionId,
        use_case: useCase,
        method: method || 'two_tower',
        model_version: modelVersion || 'two_tower_v1.0',
        recommendations: recommendations.map((rec, index) => {
          // For reverse use cases, recommendations contain user_ids (developers/investors)
          // For forward use cases, recommendations contain startup_ids
          if (isReverse) {
            return {
              user_id: rec.user_id || rec.id || rec.developer_id || rec.investor_id,
              rank: rec.rank !== undefined ? rec.rank : index + 1,
              score: rec.score !== undefined ? rec.score : rec.recommendation_score || 0.0,
              method: rec.method || method || 'two_tower'
            };
          } else {
            return {
              startup_id: rec.startup_id || rec.id,
              rank: rec.rank !== undefined ? rec.rank : index + 1,
              score: rec.score !== undefined ? rec.score : rec.recommendation_score || 0.0,
              method: rec.method || method || 'two_tower'
            };
          }
        })
      };
      
      // Add startup_id for reverse use cases
      if (isReverse && startupId) {
        sessionData.startup_id = startupId;
      }
      
      console.log('[useRecommendationContext] Storing session:', sessionData);
      
      await recommendationAPI.storeRecommendationSession(sessionData);
      
      setCurrentSession({
        sessionId: sessionId,
        recommendations: sessionData.recommendations,
        useCase: useCase,
        method: method
      });
      
      console.log('[useRecommendationContext] Session stored successfully');
    } catch (error) {
      console.error('[useRecommendationContext] Failed to store recommendation session:', error);
      // Don't throw - allow UI to continue even if session storage fails
    }
  }, []);
  
  /**
   * Get recommendation context for a specific startup
   * @param {string} startupId - Startup ID
   * @returns {Object|null} Context with sessionId and rank, or null if not found
   */
  const getRecommendationContext = useCallback((startupId) => {
    if (!currentSession || !startupId) return null;
    
    const recommendation = currentSession.recommendations.find(
      r => String(r.startup_id) === String(startupId)
    );
    
    if (!recommendation) return null;
    
    return {
      sessionId: currentSession.sessionId,
      rank: recommendation.rank,
      score: recommendation.score,
      method: recommendation.method || currentSession.method
    };
  }, [currentSession]);
  
  /**
   * Clear current session (useful when navigating away)
   */
  const clearSession = useCallback(() => {
    setCurrentSession(null);
  }, []);
  
  return {
    currentSession,
    storeSession,
    getRecommendationContext,
    clearSession
  };
};

