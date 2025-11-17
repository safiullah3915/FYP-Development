import apiClient from './axiosConfig';
import { normalizeId } from './idUtils';

const withStartupId = (id) => normalizeId(id);

// Authentication endpoints
export const authAPI = {
  signup: (userData) => apiClient.post('/signup', userData),
  login: (credentials) => apiClient.post('/auth/login', credentials),
  logout: () => apiClient.post('/auth/logout'),
  verifyEmail: (token) => apiClient.post('/auth/verify', { token }),
  forgotPassword: (email) => apiClient.post('/auth/forget-password', { email }),
  refresh: () => apiClient.post('/auth/refresh'),
  sendVerificationCode: (email) => apiClient.post('/auth/send-verification-code', { email }),
  testAuth: () => apiClient.get('/auth/test'),
  testCookie: () => apiClient.get('/auth/cookie-test'),
  getAccountByToken: (token) => apiClient.get(`/account/${token}`),
};

// User management endpoints
export const userAPI = {
  getProfile: () => apiClient.get('/api/users/profile'),
  getProfileDetail: () => apiClient.get('/api/users/profile-detail'),
  updateProfile: (data) => apiClient.put('/api/users/profile', data),
  patchProfile: (data) => apiClient.patch('/api/users/profile', data),
  getProfileData: () => apiClient.get('/api/users/profile-data'),
  updateProfileData: (data) => apiClient.patch('/api/users/profile-data', data),
  getUserStartups: () => apiClient.get('/api/users/startups'),
  getUserStartupsWithApplications: () => apiClient.get('/api/users/startups/with-applications'),
  getUserApplications: () => apiClient.get('/api/users/applications'),
  getUserFavorites: () => apiClient.get('/api/users/favorites'),
  getUserInterests: () => apiClient.get('/api/users/interests'),
};

// Startup management endpoints
export const startupAPI = {
  createStartup: (startupData) => apiClient.post('/api/startups', startupData),
  getStartup: (id) => apiClient.get(`/api/startups/${withStartupId(id)}`),
  updateStartup: (id, data) => apiClient.put(`/api/startups/${withStartupId(id)}`, data),
  patchStartup: (id, data) => apiClient.patch(`/api/startups/${withStartupId(id)}`, data),
  deleteStartup: (id) => apiClient.delete(`/api/startups/${withStartupId(id)}`),
  getStartupApplications: (id) => apiClient.get(`/api/startups/${withStartupId(id)}/applications`),
  getStartupPositions: (id) => apiClient.get(`/api/startups/${withStartupId(id)}/positions`),
  createStartupPosition: (id, positionData) => apiClient.post(`/api/startups/${withStartupId(id)}/positions`, positionData),
  getStartupInterests: (id) => apiClient.get(`/api/startups/${withStartupId(id)}/interests`),
  expressInterest: (id, data = {}) => apiClient.post(`/api/startups/${withStartupId(id)}/interest`, data),
  toggleFavorite: (id) => apiClient.post(`/api/startups/${withStartupId(id)}/favorite`),
  deleteFavorite: (id) => apiClient.delete(`/api/startups/${withStartupId(id)}/favorite`),
  convertToMarketplace: (id, data) => apiClient.post(`/api/startups/${withStartupId(id)}/convert-to-marketplace`, data),
};

// Position management endpoints
export const positionAPI = {
  getAllPositions: (params) => apiClient.get('/api/positions', { params }), // Get all available positions for job search
  getPosition: (id) => apiClient.get(`/api/positions/${id}`),
  updatePosition: (id, data) => apiClient.put(`/api/positions/${id}`, data),
  patchPosition: (id, data) => apiClient.patch(`/api/positions/${id}`, data),
  deletePosition: (id) => apiClient.delete(`/api/positions/${id}`),
  closePosition: (id) => apiClient.post(`/api/positions/${id}/close`),
  openPosition: (id) => apiClient.post(`/api/positions/${id}/open`),
  getStartupPositions: (startupId) => apiClient.get(`/api/startups/${withStartupId(startupId)}/positions`),
  createPosition: (startupId, positionData) => apiClient.post(`/api/startups/${withStartupId(startupId)}/positions`, positionData),
  getPositionApplications: (positionId) => apiClient.get(`/api/positions/${positionId}/applications`),
};

// Application management endpoints
export const applicationAPI = {
  applyForCollaboration: (id, applicationData) => apiClient.post(`/api/collaborations/${id}/apply`, applicationData),
  approveApplication: (id) => apiClient.post(`/api/applications/${id}/approve`),
  declineApplication: (id) => apiClient.post(`/api/applications/${id}/decline`),
};

// Marketplace & Discovery endpoints
export const marketplaceAPI = {
  getMarketplace: (params) => apiClient.get('/api/marketplace', { params }),
  getCollaborations: (params) => apiClient.get('/api/collaborations', { params }),
  search: (query) => apiClient.get('/api/search', { params: { q: query } }),
  getStats: () => apiClient.get('/api/stats'),
};

// Search endpoints
export const searchAPI = {
  searchStartups: (queryString) => apiClient.get(`/api/search?${queryString}`),
  searchMarketplace: (params) => apiClient.get('/api/marketplace', { params }),
  searchCollaborations: (params) => apiClient.get('/api/collaborations', { params }),
};

// Notification endpoints
export const notificationAPI = {
  getNotifications: () => apiClient.get('/api/notifications'),
  markAsRead: (id) => apiClient.post(`/api/notifications/${id}/read`),
  markAllAsRead: () => apiClient.post('/api/notifications/read-all'),
};

// Messaging endpoints
export const messageAPI = {
  getConversations: () => apiClient.get('/api/messages'),
  createConversation: (data) => apiClient.post('/api/messages', data),
  getConversation: (id) => apiClient.get(`/api/messages/${id}`),
  getMessages: (conversationId) => apiClient.get(`/api/messages/${conversationId}/messages`),
  sendMessage: (conversationId, messageData) => apiClient.post(`/api/messages/${conversationId}/messages`, messageData),
  getOnlineUsers: () => apiClient.get('/api/messages/users/online'),
};

// File upload endpoints
export const uploadAPI = {
  uploadFile: (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress
    });
  },
  uploadResume: (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/api/upload/resume', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress
    });
  },
  uploadStartupImage: (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/api/upload/startup-image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress
    });
  },
  uploadProfilePicture: (file, onUploadProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post('/api/upload/profile-picture', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress
    });
  },
  getUserUploads: () => apiClient.get('/api/uploads'),
};

// Recommendation system endpoints
export const recommendationAPI = {
  // Onboarding preferences
  getOnboardingPreferences: () => apiClient.get('/api/onboarding/preferences'),
  saveOnboardingPreferences: (preferences) => apiClient.post('/api/onboarding/preferences', preferences),
  updateOnboardingPreferences: (preferences) => apiClient.patch('/api/onboarding/preferences', preferences),
  
  // Store recommendation session
  storeRecommendationSession: (sessionData) => 
    apiClient.post('/api/recommendations/session', sessionData),
  
  // Explicit feedback with recommendation context
  likeStartup: (startupId, recommendationContext = null) => {
    const normalizedId = withStartupId(startupId);
    const data = {};
    if (recommendationContext) {
      data.recommendation_session_id = recommendationContext.sessionId;
      data.recommendation_rank = recommendationContext.rank;
    }
    return apiClient.post(`/api/startups/${normalizedId}/like`, data);
  },
  
  unlikeStartup: (startupId, recommendationContext = null) => {
    const normalizedId = withStartupId(startupId);
    const params = {};
    if (recommendationContext) {
      params.recommendation_session_id = recommendationContext.sessionId;
      params.recommendation_rank = recommendationContext.rank;
    }
    return apiClient.delete(`/api/startups/${normalizedId}/unlike`, { params });
  },
  
  dislikeStartup: (startupId, recommendationContext = null) => {
    const normalizedId = withStartupId(startupId);
    const data = {};
    if (recommendationContext) {
      data.recommendation_session_id = recommendationContext.sessionId;
      data.recommendation_rank = recommendationContext.rank;
    }
    return apiClient.post(`/api/startups/${normalizedId}/dislike`, data);
  },
  
  undislikeStartup: (startupId, recommendationContext = null) => {
    const normalizedId = withStartupId(startupId);
    const params = {};
    if (recommendationContext) {
      params.recommendation_session_id = recommendationContext.sessionId;
      params.recommendation_rank = recommendationContext.rank;
    }
    return apiClient.delete(`/api/startups/${normalizedId}/undislike`, { params });
  },
  
  getStartupInteractionStatus: (startupId) => apiClient.get(`/api/startups/${withStartupId(startupId)}/interaction-status`),
  
  // Trending startups
  getTrendingStartups: (params) => apiClient.get('/api/recommendations/trending/startups', { params }),
  
  // Personalized recommendations for entrepreneurs
  getPersonalizedDevelopers: (startupId, params) => 
    apiClient.get(`/api/recommendations/personalized/developers/${withStartupId(startupId)}`, { params }),
  
  getPersonalizedInvestors: (startupId, params) => 
    apiClient.get(`/api/recommendations/personalized/investors/${withStartupId(startupId)}`, { params }),
  
  // Personalized collaboration startups for entrepreneurs
  getPersonalizedCollaborationStartups: (params) => 
    apiClient.get('/api/recommendations/personalized/startups', { 
      params: { ...params, type: 'collaboration' } 
    }),
  
  // Personalized startup recommendations for investors
  getPersonalizedStartups: (params) => 
    apiClient.get('/api/recommendations/personalized/startups', { 
      params 
    }),
};

// Export the configured axios client for custom requests
export { apiClient };

// Helper function to check if user is authenticated
export const isAuthenticated = () => {
  return !!document.cookie.split(';').find(row => row.trim().startsWith('token='));
};