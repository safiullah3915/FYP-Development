// API utility functions
export const getApiBaseUrl = () => {
  // Check if we're in development mode
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'http://127.0.0.1:8000';
  }
  // For production, you would use your actual domain
  return 'http://127.0.0.1:8000'; // Default fallback
};

export const API_BASE_URL = getApiBaseUrl();
