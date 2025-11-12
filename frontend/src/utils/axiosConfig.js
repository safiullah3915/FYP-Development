import axios from 'axios';
import Cookies from 'js-cookie';
import { API_BASE_URL } from './api';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// List of endpoints that don't require authentication
const PUBLIC_ENDPOINTS = [
  '/signup',
  '/auth/login',
  '/auth/refresh',
  '/auth/forget-password',
  '/auth/send-verification-code',
  '/auth/verify',
  '/auth/test',
  '/auth/cookie-test',
  '/api/marketplace',
  '/api/collaborations',
  '/api/stats',
  '/api/search',
  '/'
];

// Helper function to check if endpoint is public
const isPublicEndpoint = (url) => {
  return PUBLIC_ENDPOINTS.some(endpoint => url.includes(endpoint)) || url === '/' || url === '';
};

// Helper function to check if user has auth token or session
const hasUserAuth = () => {
  // First check for auth token in localStorage (our primary auth method)
  const authToken = localStorage.getItem('auth_token');
  if (authToken) {
    console.log('🔑 Auth token found in localStorage');
    return true;
  }
  
  // Fallback to session cookie
  const sessionCookie = Cookies.get('sessionid');
  console.log('🔍 Session cookie (sessionid):', sessionCookie ? 'Found' : 'Not found');
  return !!sessionCookie;
};

// Helper function to get CSRF token
const getCSRFToken = () => {
  const csrfToken = Cookies.get('csrftoken') || Cookies.get('csrf');
  if (csrfToken) {
    console.log('🛡️ Found CSRF token in cookies');
  }
  return csrfToken;
};

// Request interceptor for session-based authentication
apiClient.interceptors.request.use(
  (config) => {
    console.log('🚀 Request interceptor:', {
      url: config.url,
      method: config.method,
      withCredentials: config.withCredentials
    });
    
    // Log all current cookies for debugging
    const allCookies = document.cookie;
    console.log('🍪 All cookies:', allCookies || 'No cookies found');
    
    // Ensure headers exist
    config.headers = config.headers || {};
    
    // Add auth token if available (primary authentication method)
    const authToken = localStorage.getItem('auth_token');
    if (authToken) {
      config.headers['Authorization'] = `Bearer ${authToken}`;
      console.log('🔑 Added auth_token to Authorization header');
    } else {
      console.log('🔑 No auth_token found in localStorage');
    }
    
    // Always add CSRF token if available
    const csrfToken = getCSRFToken();
    if (csrfToken) {
      config.headers['X-CSRFToken'] = csrfToken;
      console.log('🛡️ Added CSRF token');
    }
    
    const isPublic = isPublicEndpoint(config.url);
    
    // For non-public endpoints, check if auth exists
    if (!isPublic) {
      const hasAuth = hasUserAuth();
      console.log('🔍 Auth status for protected endpoint:', hasAuth ? 'Found' : 'Missing');
      
      if (!hasAuth) {
        console.warn('⚠️ No auth found for protected endpoint:', config.url);
        console.log('🍪 Current cookies:', document.cookie);
        console.log('🔑 auth_token in localStorage:', localStorage.getItem('auth_token') ? 'Found' : 'Missing');
      }
    }
    
    console.log('🚀 Final request config:', {
      url: config.baseURL + config.url,
      method: config.method?.toUpperCase(),
      hasCSRF: !!config.headers['X-CSRFToken'],
      withCredentials: config.withCredentials
    });
    
    return config;
  },
  (error) => {
    console.error('❌ Request interceptor ERROR:', error);
    return Promise.reject(error);
  }
);

// Response interceptor to handle authentication errors
apiClient.interceptors.response.use(
  (response) => {
    console.log('✅ API Response Success:', {
      status: response.status,
      url: response.config.url,
      method: response.config.method?.toUpperCase()
    });
    return response;
  },
  async (error) => {
    const errorDetails = {
      status: error.response?.status,
      statusText: error.response?.statusText,
      message: error.response?.data?.message || error.response?.data?.error || error.message,
      detail: error.response?.data?.detail,
      url: error.config?.url,
      method: error.config?.method?.toUpperCase()
    };
    
    console.error('🚨 API Response Error:', errorDetails);
    
    // Handle 401 Unauthorized - session expired
    if (error.response?.status === 401) {
      console.log('🔒 401 Unauthorized - Auth failed');
      
      // Clear auth token and cookies
      localStorage.removeItem('auth_token');
      Cookies.remove('sessionid');
      console.log('🔑 Cleared auth_token and session cookie');
      
      // Redirect to login if not already on auth pages
      if (!window.location.pathname.includes('/login') && !window.location.pathname.includes('/signup')) {
        console.log('Redirecting to login...');
        window.location.href = '/login';
      }
    }
    
    // Handle 403 Forbidden
    if (error.response?.status === 403) {
      console.log('🚫 403 Forbidden - Check authentication and permissions');
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;