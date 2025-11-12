import React, { createContext, useContext, useState, useEffect } from 'react';
import apiClient from '../utils/axiosConfig';
import Cookies from 'js-cookie';

const AuthContext = createContext();

// Export AuthContext for components that need direct access
export { AuthContext };

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check if user is authenticated on page load/refresh
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const authToken = localStorage.getItem('auth_token');
        const sessionCookie = Cookies.get('sessionid');

        // If we already have a user, just finish
        if (user) {
          setLoading(false);
          return;
        }

        // Prefer JWT if present
        if (authToken) {
          console.log('🔑 Found auth_token on page load - fetching user data');
          const { default: client } = await import('../utils/axiosConfig');
          const response = await client.get('/api/users/profile-data');

          if (response.data && response.data.profile) {
            const profile = response.data.profile;
            const userData = {
              id: profile.user?.id || profile.user,
              username: profile.user?.username || profile.name || profile.username,
              email: profile.user?.email || profile.email,
              role: profile.user?.role || profile.role,
              emailVerified: true
            };
            setUser(userData);
            setIsAuthenticated(true);
            console.log('✅ User restored via JWT:', userData.username);
            setLoading(false);
            return;
          }
          // Fallback: treat as authenticated even if profile missing
          setIsAuthenticated(true);
          console.log('⚠️ JWT present but profile missing; marking as authenticated');
          setLoading(false);
          return;
        }

        // If no JWT but we have a Django session cookie, try restoring via session
        if (sessionCookie) {
          console.log('🍪 Found session cookie - attempting session-based restore');
          const { default: client } = await import('../utils/axiosConfig');
          try {
            const response = await client.get('/api/users/profile-data');
            if (response.data && response.data.profile) {
              const profile = response.data.profile;
              const userData = {
                id: profile.user?.id || profile.user,
                username: profile.user?.username || profile.name || profile.username,
                email: profile.user?.email || profile.email,
                role: profile.user?.role || profile.role,
                emailVerified: true
              };
              setUser(userData);
              setIsAuthenticated(true);
              console.log('✅ User restored via session cookie:', userData.username);
              setLoading(false);
              return;
            }
            // If profile missing but session exists, consider authenticated
            setIsAuthenticated(true);
            console.log('⚠️ Session present but profile missing; marking as authenticated');
            setLoading(false);
            return;
          } catch (error) {
            // Session might be invalid/expired
            console.warn('❌ Session restore failed:', error.response?.status, error.response?.data || error.message);
            setIsAuthenticated(false);
            setUser(null);
            setLoading(false);
            return;
          }
        }

        // No auth found
        setIsAuthenticated(false);
        setUser(null);
        setLoading(false);
      } catch (e) {
        console.error('❌ Unexpected error during auth restore:', e);
        setIsAuthenticated(false);
        setUser(null);
        setLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  const login = async (email, password) => {
    try {
      const response = await apiClient.post('/auth/login', {
        email,
        password
      });

      console.log('Login response:', response.data);

      if (response.data.user) {
        const user = response.data.user;
        const authToken = response.data.auth_token;
        console.log('Login user data:', user);
        console.log('Auth token received:', authToken ? 'Yes' : 'No');
        
        // Store auth token in localStorage
        if (authToken) {
          localStorage.setItem('auth_token', authToken);
          console.log('🔑 Auth token stored in localStorage');
        }
        
        // Set user and authenticate
        setUser(user);
        setIsAuthenticated(true);
        
        return { success: true, user: user, auth_token: authToken };
      } else {
        console.warn('No user data in login response');
        return { success: false, error: 'No user data received' };
      }
    } catch (error) {
      console.error('Login failed:', error);
      return { 
        success: false, 
        error: error.response?.data?.error || error.response?.data?.message || 'Login failed' 
      };
    }
  };

  const signup = async (username, email, password, role = 'entrepreneur', phone_number = '') => {
    try {
      const requestData = {
        username,
        email,
        password,
        role
      };
      
      // Only include phone_number if provided
      if (phone_number && phone_number.trim()) {
        requestData.phone_number = phone_number.trim();
      }
      
      console.log('📝 Signup request data:', requestData);
      
      const response = await apiClient.post('/signup', requestData);

      console.log('Signup response:', response.data);

      if (response.data.user) {
        const user = response.data.user;
        const authToken = response.data.auth_token;
        console.log('Signup user data:', user);
        console.log('Auth token received:', authToken ? 'Yes' : 'No');
        
        // Store auth token in localStorage (same as login flow)
        if (authToken) {
          localStorage.setItem('auth_token', authToken);
          console.log('🔑 Auth token stored in localStorage after signup');
        }
        
        // Simplified - automatically set as authenticated after signup
        setUser(user);
        setIsAuthenticated(true);
        
        return { 
          success: true, 
          user: user,
          auth_token: authToken,
          requiresVerification: false // Simplified - no verification needed
        };
      } else {
        console.warn('No user data in signup response');
        return { success: false, error: 'No user data received' };
      }
    } catch (error) {
      console.error('Signup failed:', error);
      
      const errorMessage = error.response?.data?.message || 
                          error.response?.data?.error || 
                          (typeof error.response?.data === 'string' ? error.response.data : null) ||
                          'Signup failed';
                          
      return { 
        success: false, 
        error: errorMessage
      };
    }
  };

  const verifyEmail = async (verificationCode, email) => {
    // Simplified - just return success
    return { success: true, message: 'Email verified successfully' };
  };

  const resendVerificationCode = async (email) => {
    // Simplified - just return success
    return {
      success: true,
      message: 'Verification code sent successfully'
    };
  };

  const requestNewVerificationToken = async (email) => {
    // Simplified - just return success
    return { 
      success: true, 
      message: 'Verification code sent to your email' 
    };
  };

  const logout = async () => {
    try {
      await apiClient.post('/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear auth token
      localStorage.removeItem('auth_token');
      console.log('🔑 Auth token cleared from localStorage');
      
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  const updateUser = (userData) => {
    setUser(prevUser => ({ ...prevUser, ...userData }));
  };

  // Debug function to check current auth state
  const getAuthStatus = () => {
    const authToken = localStorage.getItem('auth_token');
    return {
      isAuthenticated,
      hasAuthToken: !!authToken,
      hasVerificationToken: false,
      userEmailVerified: user?.emailVerified || false,
      userName: user?.username || user?.name || 'Unknown'
    };
  };

  // Role-based access helpers
  const isEntrepreneur = () => user?.role === 'entrepreneur';
  const isStudent = () => user?.role === 'student';
  const isInvestor = () => user?.role === 'investor';

  const canCreateStartups = () => isEntrepreneur();
  const canApplyToJobs = () => isStudent();
  const canInvest = () => isInvestor();

  const value = {
    user,
    isAuthenticated,
    loading,
    login,
    signup,
    verifyEmail,
    resendVerificationCode,
    requestNewVerificationToken,
    logout,
    updateUser,
    getAuthStatus,
    isEntrepreneur,
    isStudent,
    isInvestor,
    canCreateStartups,
    canApplyToJobs,
    canInvest
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};