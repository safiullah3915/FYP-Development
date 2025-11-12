import Cookies from 'js-cookie';

export const debugAuthStatus = () => {
  console.log('🔍 ==================== AUTH DEBUG REPORT ====================');
  
  // Check localStorage tokens
  const localStorageKeys = Object.keys(localStorage);
  const authToken = localStorage.getItem('auth_token'); // Our primary auth token
  const accessToken = localStorage.getItem('access_token');
  const refreshToken = localStorage.getItem('refresh_token');
  const otherToken = localStorage.getItem('token');
  
  console.log('📦 LocalStorage Status:');
  console.log(`  - auth_token: ${authToken ? `${authToken.substring(0, 30)}...` : 'NOT FOUND'} [PRIMARY]`);
  console.log(`  - access_token: ${accessToken ? `${accessToken.substring(0, 30)}...` : 'NOT FOUND'} [LEGACY]`);
  console.log(`  - refresh_token: ${refreshToken ? `${refreshToken.substring(0, 30)}...` : 'NOT FOUND'} [LEGACY]`);
  console.log(`  - token: ${otherToken ? `${otherToken.substring(0, 30)}...` : 'NOT FOUND'} [LEGACY]`);
  console.log(`  - All keys: [${localStorageKeys.join(', ')}]`);
  
  // Check cookies using js-cookie library
  const sessionCookie = Cookies.get('sessionid'); // Django session cookie
  const tokenCookie = Cookies.get('token');
  const refreshCookie = Cookies.get('refresh_token');
  const csrfCookie = Cookies.get('csrftoken') || Cookies.get('csrf');
  const allCookiesObject = Cookies.get();
  const rawCookies = document.cookie;
  
  console.log('🍪 Cookie Status (using js-cookie):');
  console.log(`  - sessionid: ${sessionCookie ? `${sessionCookie.substring(0, 30)}...` : 'NOT FOUND'} [PRIMARY]`);
  console.log(`  - token: ${tokenCookie ? `${tokenCookie.substring(0, 30)}...` : 'NOT FOUND'} [LEGACY]`);
  console.log(`  - refresh_token: ${refreshCookie ? `${refreshCookie.substring(0, 30)}...` : 'NOT FOUND'} [LEGACY]`);
  console.log(`  - csrftoken: ${csrfCookie ? `${csrfCookie.substring(0, 20)}...` : 'NOT FOUND'}`);
  console.log(`  - All parsed cookies:`, allCookiesObject);
  console.log(`  - Raw document.cookie: ${rawCookies || 'No cookies found'}`);
  
  // Manual cookie parsing for debugging
  const manualCookies = rawCookies.split(';').reduce((acc, cookie) => {
    const [key, value] = cookie.trim().split('=');
    if (key && value) acc[key] = value;
    return acc;
  }, {});
  console.log(`  - Manual cookie parsing:`, manualCookies);
  
  // Determine authentication status
  const hasValidToken = !!(authToken || sessionCookie || tokenCookie || accessToken || otherToken);
  const hasRefreshToken = !!(refreshToken || refreshCookie);
  
  console.log('🔐 Authentication Analysis:');
  console.log(`  - Has auth_token: ${authToken ? '✅ YES' : '❌ NO'} [PRIMARY]`);
  console.log(`  - Has session cookie: ${sessionCookie ? '✅ YES' : '❌ NO'} [PRIMARY]`);
  console.log(`  - Has legacy tokens: ${(accessToken || tokenCookie || otherToken) ? '✅ YES' : '❌ NO'} [LEGACY]`);
  console.log(`  - Has refresh token: ${hasRefreshToken ? '✅ YES' : '❌ NO'}`);
  console.log(`  - Has CSRF token: ${csrfCookie ? '✅ YES' : '❌ NO'}`);
  console.log(`  - Can make authenticated requests: ${hasValidToken ? '✅ YES' : '❌ NO'}`);
  
  // Priority order for token selection (auth_token first, then fallbacks)
  const selectedToken = authToken || tokenCookie || accessToken || otherToken;
  const tokenSource = authToken ? 'localStorage auth_token' : tokenCookie ? 'cookie token' : accessToken ? 'localStorage access_token' : otherToken ? 'localStorage token' : 'none';
  console.log(`  - Selected token for requests: ${selectedToken ? `${selectedToken.substring(0, 20)}... (from ${tokenSource})` : 'NONE'}`);
  
  // Recommendations
  console.log('💡 Recommendations:');
  if (!hasValidToken) {
    console.log('  ❗ You need to login to get authentication tokens');
    console.log('  ❗ Navigate to /login and sign in with valid credentials');
  } else {
    console.log('  ✅ You have authentication tokens');
    console.log('  ✅ Startup creation should work if backend is running');
    if (!csrfCookie) {
      console.log('  ⚠️ No CSRF token found - might cause issues with some requests');
    }
  }
  
  console.log('🔚 ==================== END AUTH DEBUG ====================');
  
  return {
    hasAuthToken: !!authToken,
    hasSessionCookie: !!sessionCookie,
    hasAccessToken: hasValidToken, // For backwards compatibility
    hasRefreshToken: hasRefreshToken,
    hasCSRFToken: !!csrfCookie,
    canAuthenticate: hasValidToken,
    selectedToken: selectedToken,
    tokens: {
      localStorage: { authToken: !!authToken, accessToken: !!accessToken, refreshToken: !!refreshToken, token: !!otherToken },
      cookies: { sessionid: !!sessionCookie, token: !!tokenCookie, refreshToken: !!refreshCookie, csrf: !!csrfCookie }
    }
  };
};

// Function to manually set tokens (for testing)
export const setTestTokens = (authToken, refreshToken) => {
  console.log('🧪 Setting test tokens...');
  if (authToken) {
    localStorage.setItem('auth_token', authToken);
    console.log('✅ Set auth_token in localStorage');
  }
  if (refreshToken) {
    localStorage.setItem('refresh_token', refreshToken);
    console.log('✅ Set refresh_token in localStorage');
  }
  debugAuthStatus();
};

// Function to clear all tokens (for testing)
export const clearAllTokens = () => {
  console.log('🧩 Clearing all authentication tokens...');
  localStorage.removeItem('auth_token');
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('token');
  Cookies.remove('sessionid');
  Cookies.remove('token');
  Cookies.remove('refresh_token');
  console.log('✅ All tokens cleared');
  debugAuthStatus();
};

// Auto-run debug on import during development
if (process.env.NODE_ENV === 'development') {
  // Small delay to ensure localStorage is ready
  setTimeout(() => {
    debugAuthStatus();
  }, 100);
}