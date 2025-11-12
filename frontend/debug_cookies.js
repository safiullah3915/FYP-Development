// Debug utility to check cookie behavior
// Run this in browser console after login

function debugCookies() {
  console.log('🍪 Cookie Debug Information');
  console.log('=' * 50);
  
  // Check document.cookie (raw browser cookies)
  console.log('📋 Raw document.cookie:', document.cookie);
  
  // Check js-cookie library
  import('js-cookie').then(({ default: Cookies }) => {
    const allCookies = Cookies.get();
    console.log('📦 js-cookie.get() all:', allCookies);
    console.log('🎫 js-cookie.get("token"):', Cookies.get('token'));
    console.log('🔄 js-cookie.get("refresh_token"):', Cookies.get('refresh_token'));
    
    // Try to set a test cookie
    Cookies.set('test_cookie', 'test_value', { path: '/' });
    console.log('🧪 Test cookie set, can read?:', Cookies.get('test_cookie'));
    
    // Check current domain and path
    console.log('🌐 Current domain:', window.location.hostname);
    console.log('📍 Current path:', window.location.pathname);
    console.log('🔗 Current origin:', window.location.origin);
  });
}

// To use: Copy this to browser console and run debugCookies() after login