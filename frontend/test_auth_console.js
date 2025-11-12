// Copy and paste this script into your browser console to debug authentication

console.log('🔍 AUTHENTICATION DEBUG SCRIPT');
console.log('================================');

// Check localStorage tokens
const accessToken = localStorage.getItem('access_token');
const refreshToken = localStorage.getItem('refresh_token');

console.log('📱 localStorage tokens:');
console.log('- access_token:', accessToken ? `${accessToken.substring(0, 30)}...` : '❌ NOT FOUND');
console.log('- refresh_token:', refreshToken ? `${refreshToken.substring(0, 30)}...` : '❌ NOT FOUND');

// Check cookies
const allCookies = document.cookie;
console.log('🍪 All cookies:', allCookies || '❌ No cookies found');

// Parse cookies
const cookieToken = document.cookie.split(';').find(row => row.trim().startsWith('token='));
const cookieRefreshToken = document.cookie.split(';').find(row => row.trim().startsWith('refresh_token='));

console.log('🍪 Parsed cookies:');
console.log('- token:', cookieToken ? cookieToken.substring(0, 50) + '...' : '❌ NOT FOUND');
console.log('- refresh_token:', cookieRefreshToken ? cookieRefreshToken.substring(0, 50) + '...' : '❌ NOT FOUND');

// Test manual API call
async function testStartupCreation() {
    console.log('\n🚀 TESTING STARTUP CREATION REQUEST');
    console.log('====================================');
    
    const token = accessToken || (cookieToken ? cookieToken.split('=')[1] : null);
    
    if (!token) {
        console.error('❌ No token available for testing');
        console.log('🔧 Fix: Please login first');
        return;
    }
    
    console.log('🎫 Using token:', token.substring(0, 30) + '...');
    
    const testData = {
        title: 'Test Startup',
        description: 'Test startup description',
        field: 'Technology',
        type: 'marketplace',
        category: 'saas'
    };
    
    try {
        const response = await fetch('http://127.0.0.1:8000/api/startups', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            credentials: 'include',
            body: JSON.stringify(testData)
        });
        
        console.log('📊 Response status:', response.status);
        console.log('📊 Response headers:', Object.fromEntries(response.headers.entries()));
        
        const responseData = await response.text();
        console.log('📄 Response data:', responseData);
        
        if (response.ok) {
            console.log('✅ SUCCESS: Startup creation works!');
        } else {
            console.error('❌ FAILED: Startup creation failed');
            if (response.status === 403) {
                console.error('🔒 This is the 403 error you\'re experiencing');
                console.error('💡 Solution: Check if user is properly logged in and token is valid');
            }
        }
    } catch (error) {
        console.error('❌ Network error:', error);
    }
}

// Run the test
console.log('\n🧪 Running startup creation test...');
testStartupCreation();

console.log('\n💡 DEBUGGING TIPS:');
console.log('1. If no tokens found: User needs to login first');
console.log('2. If tokens found but API fails: Token might be expired or invalid');
console.log('3. Check browser Network tab for actual request headers');
console.log('4. Verify Django server is running on http://127.0.0.1:8000');