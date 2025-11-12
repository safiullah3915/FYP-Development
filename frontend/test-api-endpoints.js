// Quick API Endpoint Verification Script
// Run this in browser console to test API connectivity

async function testEndpoints() {
    const baseURL = 'http://127.0.0.1:8000';
    const endpoints = [
        { method: 'GET', url: '/', name: 'Home' },
        { method: 'GET', url: '/api/marketplace', name: 'Marketplace' },
        { method: 'GET', url: '/api/collaborations', name: 'Collaborations' },
        { method: 'GET', url: '/api/stats', name: 'Stats' },
        { method: 'GET', url: '/api/search', name: 'Search' },
    ];

    console.log('🧪 Testing API Endpoints...\n');

    for (const endpoint of endpoints) {
        try {
            const response = await fetch(baseURL + endpoint.url, {
                method: endpoint.method,
                credentials: 'include',
            });
            
            const status = response.status;
            const statusText = response.ok ? '✅ OK' : '❌ ERROR';
            
            console.log(`${statusText} ${endpoint.method} ${endpoint.url} - ${status} (${endpoint.name})`);
        } catch (error) {
            console.log(`❌ ERROR ${endpoint.method} ${endpoint.url} - ${error.message} (${endpoint.name})`);
        }
    }

    console.log('\n🔍 Testing Authentication Endpoints...');
    
    // Test signup endpoint (should return validation errors but be accessible)
    try {
        const signupResponse = await fetch(baseURL + '/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
            credentials: 'include',
        });
        console.log(`✅ POST /signup - ${signupResponse.status} (Signup endpoint accessible)`);
    } catch (error) {
        console.log(`❌ POST /signup - ${error.message}`);
    }

    console.log('\n🎯 Next Steps:');
    console.log('1. If all endpoints show ✅, proceed with use case testing');
    console.log('2. If any show ❌, check if backend server is running');
    console.log('3. Backend should be running on http://127.0.0.1:8000');
    console.log('4. Frontend should be running on http://localhost:5174');
}

// Run the test
testEndpoints();