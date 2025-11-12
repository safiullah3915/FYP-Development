import React, { useState } from 'react';
import { userAPI, marketplaceAPI } from '../../utils/apiServices';
import { useAuth } from '../../contexts/AuthContext';

const ApiTest = () => {
  const [testResults, setTestResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const { isAuthenticated, user } = useAuth();

  const addResult = (testName, success, data = null, error = null) => {
    setTestResults(prev => [...prev, {
      testName,
      success,
      data,
      error,
      timestamp: new Date().toLocaleTimeString()
    }]);
  };

  const testProtectedEndpoint = async (testName, apiCall) => {
    try {
      console.log(`Testing ${testName}...`);
      const response = await apiCall();
      addResult(testName, true, response.data);
      return true;
    } catch (error) {
      console.error(`${testName} failed:`, error);
      addResult(testName, false, null, error.response?.data || error.message);
      return false;
    }
  };

  const runTests = async () => {
    setLoading(true);
    setTestResults([]);

    // Test user profile endpoint (requires JWT)
    await testProtectedEndpoint('Get User Profile', userAPI.getProfile);
    
    // Test marketplace endpoint (might require JWT)
    await testProtectedEndpoint('Get Marketplace', marketplaceAPI.getMarketplace);
    
    // Test user profile data (requires JWT)
    await testProtectedEndpoint('Get Profile Data', userAPI.getProfileData);
    
    // Test user startups (requires JWT)
    await testProtectedEndpoint('Get User Startups', userAPI.getUserStartups);
    
    setLoading(false);
  };

  const clearResults = () => {
    setTestResults([]);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h2>JWT Token Authentication Test</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <p><strong>Authentication Status:</strong> {isAuthenticated ? '✅ Authenticated' : '❌ Not Authenticated'}</p>
        {user && (
          <p><strong>User:</strong> {user.username || user.name || 'Unknown'} ({user.email})</p>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={runTests} 
          disabled={loading}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Running Tests...' : 'Test Protected Endpoints'}
        </button>
        
        <button 
          onClick={clearResults}
          style={{
            padding: '10px 20px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Clear Results
        </button>
      </div>

      <div>
        <h3>Test Results:</h3>
        {testResults.length === 0 ? (
          <p style={{ color: '#6c757d' }}>No tests run yet. Click "Test Protected Endpoints" to start.</p>
        ) : (
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {testResults.map((result, index) => (
              <div 
                key={index} 
                style={{
                  padding: '10px',
                  margin: '10px 0',
                  border: `2px solid ${result.success ? '#28a745' : '#dc3545'}`,
                  borderRadius: '4px',
                  backgroundColor: result.success ? '#d4edda' : '#f8d7da'
                }}
              >
                <h4 style={{ margin: '0 0 10px 0' }}>
                  {result.success ? '✅' : '❌'} {result.testName}
                  <span style={{ fontSize: '12px', color: '#6c757d', float: 'right' }}>
                    {result.timestamp}
                  </span>
                </h4>
                
                {result.success ? (
                  <details>
                    <summary style={{ cursor: 'pointer', color: '#155724' }}>
                      Success - View Response Data
                    </summary>
                    <pre style={{ 
                      fontSize: '12px', 
                      backgroundColor: '#f8f9fa', 
                      padding: '10px', 
                      overflow: 'auto',
                      maxHeight: '200px'
                    }}>
                      {JSON.stringify(result.data, null, 2)}
                    </pre>
                  </details>
                ) : (
                  <details>
                    <summary style={{ cursor: 'pointer', color: '#721c24' }}>
                      Error - View Details
                    </summary>
                    <pre style={{ 
                      fontSize: '12px', 
                      backgroundColor: '#f8f9fa', 
                      padding: '10px', 
                      overflow: 'auto',
                      maxHeight: '200px'
                    }}>
                      {JSON.stringify(result.error, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ApiTest;