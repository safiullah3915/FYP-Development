# ✅ Django ↔ Flask Integration Complete!

## 🎉 What's Been Added

I've successfully integrated your **Two-Tower recommendation model** into the Django backend through proxy endpoints!

---

## 📍 New Django Endpoints

### 1. **Personalized Startup Recommendations** (Uses Two-Tower!)
```
GET /api/recommendations/personalized/startups
```

**Who can use:** Authenticated developers/students and investors

**What it does:**
- Automatically detects user role (developer/investor)
- Routes to Flask Two-Tower endpoint
- Returns personalized startup recommendations
- **Uses Two-Tower model for warm users (5+ interactions)**
- Falls back to content-based for cold users

**Query Parameters:**
- `limit` - Number of results (default: 10)
- `type` - Startup type filter
- `min_funding` / `max_funding` - Funding range
- `category` - Category filter
- `stage` - Stage filter

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/recommendations/personalized/startups?limit=20&category=AI"
```

**Example Response:**
```json
{
  "startups": [...],
  "total": 20,
  "user_id": "user-uuid",
  "user_role": "student",
  "interaction_count": 15,
  "method_used": "two_tower",
  "model_version": "two_tower_v1.0",
  "session_id": "session-uuid"
}
```

---

### 2. **Personalized Developer Recommendations for Startups**
```
GET /api/recommendations/personalized/developers/<startup_id>
```

**Who can use:** Authenticated startup owners

**What it does:**
- Returns developers that match the startup's needs
- Verifies startup ownership
- Routes to Flask recommendation service

**Query Parameters:**
- `limit` - Number of results
- `skills` - Comma-separated skill filters

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/recommendations/personalized/developers/startup-uuid?limit=10&skills=Python,React"
```

---

### 3. **Personalized Investor Recommendations for Startups**
```
GET /api/recommendations/personalized/investors/<startup_id>
```

**Who can use:** Authenticated startup owners

**What it does:**
- Returns investors that match the startup's funding needs
- Verifies startup ownership
- Routes to Flask recommendation service

**Query Parameters:**
- `limit` - Number of results
- `min_investment` - Minimum investment amount

**Example Request:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/recommendations/personalized/investors/startup-uuid?limit=5"
```

---

## 🔄 How It Works

### Architecture Flow:

```
Frontend
   ↓
Django Backend (Port 8000)
   ↓ [Proxy Request]
Flask Recommendation Service (Port 5000)
   ↓ [Two-Tower Model / Content-Based]
Response with Recommendations
```

### Authentication Flow:

1. Frontend sends request to Django with auth token
2. Django validates user authentication
3. Django extracts user info (ID, role)
4. Django constructs Flask URL based on user role
5. Django proxies request to Flask with parameters
6. Flask processes with Two-Tower or Content-Based
7. Django adds user context to response
8. Frontend receives recommendations

---

## 🎯 User Role Routing

| User Role | Django Endpoint | Flask Endpoint Called |
|-----------|----------------|----------------------|
| `student` or `developer` | `/api/recommendations/personalized/startups` | `/api/recommendations/startups/for-developer/{user_id}` |
| `investor` | `/api/recommendations/personalized/startups` | `/api/recommendations/startups/for-investor/{user_id}` |
| Startup Owner | `/api/recommendations/personalized/developers/{startup_id}` | `/api/recommendations/developers/for-startup/{startup_id}` |
| Startup Owner | `/api/recommendations/personalized/investors/{startup_id}` | `/api/recommendations/investors/for-startup/{startup_id}` |

---

## 🚀 Testing the Integration

### 1. Start Both Services

**Terminal 1 - Flask Service:**
```bash
cd recommendation_service
python app.py
```
Output: `Running on http://localhost:5000`

**Terminal 2 - Django Backend:**
```bash
cd backend
python manage.py runserver
```
Output: `Starting development server at http://127.0.0.1:8000/`

### 2. Test Personalized Recommendations

**Get your auth token:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "password"}'
```

**Get personalized startups:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/recommendations/personalized/startups?limit=10"
```

### 3. Verify Two-Tower Model is Used

Check the response for:
```json
{
  "method_used": "two_tower",  // ← Should be "two_tower" for warm users
  "model_version": "two_tower_v1.0",
  "interaction_count": 15  // ← Should be >= 5 for Two-Tower
}
```

---

## 🔧 Configuration

### Environment Variable (Optional)

You can set the Flask service URL via environment variable:

```bash
# In your .env or environment
export FLASK_RECOMMENDATION_SERVICE_URL=http://localhost:5000
```

Default: `http://localhost:5000`

---

## 📊 Error Handling

All endpoints include graceful error handling:

**If Flask is unavailable:**
```json
{
  "startups": [],
  "total": 0,
  "error": "Recommendation service temporarily unavailable",
  "details": "Connection refused"
}
```
HTTP Status: `503 Service Unavailable`

**If user not authenticated:**
```json
{
  "error": "Authentication required"
}
```
HTTP Status: `401 Unauthorized`

**If user doesn't own startup:**
```json
{
  "error": "You do not have permission to get recommendations for this startup"
}
```
HTTP Status: `403 Forbidden`

---

## 📝 Frontend Integration Example

### React/JavaScript:

```javascript
// Get personalized startup recommendations
const getPersonalizedStartups = async (authToken, filters = {}) => {
  const params = new URLSearchParams(filters);
  
  const response = await fetch(
    `http://localhost:8000/api/recommendations/personalized/startups?${params}`,
    {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    }
  );
  
  if (!response.ok) {
    throw new Error('Failed to get recommendations');
  }
  
  return response.json();
};

// Usage
try {
  const recommendations = await getPersonalizedStartups(userToken, {
    limit: 20,
    category: 'AI',
    stage: 'Series A'
  });
  
  console.log('Method:', recommendations.method_used); // "two_tower" or "content_based"
  console.log('Interactions:', recommendations.interaction_count);
  console.log('Startups:', recommendations.startups);
} catch (error) {
  console.error('Error:', error);
}
```

---

## 🎯 Summary

### ✅ What's Working Now:

1. **Django has proxy endpoints** that route to Flask
2. **Two-Tower model is integrated** via Flask endpoints
3. **Automatic user role detection** routes to correct Flask endpoint
4. **Authentication is handled** by Django before proxying
5. **All query parameters** are forwarded to Flask
6. **Graceful error handling** if Flask is unavailable
7. **User context is added** to responses

### 🔥 Your Two-Tower Model is Now Live!

- Warm users (5+ interactions) get **Two-Tower recommendations**
- Cold users get **Content-Based recommendations**
- All through clean Django endpoints
- Frontend just calls Django - no need to know about Flask!

---

## 🧪 Next Steps

1. **Test the endpoints** with Postman or curl
2. **Update frontend** to call new Django endpoints
3. **Monitor logs** to see Two-Tower model in action
4. **Collect feedback** on recommendation quality
5. **Train with more data** to improve Two-Tower performance

---

## 🎉 Congratulations!

Your complete recommendation pipeline is now live:

```
User Interactions → Database → Two-Tower Model → Personalized Recommendations
```

The model automatically switches between Two-Tower (warm users) and Content-Based (cold users) for optimal recommendations! 🚀

