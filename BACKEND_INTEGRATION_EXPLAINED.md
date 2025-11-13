# Django Backend ↔ Flask Recommendation Service Integration

## 🔍 Current Status: MINIMAL INTEGRATION

### What's Connected NOW:

**Django Backend** only calls Flask for **ONE endpoint**:

```
Django: GET /api/recommendations/trending/startups
   ↓
Flask: GET http://localhost:5000/api/recommendations/trending/startups
```

That's it! Just trending startups (non-personalized).

---

## ❌ What's NOT Connected (Yet):

Your Django backend does **NOT** call Flask for personalized recommendations. 

### These Flask endpoints exist but are NOT used by Django:
- ❌ `/api/recommendations/startups/for-developer/<user_id>`
- ❌ `/api/recommendations/startups/for-investor/<user_id>`
- ❌ `/api/recommendations/developers/for-startup/<startup_id>`
- ❌ `/api/recommendations/investors/for-startup/<startup_id>`

---

## 📊 How Django Backend Currently Works:

### 1. User Interactions (Likes, Views, etc.)
**Django handles these directly:**
```
Frontend → Django POST /api/startups/<id>/like
         → Saves to UserInteraction table
         ✓ No Flask involvement
```

**Endpoints in Django:**
- `POST /api/startups/<id>/like` - Like a startup
- `POST /api/startups/<id>/unlike` - Unlike
- `POST /api/startups/<id>/dislike` - Dislike
- `POST /api/startups/<id>/undislike` - Remove dislike
- `GET /api/startups/<id>/interaction-status` - Check status

### 2. Onboarding Preferences
**Django handles:**
```
Frontend → Django GET/PUT /api/onboarding/preferences
         → Saves to UserOnboardingPreferences table
         ✓ No Flask involvement
```

### 3. Trending Startups (Only Flask Integration)
**Django proxies to Flask:**
```
Frontend → Django GET /api/recommendations/trending/startups
         → Django calls Flask GET http://localhost:5000/api/recommendations/trending/startups
         → Returns Flask response
```

---

## 🔌 What You Need to Add to Django:

To use your Two-Tower model from Django, add these new endpoints:

### Option 1: Create New Django Endpoints (Recommended)

Add to `backend/api/views.py`:

```python
@api_view(['GET'])
def get_personalized_recommendations(request):
    """Get personalized recommendations from Flask"""
    import requests
    
    user = get_session_user(request)
    if not user:
        return Response({'error': 'Not authenticated'}, status=401)
    
    # Get params
    limit = request.query_params.get('limit', 10)
    startup_type = request.query_params.get('type', None)
    
    # Determine endpoint based on user role
    if user.role == 'student':
        flask_url = f'http://localhost:5000/api/recommendations/startups/for-developer/{user.id}'
    elif user.role == 'investor':
        flask_url = f'http://localhost:5000/api/recommendations/startups/for-investor/{user.id}'
    else:
        return Response({'error': 'Role not supported'}, status=400)
    
    # Call Flask
    try:
        params = {'limit': limit}
        if startup_type:
            params['type'] = startup_type
        
        response = requests.get(flask_url, params=params, timeout=5)
        response.raise_for_status()
        return Response(response.json(), status=200)
    except:
        return Response({'error': 'Recommendation service unavailable'}, status=503)
```

Add to `backend/api/urls.py`:
```python
path('api/recommendations/personalized', views.get_personalized_recommendations, name='personalized_recommendations'),
```

### Option 2: Frontend Calls Flask Directly (Current Setup)

Your frontend can call Flask directly:
```javascript
// Instead of calling Django
const response = await fetch(
  `http://localhost:5000/api/recommendations/startups/for-developer/${userId}?limit=10`
);
```

---

## 🎯 Recommended Architecture

### Current (Minimal):
```
Frontend
   ↓
Django (trending only)
   ↓
Flask Recommendations
```

### Recommended (Full Integration):
```
Frontend
   ↓
Django (all endpoints)
   ↓
Flask Recommendations (Two-Tower + Content-Based)
```

---

## 📝 Summary for Your Question:

### Does Django use your recommendation endpoints?

**Short Answer:** NO (except trending)

**Current Integration:**
- ✅ Django → Flask for **trending startups only**
- ❌ Django does NOT call Flask for **personalized recommendations**
- ❌ Django does NOT use your **Two-Tower model**

### Which endpoints does Django expose?

**Django Endpoints (in `api/urls.py`):**
1. `GET /api/onboarding/preferences` - User preferences
2. `POST /api/startups/<id>/like` - Like startup
3. `POST /api/startups/<id>/unlike` - Unlike
4. `POST /api/startups/<id>/dislike` - Dislike
5. `GET /api/startups/<id>/interaction-status` - Check status
6. `GET /api/recommendations/trending/startups` - Trending (calls Flask)
7. `POST /api/recommendations/session` - Store recommendation session

**Flask Endpoints (in `app.py`):**
1. `GET /api/recommendations/startups/for-developer/<user_id>` ← **Has Two-Tower!**
2. `GET /api/recommendations/startups/for-investor/<user_id>` ← **Has Two-Tower!**
3. `GET /api/recommendations/developers/for-startup/<startup_id>`
4. `GET /api/recommendations/investors/for-startup/<startup_id>`
5. `GET /api/recommendations/trending/startups`

---

## 🚀 To Use Your Two-Tower Model:

### Option A: Add Django Proxy Endpoint
Add endpoint to Django that calls Flask (see code above)

### Option B: Frontend Calls Flask Directly
```javascript
// In your React frontend
const getRecommendations = async (userId) => {
  const response = await fetch(
    `http://localhost:5000/api/recommendations/startups/for-developer/${userId}?limit=10`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    }
  );
  return response.json();
};
```

### Option C: Add Middleware to Django
Create a middleware that automatically proxies `/api/recommendations/*` to Flask

---

## 💡 Bottom Line:

**Your Two-Tower model works in Flask, but Django doesn't call it yet!**

You need to either:
1. Add Django endpoints that proxy to Flask
2. Have frontend call Flask directly
3. Add middleware to route recommendation requests to Flask

**Easiest:** Frontend calls Flask directly for recommendations while keeping Django for everything else (auth, CRUD, interactions).

