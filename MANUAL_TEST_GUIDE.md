# Manual Testing Guide - Complete Recommendation System

## Overview

This guide walks you through manually testing the entire recommendation system from frontend to backend.

---

## Prerequisites

✅ Both services must be running:
1. **Django Backend** - Port 8000
2. **Flask Recommendation Service** - Port 5000

---

## Test Flow Diagram

```
Frontend Request
       |
       v
Django Backend (:8000)
       |
       | (HTTP Proxy)
       v
Flask Service (:5000)
       |
       v
   Router (decides model)
       |
       +---> Cold Start (<5) --> Content-Based
       |
       +---> Warm (5-19) ------> ALS
       |
       +---> Hot (20+) --------> Ensemble
       |
       v
   Response (JSON)
       |
       v
Django Backend
       |
       v
  Frontend Display
```

---

## Test 1: Check Services Are Running

### 1.1 Test Django Health

**Browser/Curl:**
```bash
curl http://localhost:8000/api/
```

**Expected:** Some JSON response or 404 (means Django is running)

**If Failed:**
```bash
cd backend
python manage.py runserver
```

---

### 1.2 Test Flask Health

**Browser:**
Open: http://localhost:5000/health

**Curl:**
```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Flask Recommendation Service",
  "database": "connected",
  "models_loaded": {
    "content_based": true,
    "two_tower": true,
    "als": true,
    "ensemble": true
  },
  "timestamp": "..."
}
```

**If Models Not Loaded:**
```bash
# Train models first
train_all_models.bat

# Restart Flask
cd recommendation_service
python app.py
```

---

## Test 2: Direct Flask Endpoints

These test Flask recommendation service directly (bypassing Django).

### 2.1 Developer Startup Recommendations

**URL:**
```
GET http://localhost:5000/api/recommendations/startups/for-developer/{USER_ID}?limit=5
```

**Curl:**
```bash
curl "http://localhost:5000/api/recommendations/startups/for-developer/test-user-123?limit=5"
```

**Expected Response:**
```json
{
  "startups": [
    {
      "id": "...",
      "title": "Startup Name",
      "description": "...",
      "type": "marketplace",
      "category": "...",
      "score": 0.85,
      "match_reasons": ["..."]
    }
  ],
  "total": 5,
  "method_used": "als",  // or "content_based", "two_tower", "ensemble"
  "interaction_count": 10,
  "model_version": "als_v1.0"
}
```

**Key Fields to Check:**
- ✅ `startups` array is present
- ✅ `method_used` shows which model was used
- ✅ `interaction_count` shows user's interaction history
- ✅ `total` matches number of results

---

### 2.2 Investor Startup Recommendations

**URL:**
```
GET http://localhost:5000/api/recommendations/startups/for-investor/{USER_ID}?limit=5
```

**Curl:**
```bash
curl "http://localhost:5000/api/recommendations/startups/for-investor/test-investor-123?limit=5"
```

**Expected:** Similar structure to above, but only marketplace startups.

---

### 2.3 Trending Startups (No Auth Required)

**URL:**
```
GET http://localhost:5000/api/recommendations/trending/startups?limit=10
```

**Curl:**
```bash
curl "http://localhost:5000/api/recommendations/trending/startups?limit=10"
```

**Expected:**
```json
{
  "startups": [...],
  "total": 10,
  "method_used": "trending"
}
```

---

## Test 3: Django Proxy Endpoints

These test the complete flow: Frontend → Django → Flask → Response

### 3.1 Personalized Startup Recommendations

**URL:**
```
GET http://localhost:8000/api/recommendations/personalized/startups?limit=5
```

**Note:** This endpoint requires authentication. You need to:
1. Login to get auth token
2. Include token in request header

**Curl (with token):**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     "http://localhost:8000/api/recommendations/personalized/startups?limit=5"
```

**Without Auth:**
```bash
curl "http://localhost:8000/api/recommendations/personalized/startups?limit=5"
```

**Expected (without auth):**
```json
{
  "error": "Authentication required"
}
```

**Expected (with auth):**
```json
{
  "startups": [...],
  "total": 5,
  "method_used": "als",
  "interaction_count": 12
}
```

---

### 3.2 Get Auth Token (For Testing)

**Step 1: Login**

```bash
curl -X POST http://localhost:8000/api/login/ \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"your@email.com\", \"password\":\"yourpassword\"}"
```

**Step 2: Extract Token**

Response will contain:
```json
{
  "access": "eyJ...",
  "refresh": "eyJ...",
  "user": {...}
}
```

**Step 3: Use Token**

```bash
curl -H "Authorization: Bearer eyJ..." \
     "http://localhost:8000/api/recommendations/personalized/startups?limit=5"
```

---

## Test 4: Smart Routing Logic

Test that different users get routed to different models based on interaction count.

### 4.1 Cold Start User (< 5 interactions)

**Expected Model:** `content_based`

**How to Test:**
1. Create a new user account
2. Don't interact with any startups
3. Request recommendations
4. Check `method_used: "content_based"`

---

### 4.2 Warm User (5-19 interactions)

**Expected Model:** `als`

**How to Test:**
1. Use existing user with some interactions
2. Request recommendations
3. Check `method_used: "als"`
4. Check `interaction_count: 5-19`

---

### 4.3 Hot User (20+ interactions)

**Expected Model:** `ensemble`

**How to Test:**
1. Use user with many interactions
2. Request recommendations
3. Check `method_used: "ensemble"`
4. Check `interaction_count: 20+`

---

## Test 5: Verify Model Quality

### Check Response Structure

For any recommendation response, verify:

```json
{
  "startups": [                    // ✅ Array of startups
    {
      "id": "uuid",                // ✅ UUID present
      "title": "string",           // ✅ Title present
      "description": "string",     // ✅ Description present
      "type": "marketplace",       // ✅ Type present
      "category": "string",        // ✅ Category present
      "score": 0.85,               // ✅ Relevance score
      "match_reasons": [           // ✅ Explanation
        "Reason 1",
        "Reason 2"
      ]
    }
  ],
  "total": 10,                     // ✅ Total count
  "method_used": "als",            // ✅ Model used
  "interaction_count": 12,         // ✅ User interaction history
  "model_version": "als_v1.0"      // ✅ Model version
}
```

---

## Test 6: Error Handling

### 6.1 Invalid User ID

```bash
curl "http://localhost:5000/api/recommendations/startups/for-developer/invalid-uuid?limit=5"
```

**Expected:** 400 Bad Request or empty results with fallback

---

### 6.2 Flask Service Down

1. Stop Flask service (Ctrl+C)
2. Try Django proxy endpoint
3. **Expected:** Django returns error or fallback

---

### 6.3 Models Not Loaded

1. Delete model files from `recommendation_service/models/`
2. Restart Flask
3. **Expected:** Flask logs warnings, falls back to content-based

---

## Test 7: End-to-End Frontend Test

### Using Browser (If Frontend Exists)

1. **Login** to the platform
2. **Navigate** to startup discovery page
3. **Observe** recommended startups
4. **Check** browser DevTools → Network tab
5. **Verify** request goes to `/api/recommendations/personalized/startups`
6. **Inspect** response includes `method_used` and `interaction_count`

---

## Common Issues & Solutions

### Issue 1: "Connection refused" on port 5000

**Solution:** Flask not running. Start it:
```bash
cd recommendation_service
python app.py
```

---

### Issue 2: "Connection refused" on port 8000

**Solution:** Django not running. Start it:
```bash
cd backend
python manage.py runserver
```

---

### Issue 3: Models not loading in Flask

**Solution:** Train models first:
```bash
train_all_models.bat
```

---

### Issue 4: Empty recommendations

**Possible Causes:**
1. Empty database (no startups)
2. No interaction data for user
3. Filters too restrictive

**Solution:** Add test data or check database.

---

## Success Criteria

✅ Django running and accessible
✅ Flask running and accessible
✅ Flask health endpoint shows all models loaded
✅ Direct Flask endpoints return recommendations
✅ Django proxy endpoints forward correctly
✅ Smart routing works (cold/warm/hot users)
✅ Response structure is correct
✅ No errors in server logs

---

## Automated Test

Instead of manual testing, run:

```bash
python test_complete_recommendation_flow.py
```

This automatically tests all the above scenarios.

---

## Next Steps After Testing

1. **If tests pass:** System is working! ✅
2. **If models need training:** Run `train_all_models.bat`
3. **If services not running:** Follow START_SERVICES.md
4. **If errors persist:** Check logs:
   - Django: `backend/django.log`
   - Flask: `recommendation_service/logs/app.log`

---

## Quick Reference

| Test | URL | Expected |
|------|-----|----------|
| Django Health | `http://localhost:8000/api/` | JSON or 404 |
| Flask Health | `http://localhost:5000/health` | Status: healthy |
| Direct Rec | `http://localhost:5000/api/recommendations/startups/for-developer/USER_ID` | Startups array |
| Proxy Rec | `http://localhost:8000/api/recommendations/personalized/startups` | Requires auth |
| Trending | `http://localhost:5000/api/recommendations/trending/startups` | Public, no auth |

---

## Summary

Your recommendation system has **3 layers**:

1. **Frontend** - User interface (browser)
2. **Django** - Authentication, proxy, database
3. **Flask** - ML models, recommendations

All communication flows: **Frontend ← → Django ← → Flask ← → Models**

Test each layer independently, then test the complete flow!

