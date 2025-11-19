# Complete Recommendation Session Flow: Flask → Django → Frontend

## Overview
This document explains the complete flow of how recommendation sessions are created, stored, and used for feedback tracking.

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: FRONTEND REQUESTS RECOMMENDATIONS                      │
│                                                                 │
│ Frontend: Marketplace.jsx or Collaboration.jsx                  │
│ API Call: GET /api/recommendations/personalized/startups       │
│ Params: {limit: 20, offset: 0, type: 'collaboration', ...}   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DJANGO PROXIES TO FLASK SERVICE                         │
│                                                                 │
│ Django View: get_personalized_startup_recommendations()         │
│ Location: backend/api/views.py (line ~2600)                     │
│                                                                 │
│ 1. Authenticates user                                          │
│ 2. Determines use case (investor_startup or developer_startup)│
│ 3. Calls Flask service:                                        │
│    GET http://localhost:5000/api/recommendations/startups/   │
│    for-developer/{user_id}                                     │
│    OR                                                           │
│    GET http://localhost:5000/api/recommendations/startups/     │
│    for-investor/{user_id}                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: FLASK GENERATES RECOMMENDATIONS                         │
│                                                                 │
│ Flask Service: recommendation_service/                          │
│ Endpoint: /api/recommendations/startups/for-developer/{user_id}│
│                                                                 │
│ 1. Loads trained models (Two-Tower, ALS, Ranker)               │
│ 2. Generates recommendations with scores                       │
│ 3. Returns JSON:                                               │
│    {                                                           │
│      startup_ids: [uuid1, uuid2, ...],                        │
│      scores: {uuid1: 0.85, uuid2: 0.78, ...},                 │
│      method_used: 'two_tower',                                 │
│      model_version: 'two_tower_v1.0',                         │
│      match_reasons: {...},                                     │
│      total: 20                                                 │
│    }                                                           │
│                                                                 │
│ NOTE: Flask does NOT generate session_id                       │
│       Frontend generates session_id and sends to Django        │
│       Django is responsible for storing the session            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: DJANGO HYDRATES STARTUP DATA                           │
│                                                                 │
│ Django View: get_personalized_startup_recommendations()         │
│                                                                 │
│ 1. Receives Flask response with startup_ids                     │
│ 2. Queries Django database for full Startup objects             │
│ 3. Adds positions, categories, descriptions                    │
│ 4. Filters by status='active'                                  │
│ 5. Builds recommendations_payload with rank & score:            │
│    [                                                            │
│      {startup_id: uuid1, rank: 1, score: 0.85, method: ...}, │
│      {startup_id: uuid2, rank: 2, score: 0.78, method: ...}, │
│      ...                                                        │
│    ]                                                            │
│ 6. Returns to frontend:                                        │
│    {                                                           │
│      startups: [...full startup objects...],                   │
│      scores: {...},                                            │
│      method_used: 'two_tower',                                 │
│      model_version: 'two_tower_v1.0',                         │
│      recommendations: [{startup_id, rank, score, method}],   │
│      total: 20                                                 │
│    }                                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: FRONTEND STORES RECOMMENDATION SESSION                  │
│                                                                 │
│ Frontend: Marketplace.jsx or Collaboration.jsx                  │
│ Hook: useRecommendationContext.storeSession()                   │
│                                                                 │
│ 1. Extracts recommendations from Django response               │
│ 2. Generates session UUID (using crypto.randomUUID())          │
│ 3. Calls: POST /api/recommendations/session                    │
│    Payload:                                                    │
│    {                                                           │
│      recommendation_session_id: "uuid-generated-by-frontend",  │
│      use_case: "investor_startup" or "developer_startup",      │
│      method: "two_tower",                                      │
│      model_version: "two_tower_v1.0",                          │
│      recommendations: [                                        │
│        {startup_id: uuid1, rank: 1, score: 0.85, method: ...},│
│        {startup_id: uuid2, rank: 2, score: 0.78, method: ...} │
│      ]                                                         │
│    }                                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: DJANGO STORES SESSION IN DATABASE                      │
│                                                                 │
│ Django View: store_recommendation_session()                     │
│ Location: backend/api/views.py (line ~2468)                    │
│                                                                 │
│ 1. Validates required fields                                   │
│ 2. Normalizes recommendations data                             │
│ 3. Creates/updates RecommendationSession:                      │
│    RecommendationSession.objects.update_or_create(              │
│      id=session_id,                                            │
│      defaults={                                                │
│        user_id: user,                                          │
│        use_case: 'investor_startup',                           │
│        recommendation_method: 'two_tower',                     │
│        model_version: 'two_tower_v1.0',                        │
│        recommendations_shown: [{rank, score, ...}],           │
│        expires_at: now + 24 hours                             │
│      }                                                         │
│    )                                                           │
│ 4. Stores in database table: recommendation_sessions           │
│ 5. Returns: {message: "Session stored", session_id: "..."}     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: FRONTEND TRACKS SESSION                                │
│                                                                 │
│ Frontend Hook: useRecommendationContext                         │
│                                                                 │
│ 1. Stores session in React state:                              │
│    currentSession = {                                          │
│      sessionId: "uuid",                                        │
│      recommendations: [{startup_id, rank, score}],            │
│      useCase: "investor_startup",                              │
│      method: "two_tower"                                       │
│    }                                                           │
│ 2. Provides getRecommendationContext(startupId) function       │
│    Returns: {sessionId, rank, score, method} or null           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: USER INTERACTS WITH RECOMMENDATION                      │
│                                                                 │
│ Frontend: StartupDetails.jsx                                   │
│                                                                 │
│ User clicks Like/Dislike/Favorite/Interest                    │
│                                                                 │
│ 1. Calls getRecommendationContext(startupId)                    │
│ 2. If found, passes context to API:                           │
│    POST /api/startups/{id}/like                                │
│    Body: {                                                     │
│      recommendation_session_id: "uuid",                        │
│      recommendation_rank: 1                                    │
│    }                                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: DJANGO LINKS INTERACTION TO SESSION                     │
│                                                                 │
│ Django View: like_startup() / dislike_startup() / etc.          │
│ Service: InteractionService.create_interaction()                │
│                                                                 │
│ 1. Extracts recommendation_session_id from request              │
│ 2. Looks up RecommendationSession                             │
│ 3. Gets rank & score from session.recommendations_shown        │
│ 4. Creates UserInteraction with:                              │
│    - recommendation_session (FK link)                           │
│    - recommendation_source = 'recommendation'                   │
│    - recommendation_rank = 1                                    │
│    - recommendation_score = 0.85                               │
│    - recommendation_method = 'two_tower'                       │
│ 5. Stores in database table: user_interactions                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components Breakdown

### 1. **Flask Service** (recommendation_service/)
- **Role**: Generates recommendations using ML models
- **Does NOT**: Store sessions (only generates session_id)
- **Returns**: startup_ids, scores, method_used, model_version

### 2. **Django Backend** (backend/api/)
- **Role**: Proxy, hydration, and session storage
- **Endpoints**:
  - `GET /api/recommendations/personalized/startups` → Proxies to Flask, hydrates data
  - `POST /api/recommendations/session` → Stores session in database
  - `POST /api/startups/{id}/like` → Creates interaction with session context

### 3. **Frontend** (frontend/src/)
- **Role**: UI, session tracking, context passing
- **Components**:
  - `Marketplace.jsx` / `Collaboration.jsx` → Store sessions when displaying recommendations
  - `StartupDetails.jsx` → Pass context when user interacts
  - `useRecommendationContext` hook → Manages session state

---

## Data Flow Summary

1. **Flask generates** recommendations → Returns IDs + scores
2. **Django hydrates** startup objects → Adds full data
3. **Frontend stores** session → POST to Django
4. **Django saves** session → Database table
5. **User interacts** → Frontend passes session context
6. **Django links** interaction → Stores rank/score/method

---

## Important Notes

- **Session ID**: 
  - **Frontend generates** session_id using `crypto.randomUUID()`
  - Flask does NOT generate session_id (removed from Flask responses)
  - Django stores whatever session_id frontend sends in POST request
- **Session Storage**: Django stores it, Flask doesn't need to know about it
- **Session Expiry**: 24 hours (configurable in Django)
- **Context Passing**: Frontend must pass sessionId + rank when interacting

---

## Current Status

✅ **Implemented**:
- Django proxies to Flask
- Django stores sessions
- Frontend stores sessions
- Frontend passes context

✅ **Current Implementation**: 
- Frontend generates session_id and sends it to Django
- Flask does NOT generate session_id (removed from Flask)
- Django stores the frontend-generated session_id
- All interactions link to sessions using the frontend-generated session_id

