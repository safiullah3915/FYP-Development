# Complete Recommendation Feedback Collection & Training Flow

## Overview
This document explains how recommendation feedback is collected, stored, and used to train models to distinguish good vs bad recommendations.

---

## Part 1: Recommendation Display & Session Tracking

### Step 1: User Requests Recommendations
**Frontend:** User navigates to dashboard/recommendations page
**API Call:** `GET /api/recommendations/personalized/startups`
**Flow:**
- Django authenticates user
- Django proxies request to Flask service: `/api/recommendations/startups/for-developer/{user_id}`
- Flask generates recommendations using Two-Tower/ALS model
- Flask returns: `{startup_ids: [...], scores: {...}, method_used: 'two_tower', ...}`
- Django hydrates startup objects and adds rank/score to response
- **Response includes:** Each startup with `rank` (position 1-N), `score` (model prediction)

### Step 2: Store Recommendation Session
**Frontend:** After receiving recommendations, call `storeRecommendationSession()`
**API Call:** `POST /api/recommendations/session`
**Payload:**
```javascript
{
  recommendation_session_id: "uuid-here",
  use_case: "developer_startup",
  method: "two_tower",
  model_version: "two_tower_v1.0",
  recommendations: [
    {startup_id: "uuid1", rank: 1, score: 0.85, method: "two_tower"},
    {startup_id: "uuid2", rank: 2, score: 0.78, method: "two_tower"},
    // ... up to rank N
  ]
}
```
**Backend:** Stores in `recommendation_sessions` table with `recommendations_shown` JSON field

**Current Status:** This step is implemented but frontend may not be calling it yet. Need to add session tracking when displaying recommendations.

---

## Part 2: User Interaction & Feedback Collection

### Step 3: User Interacts with Recommended Startup
**Actions that create feedback:**
- **View:** User clicks on startup card → `GET /api/startups/{id}` → Auto-creates `view` interaction
- **Like:** User clicks like button → `POST /api/startups/{id}/like` → Creates `like` interaction
- **Dislike:** User clicks dislike → `POST /api/startups/{id}/dislike` → Creates `dislike` interaction
- **Apply:** User submits application → `POST /api/collaborations/{id}/apply` → Creates `apply` interaction
- **Favorite:** Investor favorites → `POST /api/startups/{id}/favorite` → Creates `favorite` interaction
- **Interest:** Investor expresses interest → `POST /api/startups/{id}/interest` → Creates `interest` interaction

### Step 4: Link Interaction to Recommendation Session
**Frontend:** When user interacts, include recommendation context:
```javascript
// Example: Like a startup from recommendations
await recommendationAPI.likeStartup(startupId, {
  sessionId: currentSession.sessionId,
  rank: currentSession.recommendations.find(r => r.startup_id === startupId)?.rank
});
```

**Backend:** `InteractionService.create_interaction()` extracts context:
- Gets `recommendation_session_id` from request
- Looks up `RecommendationSession` to get rank/score
- Creates `UserInteraction` with:
  - `recommendation_session_id` (FK link)
  - `recommendation_source = 'recommendation'`
  - `recommendation_rank` (position shown)
  - `recommendation_score` (model prediction)
  - `recommendation_method` (two_tower/als/etc.)

**Result:** Every interaction knows if it came from a recommendation and which position it was shown at.

---

## Part 3: Distinguishing Good vs Bad Recommendations

### Good Recommendations (Positive Feedback)
**Identified by:** User interactions with `recommendation_source = 'recommendation'`
**Signals:**
- `like` (weight: 2.0) - User liked the recommendation
- `favorite` (weight: 2.5) - Investor favorited it
- `apply` (weight: 3.0) - User applied to position
- `interest` (weight: 3.5) - Investor expressed interest
- `view` (weight: 0.5) - User viewed details (weak positive)

**Training Label:** `label = 1` (positive example)

### Bad Recommendations (Negative Feedback)
**Identified by:** Two types of negatives:

**Type 1: Explicit Negative**
- `dislike` (weight: -1.0) - User explicitly disliked
- **Training Label:** `label = 0` (negative example)

**Type 2: Implicit Negative (Hard Negatives)**
- Startup was shown in recommendations (`recommendation_session.recommendations_shown`)
- User saw it (has `view` interaction OR was in session)
- BUT user did NOT like/favorite/apply/express interest
- Especially bad if: High `recommendation_score` but no engagement
- **Training Label:** `label = 0` (negative example)

**Why Hard Negatives Matter:**
- Model predicted high score (0.85) but user ignored it
- Teaches model: "This startup-user pair looked good but wasn't actually relevant"
- More informative than random negatives

---

## Part 4: Dataset Generation for Training

### Ranker Dataset (`generate_ranker_dataset.py`)

**Positive Samples:**
```python
# Load interactions with recommendation context
UserInteraction.objects.filter(
    interaction_type__in=['like', 'favorite', 'apply', 'interest'],
    recommendation_source='recommendation'  # Only from recommendations
).values(
    'user_id', 'startup_id', 
    'recommendation_rank', 'recommendation_score', 'recommendation_method'
)
```

**Negative Samples (Hard Negatives):**
```python
# For each user, find startups that were:
# 1. Shown in their recommendation sessions
# 2. Had high scores
# 3. But user did NOT interact positively

# Query RecommendationSession for user
sessions = RecommendationSession.objects.filter(user_id=user_id)
shown_startups = []
for session in sessions:
    for rec in session.recommendations_shown:
        if rec['score'] > 0.7:  # High score threshold
            shown_startups.append({
                'startup_id': rec['startup_id'],
                'rank': rec['rank'],
                'score': rec['score'],
                'method': rec['method']
            })

# Filter out startups user DID interact with
user_positive_startups = set(positive_interactions['startup_id'])
hard_negatives = [
    s for s in shown_startups 
    if s['startup_id'] not in user_positive_startups
]
```

**Dataset Columns:**
- `user_id`, `startup_id`
- `recommendation_rank` (position shown)
- `recommendation_score` (original model prediction)
- `original_score` (same as recommendation_score, for teacher signal)
- `exposure_weight` (1 / log2(rank + 1)) - Corrects for position bias
- `recommendation_method` (two_tower/als/etc.)
- `label` (1 = positive, 0 = negative)

### Two-Tower Dataset (`generate_two_tower_dataset.py`)

**Positive Samples:**
- All interactions (view, like, apply, etc.)
- Weighted by rank: `weight = base_weight * (1 / log2(rank + 1))`
- Deeper ranks get higher weight (stronger signal - user found it despite low position)

**Negative Samples:**
- Hard negatives from recommendation sessions (high score, no interaction)
- Random negatives (startups never shown to user)

**Dataset Columns:**
- `user_id`, `startup_id`
- `recommendation_rank`, `recommendation_score`
- `rank_weight` (exposure correction weight)
- `label` (0.0 to 1.0 based on interaction type)
- User features (embedding, preferences, skills)
- Startup features (embedding, category, tags, positions)

---

## Part 5: Model Training from Feedback

### Ranker Model Training

**Input Features:**
- `model_score` (from base recommender)
- `recency_score` (startup age)
- `popularity_score` (views/interactions)
- `diversity_score` (avoid clustering)
- `original_score` (recommendation_score - teacher signal)

**Training Process:**
1. **Pairwise Learning:** Compare positive vs negative pairs
2. **Exposure Bias Correction:** Weight loss by `exposure_weight`
   - Rank 1 interaction: weight = 1.0
   - Rank 5 interaction: weight = 1 / log2(6) ≈ 0.39
   - Rank 10 interaction: weight = 1 / log2(11) ≈ 0.29
3. **Teacher-Student Learning:** Model learns to adjust `original_score` based on actual feedback
   - If `original_score = 0.85` but user disliked → Model learns to reduce score
   - If `original_score = 0.60` but user applied → Model learns to increase score

**Loss Function:**
```python
loss = exposure_weight * pairwise_ranking_loss(
    predicted_score(positive) - predicted_score(negative)
)
```

### Two-Tower Model Training

**Training Process:**
1. **Contrastive Learning:** User embedding vs Startup embedding similarity
2. **Rank-Based Weighting:** Positive samples weighted by `rank_weight`
   - Interactions from rank 10 get higher weight than rank 1 (stronger signal)
3. **Hard Negative Mining:** Prioritize negatives with high scores but no interaction

**Loss Function:**
```python
loss = rank_weight * contrastive_loss(
    user_embedding, startup_embedding, label
)
```

### ALS Model Training

**Training Process:**
1. **Collaborative Filtering:** Uses interaction matrix (user × startup)
2. **Weighted by Interaction Type:** Apply interactions (weight 3.0) count more than views (0.5)
3. **Implicit Feedback:** All interactions treated as positive signals (no explicit negatives)

---

## Part 6: Complete Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER REQUESTS RECOMMENDATIONS                           │
│    GET /api/recommendations/personalized/startups          │
│    → Flask generates: [startup1(rank=1, score=0.85), ...] │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. STORE RECOMMENDATION SESSION                            │
│    POST /api/recommendations/session                       │
│    → Saves: {session_id, recommendations_shown: [...]}     │
│    → Table: recommendation_sessions                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. USER INTERACTS WITH RECOMMENDATION                       │
│    Like/Apply/Favorite/Dislike                             │
│    → Includes: recommendation_session_id, rank              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. STORE USER INTERACTION                                   │
│    InteractionService.create_interaction()                  │
│    → Links to RecommendationSession                         │
│    → Stores: rank, score, method, source='recommendation'   │
│    → Table: user_interactions                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. IDENTIFY GOOD VS BAD RECOMMENDATIONS                     │
│                                                              │
│    GOOD (Positive):                                         │
│    - User liked/applied/favorited                           │
│    - recommendation_source = 'recommendation'                │
│    - Label: 1                                               │
│                                                              │
│    BAD (Negative):                                          │
│    - User disliked (explicit)                                │
│    - OR: Shown in session but no positive interaction      │
│    - Especially: High score but ignored (hard negative)     │
│    - Label: 0                                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. GENERATE TRAINING DATASET                                │
│    python manage.py generate_ranker_dataset                 │
│    python manage.py generate_two_tower_dataset              │
│                                                              │
│    For each user:                                           │
│    - Positives: Interactions with recommendations            │
│    - Negatives: Shown but not interacted (hard negatives)  │
│    - Features: rank, score, method, exposure_weight          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. TRAIN MODELS                                             │
│    python train_ranker.py                                   │
│    python train_two_tower.py                                │
│    python train_als.py                                      │
│                                                              │
│    Models learn:                                            │
│    - Which recommendations led to engagement (good)         │
│    - Which recommendations were ignored (bad)              │
│    - How to adjust scores based on feedback                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. DEPLOY IMPROVED MODELS                                   │
│    → Better recommendations                                 │
│    → Higher engagement                                      │
│    → More accurate scores                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       └──→ Loop back to Step 1
```

---

## Key Insights

### How Models Learn Good vs Bad:

1. **Positive Examples:**
   - User saw startup at rank 5 with score 0.72
   - User applied (strong signal)
   - Model learns: "This user-startup pair is relevant despite lower rank"

2. **Hard Negative Examples:**
   - Model predicted score 0.88 (very high)
   - Startup shown at rank 2 (top position)
   - User saw it but did NOT interact
   - Model learns: "High score was wrong - this pair is NOT relevant"

3. **Exposure Bias Correction:**
   - Rank 1 interactions are common (users see top items)
   - Rank 10 interactions are rare (users scroll less)
   - But rank 10 interactions are STRONGER signals (user really wanted it)
   - Exposure weight: `1 / log2(rank + 1)` balances this

4. **Teacher Signal:**
   - Original model predicted score 0.75
   - User applied → Model learns to increase score
   - User disliked → Model learns to decrease score
   - New model predicts residual adjustment

---

## Current Implementation Status

✅ **Implemented:**
- Database schema with recommendation context fields
- InteractionService extracts rank/score from sessions
- Dataset generators include hard negative sampling
- Training scripts use exposure weights and teacher signals

⚠️ **Needs Frontend Integration:**
- Frontend should call `storeRecommendationSession()` when displaying recommendations
- Frontend should pass `recommendationContext` when user interacts (sessionId, rank)

---

## Next Steps to Complete Feedback Loop

1. **Update Frontend to Store Sessions:**
   - When displaying recommendations, call `storeRecommendationSession()`
   - Include rank and score for each recommendation

2. **Update Frontend to Pass Context:**
   - When user likes/applies, include `recommendationContext` with sessionId and rank

3. **Generate Training Data:**
   - Run dataset generation commands
   - Verify hard negatives are being created from recommendation sessions

4. **Train Models:**
   - Train ranker with new features
   - Train Two-Tower with rank-based weights
   - Train ALS with weighted interactions

5. **Evaluate:**
   - Use evaluation scripts to verify model improvements
   - Compare old vs new model performance

