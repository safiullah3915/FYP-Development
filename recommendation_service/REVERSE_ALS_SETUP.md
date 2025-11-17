# Reverse ALS Setup and Testing Guide

## Overview

This guide explains how to set up and test the bi-directional ALS recommendation system that now supports:
- **User → Startup** recommendations (developers/investors finding startups)
- **Startup → User** recommendations (founders finding developers/investors)

## Setup Steps

### Step 1: Generate Dataset with Reverse Matrix

```bash
cd backend
python manage.py generate_als_dataset
```

**Expected Output:**
```
=== ALS Dataset Generation ===
Loading interactions from database...
Loaded X interactions
Creating user and item mappings...
Users: Y, Startups: Z
Building sparse interaction matrix...
Matrix shape: (Y, Z), Non-zero entries: X
Building reverse interaction matrix (Startups × Users)...
Reverse matrix shape: (Z, Y), Non-zero entries: X

Forward Matrix (Users × Startups):
  Matrix: ../recommendation_service/data/als_interactions.npz
  User mapping: ../recommendation_service/data/als_user_mapping.json
  Item mapping: ../recommendation_service/data/als_item_mapping.json

Reverse Matrix (Startups × Users):
  Matrix: ../recommendation_service/data/als_interactions_reverse.npz
  User mapping: ../recommendation_service/data/als_reverse_user_mapping.json
  Item mapping: ../recommendation_service/data/als_reverse_item_mapping.json
```

### Step 2: Train Forward ALS Model (User → Startup)

```bash
cd recommendation_service
python train_als.py
```

**Expected Output (abridged):**
```
SVD COLLABORATIVE FILTERING TRAINING
Loading data from data/als_interactions.npz...
Training completed in X.XX seconds
=== Evaluating Model (diagnostic) ===
  Explained variance ratio: 0.34
  Sampled reconstruction MSE: 0.000123
Saved user factors: models/als_v1_user_factors.npy
Saved item factors: models/als_v1_item_factors.npy
Saved config: models/als_v1_config.json
```

### Step 3: Train Reverse ALS Model (Startup → User)

```bash
python train_als_reverse.py
```

**Expected Output:**
```
SVD REVERSE COLLABORATIVE FILTERING TRAINING
(Startup → User Recommendations)
Loading REVERSE data from data/als_interactions_reverse.npz...
Training completed in X.XX seconds
=== Evaluating Reverse Model (diagnostic) ===
  Explained variance ratio: 0.29
  Sampled reconstruction MSE: 0.000210
Saved startup factors: models/als_reverse_v1_user_factors.npy
Saved user factors: models/als_reverse_v1_item_factors.npy
Saved config: models/als_reverse_v1_config.json
```

### Step 4: Verify Model Files

```bash
ls -lh models/als*
```

**Expected Files:**
```
als_v1_user_factors.npy
als_v1_item_factors.npy
als_v1_user_mapping.json
als_v1_item_mapping.json
als_v1_config.json

als_reverse_v1_user_factors.npy
als_reverse_v1_item_factors.npy
als_reverse_v1_user_mapping.json
als_reverse_v1_item_mapping.json
als_reverse_v1_config.json
```

### Step 5: Start Flask Service

```bash
python app.py
```

**Expected Startup Logs:**
```
✓ ALS Forward model loaded successfully!
✓ ALS Reverse model loaded successfully!
  -> Will be used for Founder → Developer/Investor recommendations
Starting Flask Recommendation Service on 0.0.0.0:5001
```

## Testing

### Test 1: Developer → Startup (Forward ALS)

```bash
curl "http://localhost:5001/api/recommendations/startups/for-developer/<USER_ID>?limit=10"
```

**Expected Response:**
```json
{
  "startups": [...],
  "total": 10,
  "method_used": "als",  // or "content_based" for cold-start users
  "interaction_count": 15
}
```

### Test 2: Investor → Startup (Forward ALS)

```bash
curl "http://localhost:5001/api/recommendations/startups/for-investor/<USER_ID>?limit=10"
```

**Expected Response:**
```json
{
  "startups": [...],
  "total": 10,
  "method_used": "als",  // or "content_based" for cold-start users
  "interaction_count": 12
}
```

### Test 3: Founder → Developer (Reverse ALS)

```bash
curl "http://localhost:5001/api/recommendations/developers/for-startup/<STARTUP_ID>?limit=10"
```

**Expected Response:**
```json
{
  "users": [...],
  "total": 10,
  "method_used": "als_reverse",  // or "content_based" for cold-start startups
  "startup_id": "<STARTUP_ID>"
}
```

### Test 4: Founder → Investor (Reverse ALS)

```bash
curl "http://localhost:5001/api/recommendations/investors/for-startup/<STARTUP_ID>?limit=10"
```

**Expected Response:**
```json
{
  "users": [...],
  "total": 10,
  "method_used": "als_reverse",  // or "content_based" for cold-start startups
  "startup_id": "<STARTUP_ID>"
}
```

## Routing Logic

### Forward Direction (User → Startup)

| User Interactions | Method Used | Model |
|------------------|-------------|-------|
| < 5 | content_based | None (profile matching) |
| 5-19 | als | als_v1_config.json + embeddings |
| 20+ | ensemble | als_v1_config.json + two_tower_v1.pth |

### Reverse Direction (Startup → User)

| Unique User Interactions | Method Used | Model |
|-------------------------|-------------|-------|
| < 5 | content_based | None (profile matching) |
| >= 5 | als_reverse | als_reverse_v1_config.json + embeddings |

## Troubleshooting

### Issue: "ALS Reverse model not found"

**Cause:** Model hasn't been trained yet

**Fix:**
```bash
cd backend
python manage.py generate_als_dataset
cd ../recommendation_service
python train_als_reverse.py
```

### Issue: "Startup not in ALS Reverse model"

**Cause:** Startup has no interactions in training data

**Fix:** This is expected behavior. The system will automatically fall back to content-based recommendations.

### Issue: "No interactions found" during dataset generation

**Cause:** Empty UserInteraction table

**Fix:** 
1. Ensure users are interacting with startups (views, likes, applies, etc.)
2. Check that interactions are being tracked in the `user_interactions` table
3. Lower min-interactions threshold: `python manage.py generate_als_dataset --min-interactions 1`

### Issue: Poor reverse recommendations quality

**Solutions:**
1. Collect more interaction data (aim for 1000+ interactions)
2. Ensure diverse user interactions across different startups
3. Retrain model weekly as more data accumulates

## Performance Metrics

**Good Performance Indicators:**
- Forward ALS explained variance ≥ 0.30
- Reverse ALS explained variance ≥ 0.25
- API response time < 200ms

**If metrics are low:**
1. Collect more interaction data
2. Tune hyperparameters (factors, iterations)
3. Verify interaction weights are appropriate

## Integration with Django Backend

The Django backend proxies recommendation requests to Flask:

```python
# Developer seeing startups (forward)
GET /api/recommendations/personalized/startups?type=collaboration

# Founder seeing developers (reverse)  
GET /api/recommendations/personalized/developers/<startup_id>

# Founder seeing investors (reverse)
GET /api/recommendations/personalized/investors/<startup_id>
```

All endpoints automatically:
1. Check interaction count
2. Route to appropriate model (ALS forward/reverse or content-based)
3. Apply business rules and diversity
4. Return formatted recommendations

## Retraining Schedule

**Recommended:**
- Weekly retraining for active platforms
- Monthly for moderate activity
- After significant data growth (2x interactions)

**Quick Retrain:**
```bash
cd backend && python manage.py generate_als_dataset
cd ../recommendation_service
python train_als.py && python train_als_reverse.py
# Restart Flask service
```

## Summary

✅ **Forward ALS** handles User → Startup (developers/investors finding startups)
✅ **Reverse ALS** handles Startup → User (founders finding developers/investors)
✅ **Content-Based** handles all cold-start scenarios
✅ **No new tables** required - reuses existing UserInteraction data
✅ **Automatic routing** based on interaction counts
✅ **Graceful fallbacks** for all edge cases

The system is now fully bi-directional and uses collaborative filtering for ALL use cases!

