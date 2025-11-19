# Quick Model Verification Guide

## Step 1: Run Verification Script

```bash
cd recommendation_service
python verify_models.py
```

This checks:
- ✓ Model files exist
- ✓ Models loaded in memory  
- ✓ Test inference works

## Step 2: Check Flask Startup Logs

When you start Flask (`python app.py`), look for:

```
✓ Two-Tower Reverse model loaded successfully!
✓ ALS Reverse (SVD) model loaded successfully!
✓ Ranker Reverse model loaded successfully!
```

## Step 3: Test API Endpoint

```bash
# Get a startup ID from your database first
STARTUP_ID="<your-startup-uuid>"

# Test developer recommendations
curl "http://localhost:5000/api/recommendations/developers/for-startup/$STARTUP_ID?limit=5"

# Check response for:
# - "method_used": "two_tower_reverse" or "als_reverse" or "content_based"
# - "model_version": shows which model version was used
# - "reranked": true (if ranker was applied)
```

## Step 4: Check Response Fields

The API response includes these fields to verify model usage:

```json
{
  "developers": [...],
  "method_used": "two_tower_reverse",  // ← This tells you which model was used
  "model_version": "two_tower_reverse_v1.0",
  "reranked": true,  // ← Ranker was applied
  "total": 5,
  ...
}
```

## Model Routing Logic

The system uses smart routing:

- **Cold Start** (< 5 interactions): `content_based`
- **Warm** (5-19 interactions): `two_tower_reverse` → `als_reverse` → `content_based`
- **Hot** (20+ interactions): `two_tower_reverse` → `als_reverse` → `content_based`

Check Flask logs to see routing decisions:
```
Startup {id} has X unique user interactions
-> Using Two-Tower Reverse or ALS Reverse (warm startup: X interactions)
```

## Troubleshooting

**If models not loading:**
1. Check model files exist: `ls models/two_tower_reverse_v1*`
2. Check Flask logs for errors
3. Verify file paths match in `app.py`

**If getting "content_based":**
- Startup needs >= 5 interactions for ML models
- Check interaction count in logs
- Models might have failed to load (check startup logs)

