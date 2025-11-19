# Testing Reverse Models in Inference

This guide shows how to verify that your trained reverse models are being used in inference.

## Quick Verification

### 1. Check Model Files Exist

```bash
cd recommendation_service
python verify_models.py
```

This will show:
- ✓/✗ Model files exist
- ✓/✗ Models loaded in memory
- Test inference with sample data

### 2. Check Flask App Startup Logs

When you start the Flask app, look for these log messages:

```
✓ Two-Tower Reverse model loaded successfully!
  -> Will be used for Startup → Developer/Investor recommendations
✓ ALS Reverse (SVD) model loaded successfully!
  -> Will be used for Startup → Developer/Investor recommendations
✓ Ranker Reverse model loaded successfully!
```

If models fail to load, you'll see warnings:
```
⚠ Two-Tower Reverse model not found at models/two_tower_reverse_v1_best.pth
  -> Reverse recommendations will use ALS Reverse or content-based
```

### 3. Test via API Endpoints

#### Test Developer Recommendations

```bash
# Get a startup ID first
curl http://localhost:5000/api/recommendations/developers/for-startup/<STARTUP_ID>?limit=5

# Response will include:
{
  "developers": [...],
  "method_used": "two_tower_reverse",  # or "als_reverse" or "content_based"
  "model_version": "two_tower_reverse_v1.0",
  "reranked": true,  # if ranker was used
  ...
}
```

#### Test Investor Recommendations

```bash
curl http://localhost:5000/api/recommendations/investors/for-startup/<STARTUP_ID>?limit=5
```

### 4. Check Model Routing Logic

The system uses smart routing based on interaction count:

- **Cold Start (< 5 interactions)**: `content_based`
- **Warm (5-19 interactions)**: `two_tower_reverse` → fallback to `als_reverse` → fallback to `content_based`
- **Hot (20+ interactions)**: `two_tower_reverse` → fallback to `als_reverse` → fallback to `content_based`

The `method_used` field in the response tells you which model was actually used.

### 5. Verify Ranker is Applied

Check the response for:
- `"reranked": true` - indicates ranker was applied
- Log message: `"Reranked X candidates using reverse ranker"`

## Expected Model Files

After training, you should have:

```
models/
├── two_tower_reverse_v1_best.pth          # Two-Tower reverse model
├── two_tower_reverse_v1_encoder.pkl       # Feature encoder
├── two_tower_reverse_v1_config.json      # Model config
├── ranker_reverse_v1.pth                  # Ranker reverse model
├── als_reverse_developer_v1_config.json  # ALS reverse config (developer)
├── als_reverse_developer_v1_user_factors.npy
├── als_reverse_developer_v1_item_factors.npy
├── als_reverse_investor_v1_config.json    # ALS reverse config (investor)
├── als_reverse_investor_v1_user_factors.npy
└── als_reverse_investor_v1_item_factors.npy
```

## Troubleshooting

### Models not loading?

1. Check file paths match in `app.py`:
   - `TWO_TOWER_REVERSE_MODEL_PATH = models/two_tower_reverse_v1_best.pth`
   - `RANKER_REVERSE_MODEL_PATH = models/ranker_reverse_v1.pth`
   - `ALS_REVERSE_MODEL_NAME = "als_reverse_v1"`

2. Check Flask logs for loading errors

3. Verify model files exist:
   ```bash
   ls -lh models/two_tower_reverse_v1*
   ls -lh models/ranker_reverse_v1*
   ls -lh models/als_reverse_*
   ```

### Getting "content_based" instead of ML models?

- Check interaction count: Need >= 5 interactions for ML models
- Check logs: `"Startup {id} has X unique user interactions"`
- Models might have failed to load (check startup logs)

### Ranker not working?

- Check if `ranker_reverse_model` is loaded (check startup logs)
- Check response for `"reranked": true`
- Ranker only applies to personalized recommendations (not trending/popular)

