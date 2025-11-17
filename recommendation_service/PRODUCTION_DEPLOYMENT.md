# Reverse ALS Production Deployment Guide

## Quick Start (First Time Setup)

### Step 1: Generate Dataset
```bash
cd backend
python manage.py generate_als_dataset
```

This creates 6 files in `recommendation_service/data/`:
- `als_interactions.npz` (forward matrix)
- `als_user_mapping.json`
- `als_item_mapping.json`
- `als_interactions_reverse.npz` (reverse matrix)
- `als_reverse_user_mapping.json`
- `als_reverse_item_mapping.json`

### Step 2: Train Both Models
```bash
cd ../recommendation_service

# Train forward model (User → Startup)
python train_als.py

# Train reverse model (Startup → User)
python train_als_reverse.py
```

### Step 3: Verify Models Created
```bash
ls -lh models/als*_config.json
```

You should see:
- `als_v1_config.json` (forward metadata)
- `als_reverse_v1_config.json` (reverse metadata)

Plus their associated factor files (`*_user_factors.npy`, `*_item_factors.npy`) and mappings.

### Step 4: Restart Flask Service
```bash
# Stop existing service if running
# Then start:
python app.py
```

Look for these log lines:
```
✓ ALS Forward model loaded successfully!
✓ ALS Reverse model loaded successfully!
  -> Will be used for Founder → Developer/Investor recommendations
```

## Verification Tests

### Test 1: Check Health
```bash
curl http://localhost:5001/health
```

Expected:
```json
{
  "status": "healthy",
  "database_connected": true
}
```

### Test 2: Developer → Startup (Forward ALS)
```bash
# Replace <USER_ID> with actual user UUID
curl "http://localhost:5001/api/recommendations/startups/for-developer/<USER_ID>?limit=5"
```

Check response for:
- `method_used`: should be "als" (if user has 5+ interactions) or "content_based"
- `startups`: array of recommendations

### Test 3: Founder → Developer (Reverse ALS)
```bash
# Replace <STARTUP_ID> with actual startup UUID
curl "http://localhost:5001/api/recommendations/developers/for-startup/<STARTUP_ID>?limit=5"
```

Check response for:
- `method_used`: should be "als_reverse" (if startup has 5+ interactions) or "content_based"
- `users`: array of user recommendations

### Test 4: Founder → Investor (Reverse ALS)
```bash
curl "http://localhost:5001/api/recommendations/investors/for-startup/<STARTUP_ID>?limit=5"
```

Check response for:
- `method_used`: should be "als_reverse" or "content_based"
- `users`: array of investor recommendations

## How It Works in Production

### Automatic Model Selection

The system automatically chooses the best model based on available data:

**For Users Finding Startups:**
- Cold start (<5 interactions) → Content-Based matching
- Warm (5-19 interactions) → ALS Forward
- Hot (20+ interactions) → Ensemble (ALS + Two-Tower)

**For Founders Finding Users:**
- Cold start (<5 unique user interactions) → Content-Based matching
- Warm/Hot (≥5 unique user interactions) → ALS Reverse

### Fallback Chain

Every request has multiple fallback layers:
1. Try ALS (forward or reverse)
2. If user/startup not in model → Content-Based
3. If Content-Based fails → Return empty results (rare)

This ensures **100% uptime** even with:
- New users/startups
- Missing model files
- Model loading errors

## Maintenance

### Weekly Retraining (Recommended)

Create a cron job or scheduled task:

```bash
#!/bin/bash
# retrain_models.sh

echo "Starting weekly model retraining..."

# Generate fresh dataset
cd /path/to/backend
python manage.py generate_als_dataset

# Train both models
cd /path/to/recommendation_service
python train_als.py
python train_als_reverse.py

# Restart Flask service
# (use your service manager: systemd, supervisor, pm2, etc.)
sudo systemctl restart recommendation-service

echo "Retraining complete!"
```

Schedule it:
```bash
# Run every Sunday at 2 AM
0 2 * * 0 /path/to/retrain_models.sh
```

### Monitoring

**Key Metrics to Track:**

1. **Model Usage Distribution**
   - % using ALS forward
   - % using ALS reverse  
   - % falling back to content-based

2. **Response Times**
   - Target: < 200ms for recommendations
   - Alert if > 500ms

3. **Interaction Growth**
   - Track daily new interactions
   - Retrain when interactions double

4. **Recommendation Quality**
   - Click-through rate on recommendations
   - Application rate from recommendations

### Log Monitoring

Check logs regularly:
```bash
tail -f recommendation_service/logs/recommendation_service.log
```

Look for:
- Model loading success/failure
- Routing decisions (which model used)
- Fallback usage (indicates cold-start or errors)

## Troubleshooting

### Issue: "ALS Reverse model not found"

**Symptoms:**
- Flask logs show warning about missing model
- All founder use cases fall back to content-based

**Fix:**
```bash
cd backend
python manage.py generate_als_dataset
cd ../recommendation_service
python train_als_reverse.py
# Restart Flask service
```

### Issue: Poor recommendation quality for founders

**Possible Causes:**
1. Not enough interaction data (< 1000 total interactions)
2. Sparse interaction matrix (few users per startup)
3. Model not retrained recently

**Solutions:**
1. Wait for more interactions to accumulate
2. Encourage user engagement (views, likes, applies)
3. Retrain models weekly
4. Lower cold-start threshold temporarily:
   - Edit `recommendation_service/engines/router.py`
   - Change `COLD_START_THRESHOLD = 5` to `3`

### Issue: High API response times

**Diagnosis:**
```bash
# Check model file sizes
ls -lh recommendation_service/models/

# Check database query performance
# Look for slow interaction count queries
```

**Solutions:**
1. Add database indexes on `user_interactions` table:
   ```sql
   CREATE INDEX idx_interactions_startup_user 
   ON user_interactions(startup_id, user_id);
   ```
2. Cache interaction counts (Redis/Memcached)
3. Pre-compute recommendations for popular startups

### Issue: Models won't load after update

**Symptoms:**
- Flask fails to start
- ImportError or pickle errors

**Fix:**
```bash
# Retrain models with current code version
cd recommendation_service
python train_als.py
python train_als_reverse.py
```

Models are pickle files - they need to match the code version.

## Performance Tuning

### Database Optimization

Add indexes for faster interaction counting:

```sql
-- Forward direction (user interaction count)
CREATE INDEX idx_user_interactions_user_id 
ON user_interactions(user_id);

-- Reverse direction (startup interaction count)
CREATE INDEX idx_user_interactions_startup_user 
ON user_interactions(startup_id, user_id);
```

### Model Optimization

If models are too slow, try smaller embedding dimensions:

```bash
# Train with 64 dimensions instead of 128
python train_als.py --factors 64
python train_als_reverse.py --factors 64
```

Trade-off: Faster inference, slightly lower quality.

### Caching

Implement Redis caching for:
- Interaction counts (cache for 5 minutes)
- Popular startup recommendations (cache for 1 hour)
- User recommendations (cache for 30 minutes)

Example:
```python
import redis
r = redis.Redis()

# Cache interaction count
cache_key = f"interaction_count:user:{user_id}"
count = r.get(cache_key)
if count is None:
    count = get_interaction_count(user_id)
    r.setex(cache_key, 300, count)  # 5 min cache
```

## Scaling Considerations

### Horizontal Scaling

Flask recommendation service is stateless - scale horizontally:

1. Load models once at startup (already implemented)
2. Put behind load balancer (nginx, haproxy)
3. Run multiple Flask instances
4. Share database connection pool

### Model Serving

For very high traffic:

1. **Model Serving Service:** Use TensorFlow Serving or TorchServe
2. **Batch Predictions:** Pre-compute recommendations for all users/startups
3. **Approximate Nearest Neighbors:** Use FAISS for faster similarity search

### Database Scaling

If interaction counting becomes slow:

1. **Read Replicas:** Query interaction counts from read replica
2. **Materialized View:** Pre-aggregate interaction counts
3. **Denormalization:** Store interaction count on user/startup records

## Security

### API Authentication

Ensure Flask endpoints are behind authentication:

```python
# In Django backend, proxy through authenticated endpoints
# Flask should only be accessible from Django backend, not public
```

### Model File Security

Protect model files:
```bash
chmod 600 recommendation_service/models/als*_*.{json,npy}
chown appuser:appuser recommendation_service/models/als*_*.{json,npy}
```

### Data Privacy

Models contain learned patterns - ensure compliance:
- Don't expose raw model files publicly
- Log access to recommendation endpoints
- Allow users to opt-out of personalization

## Rollback Plan

If new models cause issues:

1. **Keep Previous Models:**
   ```bash
   mkdir -p models/backups
   cp models/als_v1_* models/backups/
   cp models/als_reverse_v1_* models/backups/
   ```

2. **Quick Rollback:**
   ```bash
   cp models/backups/als_v1_* models/
   cp models/backups/als_reverse_v1_* models/
   # Restart Flask service
   ```

3. **Version Models:**
   - Use timestamped prefixes: `als_v1_20240115_user_factors.npy`, etc.
   - Update Flask to point at the desired prefix
   - Keep last 3 versions

## Success Metrics

Track these KPIs:

1. **Recommendation Coverage:**
   - Target: 100% (via content-based fallback)
   
2. **ALS Usage:**
   - Target: >60% of requests use ALS (forward or reverse)
   - If lower, need more interaction data

3. **Click-Through Rate:**
   - Measure: % of recommendations clicked
   - Compare: ALS vs content-based
   - Expected: ALS 2-3x better than content-based

4. **Conversion Rate:**
   - Applications from recommendations
   - Target: >10% conversion

5. **API Performance:**
   - P50 latency < 100ms
   - P95 latency < 200ms
   - P99 latency < 500ms

## Summary

✅ **Initial Setup:** ~30 minutes
✅ **Weekly Maintenance:** ~5 minutes (automated)
✅ **Monitoring:** Daily log check
✅ **Retraining:** Automated weekly

The system is designed to be:
- **Self-healing:** Automatic fallbacks
- **Low maintenance:** Automated retraining
- **High performance:** <200ms responses
- **Scalable:** Stateless design

You're production-ready! 🚀

