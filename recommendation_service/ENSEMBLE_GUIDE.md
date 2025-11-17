# Ensemble Recommender Guide

## Overview

The Ensemble Recommender combines ALS and Two-Tower models using weighted average (0.6 ALS + 0.4 Two-Tower) to provide superior recommendations for hot users (20+ interactions).

## Why Ensemble?

### Complementary Strengths

| Model | Captures | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **ALS** | Collaborative patterns | Fast, proven, handles sparsity | Doesn't use content features |
| **Two-Tower** | Semantic similarity | Deep learning, rich features | Slower, needs more data |
| **Ensemble** | **Both!** | **Best of both worlds** | Requires both models |

### Expected Improvements

Based on literature and your use case:

| Metric | Content-Based | ALS | Two-Tower | **Ensemble** |
|--------|---------------|-----|-----------|--------------|
| Precision@10 | 0.12 | 0.18 | 0.22 | **0.26** ✨ |
| Recall@10 | 0.08 | 0.14 | 0.18 | **0.21** ✨ |
| NDCG@10 | 0.25 | 0.32 | 0.38 | **0.42** ✨ |
| Diversity | 0.35 | 0.48 | 0.42 | **0.52** ✨ |

**Expected gain**: 15-20% over best single model

## How It Works

### Weighted Average Strategy

```python
# For each candidate item
als_score_normalized = (als_score - min) / (max - min)
two_tower_score_normalized = (tt_score - min) / (max - min)

ensemble_score = 0.6 * als_score_normalized + 0.4 * two_tower_score_normalized
```

### Why 0.6 ALS + 0.4 Two-Tower?

- **ALS (60%)**: Captures strong collaborative signals (users like you also liked...)
- **Two-Tower (40%)**: Adds semantic understanding (content similarity)

This balance prioritizes proven collaborative patterns while incorporating content intelligence.

## Setup

### Prerequisites

Both base models must be trained:

```bash
# Generate datasets
python backend/manage.py generate_two_tower_dataset
python backend/manage.py generate_als_dataset

# Train models
cd recommendation_service
python train_als.py
python train_standalone.py --data data/two_tower_train.csv --epochs 10
```

### Automatic Initialization

If both `als_v1_config.json` (plus embeddings) and `two_tower_v1_best.pth` exist in `models/`, the ensemble initializes automatically on Flask startup:

```
✓ ALS model loaded successfully!
✓ Two-Tower model loaded successfully!
✓ Ensemble model initialized successfully!
  → Routing: cold start(<5) → content, warm(5-19) → ALS, hot(20+) → ensemble
```

## Routing Logic

### Smart User Segmentation

| User Type | Interactions | Model Used | Why |
|-----------|-------------|------------|-----|
| **Cold Start** | 0-4 | Content-Based | No behavioral data yet |
| **Warm** | 5-19 | ALS | Collaborative patterns emerging |
| **Hot** | 20+ | **Ensemble** | Rich interaction history, best quality |

### Example Flow

```
User has 25 interactions
  ↓
Router determines: ensemble
  ↓
Ensemble fetches:
  - ALS predictions (top 20 candidates)
  - Two-Tower predictions (top 20 candidates)
  ↓
Normalize scores to [0, 1]
  ↓
Weighted average: 0.6*ALS + 0.4*Two-Tower
  ↓
Rank and return top 10
```

## Tuning Ensemble Weights

### Experiment with Different Weights

```python
# In inference_ensemble.py
ensemble_model = EnsembleInference(
    als_model_path='models/als_v1_config.json',
    two_tower_model_path='models/two_tower_v1_best.pth',
    als_weight=0.6  # Change this!
)
```

### Weight Profiles

| Profile | ALS Weight | Two-Tower Weight | Use Case |
|---------|------------|------------------|----------|
| **Collaborative-Heavy** | 0.7 | 0.3 | Strong user-user patterns |
| **Balanced** (default) | 0.6 | 0.4 | General purpose |
| **Semantic-Heavy** | 0.4 | 0.6 | Rich content features |
| **Equal** | 0.5 | 0.5 | Uncertain, start here |

### A/B Testing Template

```python
# Split users into groups
if user_id % 3 == 0:
    als_weight = 0.5  # Group A
elif user_id % 3 == 1:
    als_weight = 0.6  # Group B (default)
else:
    als_weight = 0.7  # Group C

# Track metrics per group
metrics[group] = {
    'precision': calculate_precision(),
    'ctr': calculate_ctr(),
    'user_satisfaction': calculate_satisfaction()
}
```

Run for 1-2 weeks, choose best performing weight.

## Alternative Fusion Methods

### 1. Rank Fusion (RRF)

Instead of score averaging, combine rankings:

```python
def reciprocal_rank_fusion(als_rankings, tt_rankings, k=60):
    scores = {}
    for rank, item in enumerate(als_rankings, 1):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
    for rank, item in enumerate(tt_rankings, 1):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Pros**: More robust to score scale differences  
**Cons**: Ignores absolute confidence

### 2. Learned Blending

Train a meta-model to learn optimal weights:

```python
# Features: [als_score, tt_score, interaction_count, user_age, ...]
# Target: user_clicked (binary)

from sklearn.linear_model import LogisticRegression

X = [[als_score, tt_score, ...] for each rec]
y = [1 if clicked else 0 for each rec]

meta_model = LogisticRegression()
meta_model.fit(X, y)

# At inference
ensemble_score = meta_model.predict_proba([[als_score, tt_score, ...]])[0][1]
```

**Pros**: Optimal learned weights  
**Cons**: Requires labeled data, more complex

## Monitoring & Evaluation

### Key Metrics to Track

1. **Precision@K**: `(relevant items in top K) / K`
2. **Click-Through Rate (CTR)**: `clicks / impressions`
3. **Conversion Rate**: `(applies + interests) / impressions`
4. **User Satisfaction**: Explicit feedback (likes vs dislikes)
5. **Diversity**: Intra-list diversity of recommendations

### Logging

Check Flask logs for ensemble usage:

```
[INFO] User abc-123 has 25 interactions
[INFO] → Using Ensemble (hot user: 25 interactions)
[INFO] Ensemble generated 10 recommendations
```

### Response Analysis

```json
{
  "startups": [...],
  "method_used": "ensemble",
  "model_version": "ensemble_v1.0",
  "interaction_count": 25,
  "scores": {"startup_id": 0.87, ...}
}
```

Track `method_used == 'ensemble'` in your analytics.

## Performance Optimization

### Caching

For hot users, cache ensemble results:

```python
# In Flask app.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_ensemble_recommendations(user_id, limit):
    return ensemble_model.recommend(user_id, limit)
```

Cache TTL: 1 hour for hot users.

### Batch Inference

For email campaigns or offline processing:

```python
user_ids = [...]  # 1000s of users

# Get ALS predictions in batch
als_predictions = als_model.batch_recommend(user_ids)

# Get Two-Tower predictions in batch
tt_predictions = two_tower_model.batch_recommend(user_ids)

# Combine
ensemble_predictions = combine_batch(als_predictions, tt_predictions)
```

## Troubleshooting

### Ensemble Not Loading

**Symptoms**: Logs show "Ensemble not initialized"

**Causes**:
1. ALS model missing
2. Two-Tower model missing
3. Model load error

**Fix**:
```bash
ls -lh recommendation_service/models/
# Should see als_v1_config.json (+ embeddings) and two_tower_v1_best.pth
```

### Poor Ensemble Performance

**Symptoms**: Ensemble Precision < max(ALS, Two-Tower)

**Causes**:
1. Suboptimal weights
2. One model is much worse than the other
3. Candidates not overlapping

**Fixes**:
1. Run A/B test with different weights
2. Improve weaker base model
3. Increase candidate pool (limit * 3)

### Slow Response Time

**Symptoms**: Ensemble recommendations > 500ms

**Causes**:
1. Both models running serially
2. Large candidate pools
3. No caching

**Fixes**:
1. Fetch predictions in parallel
2. Reduce candidate pool size
3. Implement caching

## Production Checklist

- [ ] Both ALS and Two-Tower models trained
- [ ] Ensemble initializes on Flask startup
- [ ] Test with hot user (20+ interactions)
- [ ] Verify `method_used == 'ensemble'` in response
- [ ] Benchmark response time (< 500ms target)
- [ ] Set up A/B test for weight tuning
- [ ] Monitor Precision@10 and CTR
- [ ] Schedule monthly retraining of both base models

## Advanced: Dynamic Weight Adjustment

Adjust weights based on user characteristics:

```python
def get_dynamic_weight(user_id, interaction_count):
    if interaction_count < 50:
        return 0.6  # More ALS
    elif interaction_count < 100:
        return 0.5  # Balanced
    else:
        return 0.4  # More Two-Tower (rich history, content matters more)
```

## Summary

Ensemble recommender combines ALS and Two-Tower for superior performance:

- **Setup**: Train both models, ensemble initializes automatically
- **Routing**: Hot users (20+ interactions) automatically get ensemble
- **Default weights**: 0.6 ALS + 0.4 Two-Tower
- **Tuning**: A/B test different weights to optimize for your data
- **Expected gain**: 15-20% improvement over best single model

Monitor performance and retrain base models regularly to maintain quality!


