# ALS Collaborative Filtering (SVD) Training Guide

## Overview

This guide explains how to train and deploy the collaborative filtering model that powers the “ALS” pathway in the service. The implementation now uses a dependency-free TruncatedSVD pipeline (scikit-learn) while preserving the existing ALS-style inference APIs.

## What is ALS/SVD?

Conceptually we still rely on ALS-style user/item embeddings for implicit feedback data (views, clicks, likes). Under the hood the embeddings are produced via `TruncatedSVD`, which approximates the same factorization without the heavyweight `implicit` library.

### Why this approach?

- **Fast inference**: Precomputed embeddings enable millisecond recommendations
- **Lightweight dependencies**: Pure NumPy/SciPy/scikit-learn, no binary wheels
- **Proven effectiveness**: Same factorization quality as ALS in practice
- **Handles sparsity**: Works well with sparse interaction matrices
- **Implicit feedback**: Designed for views, clicks, likes (not explicit ratings)

## Training Pipeline

### Step 1: Generate Dataset

```bash
cd backend
python manage.py generate_als_dataset
```

**What it does:**
- Queries `UserInteraction` table
- Builds sparse user-item matrix
- Applies interaction weights:
  - view: 0.5
  - click: 1.0
  - like: 2.0
  - dislike: -1.0
  - favorite: 2.5
  - apply: 3.0
  - interest: 3.5
- Saves to `recommendation_service/data/`:
  - `als_interactions.npz` (sparse matrix)
  - `als_user_mapping.json` (user UUID → index)
  - `als_item_mapping.json` (startup UUID → index)

**Options:**
```bash
python manage.py generate_als_dataset --output-dir ../recommendation_service/data --min-interactions 1
```

### Step 2: Train SVD Model

```bash
cd recommendation_service
python train_als.py
```

**Default hyperparameters:**
- `factors`: 128 (latent dimensions / SVD components)
- `iterations`: 10 (power iterations for TruncatedSVD)
- `random-state`: 42 (reproducibility)

**Custom training:**
```bash
python train_als.py \
  --data data/als_interactions.npz \
  --factors 128 \
  --iterations 10 \
  --test-split 0.1
```

**Output:**
- `models/als_v1_user_factors.npy` (user embeddings)
- `models/als_v1_item_factors.npy` (item embeddings)
- `models/als_v1_user_mapping.json`
- `models/als_v1_item_mapping.json`
- `models/als_v1_config.json` (metadata describing the SVD run)

### Step 3: Verify Model

Check that models were created:
```bash
ls -lh recommendation_service/models/als_v1*
```

You should see 5 files (2 embeddings, 2 mappings, 1 config).

## Hyperparameter Tuning

### Factors (Embedding Dimensions)

| Factors | Memory | Training Time | Quality |
|---------|--------|---------------|---------|
| 64 | Low | Fast | Good |
| 128 | Medium | Medium | **Best** |
| 256 | High | Slow | Marginal gain |

**Recommendation**: Start with 128, increase to 256 only if you have >100K interactions.

### Regularization

| Iterations | Training Time | Stability |
|------------|---------------|-----------|
| 5 | Fast | May underfit |
| 10 | **Balanced** | **Good** |
| 20 | Slower | Marginal gains |

**Recommendation**: 10 iterations is sufficient for most datasets.

## Evaluation Metrics

During training, the script now reports lightweight diagnostics:
- **Explained variance ratio**: Total variance captured by the latent space
- **Sampled reconstruction MSE**: Sanity check on matrix reconstruction

Higher explained variance (≥0.3 for sparse data) is generally good; monitor MSE for large spikes when tuning.

## Integration

### Automatic Loading

The Flask app automatically loads the SVD artifacts on startup if the `.npy`, mapping JSONs, and config JSON exist in `recommendation_service/models/`.

### Routing Logic

- **Cold start** (< 5 interactions): Content-Based
- **Warm users** (5-19 interactions): **ALS**
- **Hot users** (20+ interactions): Ensemble (ALS + Two-Tower)

### Manual Testing

```python
from inference_als import ALSInference

# Load model (pass the config path; loader finds companion files)
als = ALSInference('models/als_v1_config.json')

# Get recommendations
results = als.recommend(user_id='some-uuid', limit=10)

print(results['startups'])
print(results['method_used'])  # Should be 'als'
```

## Retraining Schedule

### When to Retrain

- **Weekly**: For active platforms with daily new interactions
- **Monthly**: For moderate activity platforms
- **As needed**: After major user/item growth

### Quick Retrain

```bash
./train_all_models.sh  # Trains both ALS and Two-Tower
```

## Troubleshooting

### Error: "No interactions found"

**Cause**: Empty UserInteraction table  
**Fix**: Ensure users have interacted with startups (views, likes, etc.)

### Error: "Matrix too sparse"

**Cause**: Too few interactions per user/item  
**Fix**: Lower `--min-interactions` threshold

### Error: "Model loading failed"

**Cause**: Missing model files  
**Fix**: Retrain model, ensure all 5 files are present

### Poor Performance (Precision < 0.10)

**Causes**:
1. Insufficient data (< 1000 interactions)
2. Too many cold-start users
3. Wrong hyperparameters

**Fixes**:
1. Collect more interaction data
2. Use content-based for cold-start
3. Tune hyperparameters (see tuning section)

## Advanced: Grid Search

Find optimal hyperparameters:

```python
import itertools

factors = [64, 128, 256]
regularization = [0.01, 0.05, 0.1]
iterations = [15, 20, 30]

best_precision = 0
best_params = None

for f, r, i in itertools.product(factors, regularization, iterations):
    # Train with these params
    precision = train_and_evaluate(f, r, i)
    
    if precision > best_precision:
        best_precision = precision
        best_params = (f, r, i)

print(f"Best params: factors={best_params[0]}, reg={best_params[1]}, iter={best_params[2]}")
```

## Production Checklist

- [ ] Dataset generated successfully
- [ ] ALS/SVD model trained (explained variance ≥ 0.25)
- [ ] All 5 model files present in `models/`
- [ ] Flask app loads ALS model on startup
- [ ] Test recommendations for warm users (5-19 interactions)
- [ ] Monitor `method_used` in API responses

## Summary

ALS-style SVD provides fast, effective collaborative filtering for your recommendation system:

1. **Generate dataset**: `python manage.py generate_als_dataset`
2. **Train model**: `python train_als.py`
3. **Deploy**: Models load automatically
4. **Monitor**: Check Precision@10 and user feedback
5. **Retrain**: Weekly or as needed

For questions or issues, check logs in `recommendation_service/logs/`.


