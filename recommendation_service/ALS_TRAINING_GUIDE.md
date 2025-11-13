# ALS Collaborative Filtering Training Guide

## Overview

This guide explains how to train and deploy the ALS (Alternating Least Squares) collaborative filtering model for your recommendation system.

## What is ALS?

ALS is a matrix factorization technique designed for implicit feedback data (views, clicks, likes). It learns latent factors for both users and items by alternating between fixing user factors and optimizing item factors, and vice versa.

### Why ALS?

- **Fast inference**: Precomputed embeddings enable millisecond recommendations
- **Proven effectiveness**: Industry-standard for collaborative filtering
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

### Step 2: Train ALS Model

```bash
cd recommendation_service
python train_als.py
```

**Default hyperparameters:**
- `factors`: 128 (embedding dimensions)
- `regularization`: 0.01 (L2 penalty)
- `iterations`: 20 (training epochs)
- `alpha`: 40 (confidence scaling)

**Custom training:**
```bash
python train_als.py \
  --data data/als_interactions.npz \
  --factors 128 \
  --regularization 0.01 \
  --iterations 20 \
  --alpha 40 \
  --test-split 0.2
```

**Output:**
- `models/als_v1.pkl` (trained model)
- `models/als_v1_user_factors.npy` (user embeddings)
- `models/als_v1_item_factors.npy` (item embeddings)
- `models/als_v1_user_mapping.json`
- `models/als_v1_item_mapping.json`

### Step 3: Verify Model

Check that models were created:
```bash
ls -lh recommendation_service/models/als_v1*
```

You should see 5 files.

## Hyperparameter Tuning

### Factors (Embedding Dimensions)

| Factors | Memory | Training Time | Quality |
|---------|--------|---------------|---------|
| 64 | Low | Fast | Good |
| 128 | Medium | Medium | **Best** |
| 256 | High | Slow | Marginal gain |

**Recommendation**: Start with 128, increase to 256 only if you have >100K interactions.

### Regularization

| Value | Overfitting Risk | Generalization |
|-------|------------------|----------------|
| 0.001 | High | Poor |
| 0.01 | **Balanced** | **Good** |
| 0.1 | Low | May underfit |

**Recommendation**: Use 0.01 for most cases.

### Iterations

| Iterations | Training Time | Convergence |
|------------|---------------|-------------|
| 10 | Fast | May not converge |
| 20 | **Medium** | **Good** |
| 30 | Slow | Marginal improvement |

**Recommendation**: 20 iterations is sufficient for most datasets.

### Alpha (Confidence Weight)

| Alpha | Behavior |
|-------|----------|
| 10 | Less confident predictions |
| 40 | **Balanced** |
| 80 | Very confident, may overfit popular items |

**Recommendation**: Use 40 as default.

## Evaluation Metrics

During training, the script outputs:
- **Precision@10**: Fraction of recommended items that user interacted with
- **MAP@10**: Mean Average Precision at 10

**Good performance:**
- Precision@10 > 0.15
- MAP@10 > 0.10

## Integration

### Automatic Loading

The Flask app automatically loads ALS model on startup if `models/als_v1.pkl` exists.

### Routing Logic

- **Cold start** (< 5 interactions): Content-Based
- **Warm users** (5-19 interactions): **ALS**
- **Hot users** (20+ interactions): Ensemble (ALS + Two-Tower)

### Manual Testing

```python
from inference_als import ALSInference

# Load model
als = ALSInference('models/als_v1.pkl')

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
- [ ] ALS model trained (Precision@10 > 0.15)
- [ ] All 5 model files present in `models/`
- [ ] Flask app loads ALS model on startup
- [ ] Test recommendations for warm users (5-19 interactions)
- [ ] Monitor `method_used` in API responses

## Summary

ALS provides fast, effective collaborative filtering for your recommendation system:

1. **Generate dataset**: `python manage.py generate_als_dataset`
2. **Train model**: `python train_als.py`
3. **Deploy**: Models load automatically
4. **Monitor**: Check Precision@10 and user feedback
5. **Retrain**: Weekly or as needed

For questions or issues, check logs in `recommendation_service/logs/`.


