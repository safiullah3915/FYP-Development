# Two-Tower Model Training Guide

This guide explains how to train and deploy the Two-Tower neural network recommendation model.

## Overview

The Two-Tower model is a deep learning architecture for recommendation that learns separate embeddings for users and items (startups), then computes similarity via dot product.

**Architecture:**
- **User Tower**: Encodes user features → 128-dim embedding
- **Startup Tower**: Encodes startup features → 128-dim embedding  
- **Scoring**: `similarity = sigmoid(dot_product(user_emb, startup_emb))`

## Prerequisites

1. **Install PyTorch and dependencies:**
   ```bash
   cd recommendation_service
   pip install torch numpy scikit-learn tqdm
   ```

2. **Generate synthetic data (if not done):**
   ```bash
   cd ../backend
   python manage.py generate_recommendation_dataset --users 350 --startups 250 --interactions 7500
   ```

3. **Generate embeddings for all users and startups:**
   ```bash
   python manage.py generate_embeddings
   ```

## Step 1: Generate Training Dataset

The dataset generator creates training data with smart labeling:

**Labeling Strategy:**
- `apply` (weight=3.0) → label=1.0 (strongest positive)
- `interest` (weight=3.5) → label=1.0
- `favorite` (weight=2.5) → label=0.9
- `like` (weight=2.0) → label=0.8
- `click` (weight=1.0) → label=0.6 (weak positive)
- `view` (weight=0.5) → label=0.4 (very weak positive)
- `dislike` (weight=-1.0) → label=0.0 (explicit negative)
- `negative_sample` → label=0.0 (implicit negative)

**Generate dataset:**
```bash
cd backend
python manage.py generate_two_tower_dataset \
    --output ../recommendation_service/data/two_tower_train.csv \
    --negative-samples 2 \
    --use-case developer_startup \
    --min-interactions 1
```

**Output:**
- CSV file with user-startup pairs, labels, and features
- Includes embeddings, categorical features, and metadata

## Step 2: Train the Model

Train the Two-Tower model using the generated dataset:

```bash
cd ../recommendation_service
python train_two_tower.py \
    --data data/two_tower_train.csv \
    --output-dir models \
    --model-name two_tower_v1 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --embedding-dim 128
```

**Training Process:**
1. Loads and processes dataset (train/val/test split: 70/15/15)
2. Fits feature encoders on training data
3. Trains model with early stopping (patience=5)
4. Saves best model based on validation NDCG@10
5. Outputs training history and metrics

**Saved Files:**
- `models/two_tower_v1.pth` - Best model checkpoint
- `models/two_tower_v1_encoder.pkl` - Feature encoder
- `models/two_tower_v1_config.json` - Model configuration
- `models/two_tower_v1_history.json` - Training history
- `models/two_tower_v1_latest.pth` - Latest checkpoint

**Training Metrics:**
- Loss: Weighted Binary Cross-Entropy
- Evaluation: Precision@10, Recall@10, NDCG@10, F1@10, Hit Rate@10, MAP

## Step 3: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python train_two_tower.py \
    --evaluate \
    --model models/two_tower_v1.pth \
    --data data/two_tower_train.csv
```

**Expected Metrics:**
- NDCG@10: 0.4-0.6 (good)
- Precision@10: 0.2-0.4
- Recall@10: 0.3-0.5
- Hit Rate@10: 0.6-0.8

## Step 4: Deploy the Model

### Option A: Enable in Flask Service

Update `recommendation_service/app.py` to enable two-tower:

```python
# In the endpoint handlers, initialize with enable_two_tower=True
rec_service = RecommendationService(db, enable_two_tower=True)
```

### Option B: Manual Testing

Test the model directly:

```python
from engines.two_tower import TwoTowerRecommender
from database.connection import SessionLocal

db = SessionLocal()
recommender = TwoTowerRecommender(
    db_session=db,
    model_path='models/two_tower_v1.pth',
    encoder_path='models/two_tower_v1_encoder.pkl'
)

# Get recommendations
results = recommender.recommend(
    user_id='<user-uuid>',
    use_case='developer_startup',
    limit=10,
    filters={'type': 'collaboration'}
)

print(f"Recommended startups: {results['item_ids']}")
```

## Model Configuration

### Default Configuration
- **Hidden layers**: [512, 256]
- **Embedding dim**: 128
- **Dropout**: 0.3 (first layer), 0.2 (middle layers)
- **Learning rate**: 0.001 (with cosine annealing)
- **Batch size**: 256
- **Optimizer**: AdamW (weight_decay=0.01)
- **Early stopping**: patience=5

### Custom Configuration

Create a custom config file:

```json
{
  "embedding_dim": 256,
  "hidden_dims": [1024, 512, 256],
  "dropout_rate": 0.4,
  "learning_rate": 0.0005,
  "batch_size": 128,
  "num_epochs": 100
}
```

Train with custom config:
```bash
python train_two_tower.py \
    --data data/two_tower_train.csv \
    --config my_config.json
```

## Routing Logic

The recommendation router automatically selects the best engine:

**Cold Start (< 5 interactions):**
- Uses content-based recommendations
- Relies on user preferences and embeddings

**Warm Users (5-20 interactions):**
- Uses two-tower model if enabled
- Falls back to content-based if model unavailable

**Hot Users (> 20 interactions):**
- Uses two-tower model (best performance)
- Model has learned user preferences from history

## Monitoring & Health Metrics

### Check Model Performance

```bash
# View training history
cat models/two_tower_v1_history.json

# View test metrics
cat models/two_tower_v1_test_metrics.json
```

### List Available Models

```python
from engines.model_registry import get_registry

registry = get_registry()
models = registry.list_available_models()
for model in models:
    print(f"{model['name']}: {model['size_mb']:.2f} MB")
```

## Troubleshooting

### Issue: "No data loaded"
**Solution:** Ensure you've generated interactions and embeddings first.

### Issue: "Missing embeddings"
**Solution:** Run `python manage.py generate_embeddings` to create embeddings for all users/startups.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU:
```bash
python train_two_tower.py --batch-size 128 ...
```

### Issue: "Model not loading in Flask"
**Solution:** Check model paths in `model_registry.py` and ensure files exist:
```bash
ls -la models/two_tower_v1*
```

## Performance Tips

### 1. Data Quality
- Ensure sufficient training data (>5000 samples)
- Balance positive/negative examples
- Include diverse interaction types

### 2. Feature Engineering
- Use high-quality embeddings (sentence-transformers)
- Include relevant categorical features
- Normalize features properly

### 3. Hyperparameter Tuning
- Adjust learning rate (0.0001 - 0.001)
- Experiment with embedding dimensions (64, 128, 256)
- Try different architectures ([256, 128], [512, 256], [1024, 512, 256])

### 4. Training Optimization
- Use GPU if available (10-20x faster)
- Enable mixed precision for larger models
- Monitor validation metrics closely

## Future Enhancements

This implementation is designed to be easily extended:

### ETL Pipeline (Future)
- Scheduled data extraction
- Incremental training dataset updates
- Data versioning with DVC

### Scheduled Training (Future)
- Weekly/monthly retraining
- Automatic model deployment
- A/B testing framework

### Model Versioning (Future)
- Track experiments with MLflow
- Model registry in database
- Rollback capabilities

### Health Monitoring (Future)
- Precision/recall drift detection
- Coverage monitoring
- Online evaluation metrics

## Dataset Statistics

After generating the dataset, you should see:

```
Total samples: ~15,000-20,000
Label Distribution:
  label=1.0: ~3000 (15-20%)  # Strong positives
  label=0.9: ~1500 (8-10%)   # Favorites
  label=0.8: ~2000 (10-12%)  # Likes
  label=0.6: ~3500 (18-20%)  # Clicks
  label=0.4: ~7000 (35-40%)  # Views
  label=0.0: ~3000 (15-20%)  # Negatives

Embedding Coverage: >95%
```

## Example Training Output

```
Loading dataset from data/two_tower_train.csv...
Loaded 18543 samples

Processing dataset...
Split: train=12980, val=2781, test=2782

Feature dimensions:
  User: 512
  Startup: 468

Starting training...
  Epochs: 50
  Batch size: 256
  Learning rate: 0.001
  Device: cuda

Epoch 1/50
Training: 100%|████████| 51/51 [00:12<00:00]
  Train loss: 0.4523
Validation: 100%|████████| 11/11 [00:02<00:00]
  Val loss: 0.4312
  Val metrics:
    ndcg@10: 0.4523
    precision@10: 0.2841
    recall@10: 0.3621
  New best model! NDCG@10: 0.4523

...

Training completed!
  Best epoch: 18
  Best NDCG@10: 0.5234
```

## Quick Reference Commands

```bash
# 1. Generate dataset
cd backend
python manage.py generate_two_tower_dataset --output ../recommendation_service/data/train.csv

# 2. Train model
cd ../recommendation_service
python train_two_tower.py --data data/train.csv --epochs 50 --batch-size 256

# 3. Evaluate model
python train_two_tower.py --evaluate --model models/two_tower_v1.pth --data data/train.csv

# 4. Test in Flask (enable two-tower in app.py first)
python app.py

# 5. Make recommendation request
curl "http://localhost:5000/api/recommendations/startups/for-developer/<user-id>?limit=10"
```

---

For questions or issues, check the logs in `recommendation_service/logs/` or consult the main README.

