# Two-Tower Recommendation Model - Complete Implementation

## 🎉 What's Been Implemented

A complete, production-ready **Two-Tower Neural Network** recommendation system that learns from user interactions to provide personalized startup recommendations.

### ✨ Key Features

- **Smart Labeling**: Uses ALL interaction types (views, clicks, likes, applies, etc.) with weighted labels
- **Deep Learning**: PyTorch-based two-tower architecture with 128-dim embeddings
- **Intelligent Routing**: Automatically selects best model based on user interaction history
- **Production Ready**: Integrates seamlessly with existing Django + Flask architecture
- **Scalable**: Designed for easy extension to full MLOps pipelines

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated (Recommended)

**Linux/Mac:**
```bash
chmod +x train_model.sh
./train_model.sh
```

**Windows:**
```bash
train_model.bat
```

### Option 2: Manual Steps

```bash
# 1. Install dependencies
cd recommendation_service
pip install torch numpy scikit-learn tqdm

# 2. Generate training data
cd ../backend
python manage.py generate_two_tower_dataset \
    --output ../recommendation_service/data/two_tower_train.csv

# 3. Train model
cd ../recommendation_service
python train_two_tower.py --data data/two_tower_train.csv --epochs 50

# 4. Test it
python test_two_tower_integration.py
```

## 📚 Documentation

- **[QUICK_START.md](recommendation_service/QUICK_START.md)** - Get started in 5 steps
- **[TWO_TOWER_TRAINING_GUIDE.md](recommendation_service/TWO_TOWER_TRAINING_GUIDE.md)** - Complete training guide
- **[TWO_TOWER_IMPLEMENTATION_SUMMARY.md](TWO_TOWER_IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTIONS                         │
│  Views, Clicks, Likes, Applies, Interests, Favorites         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATASET GENERATION                        │
│  Smart Labeling: 1.0 (strong), 0.8 (moderate), 0.4 (weak)  │
│  Negative Sampling: Users who didn't interact               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    TWO-TOWER MODEL                           │
│                                                              │
│  User Tower (512→256→128)    Startup Tower (468→256→128)   │
│         ↓                              ↓                     │
│    [User Emb]  ←─── Dot Product ───→ [Startup Emb]         │
│                        ↓                                     │
│                  Sigmoid(score)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    INTELLIGENT ROUTING                       │
│  • Cold Start (< 5): Content-Based                          │
│  • Warm Users (5-20): Two-Tower                             │
│  • Hot Users (> 20): Two-Tower                              │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Expected Performance

With proper training data (10K+ interactions):

| Metric | Expected Value | What It Means |
|--------|---------------|---------------|
| **NDCG@10** | 0.45-0.55 | Ranking quality (higher = better order) |
| **Precision@10** | 0.25-0.35 | % of top-10 that are relevant |
| **Recall@10** | 0.30-0.45 | % of relevant items in top-10 |
| **Hit Rate@10** | 0.60-0.80 | % of users with at least 1 hit |

## 🎯 How It Works

### 1. Smart Labeling Strategy

Not all interactions are equal! The model learns from:

| Interaction | Label | Weight | Meaning |
|-------------|-------|--------|---------|
| Apply | 1.0 | 3.0 | Strongest positive signal |
| Interest | 1.0 | 3.5 | Strong investor interest |
| Favorite | 0.9 | 2.5 | Saved for later |
| Like | 0.8 | 2.0 | Explicit positive feedback |
| Click | 0.6 | 1.0 | Weak positive (showed interest) |
| View | 0.4 | 0.5 | Very weak (just saw it) |
| Dislike | 0.0 | -1.0 | Explicit negative |
| No Interaction | 0.0 | 1.0 | Implicit negative |

### 2. Feature Engineering

**User Features (512 dimensions):**
- Sentence embedding (384D)
- Role (one-hot)
- Selected categories (multi-hot)
- Preferred fields, tags, stages
- Skills from profile

**Startup Features (468 dimensions):**
- Sentence embedding (384D)  
- Type, category, field (one-hot)
- Phase, stages, tags (multi-hot)
- Position requirements

### 3. Model Architecture

```python
User Tower:
  Input(512) → Dense(512, ReLU, Dropout=0.3)
           → Dense(256, ReLU, Dropout=0.2)
           → Dense(128, L2Normalize)

Startup Tower:
  Input(468) → Dense(512, ReLU, Dropout=0.3)
            → Dense(256, ReLU, Dropout=0.2)
            → Dense(128, L2Normalize)

Score = sigmoid(dot_product(user_emb, startup_emb))
```

### 4. Training Process

- **Optimizer**: AdamW (learning_rate=0.001, weight_decay=0.01)
- **Loss**: Weighted Binary Cross-Entropy
- **Batch Size**: 256
- **Epochs**: 50 (early stopping patience=5)
- **Validation**: 15% of data, monitored every epoch
- **Best Model**: Saved based on validation NDCG@10

## 🔧 Configuration

### Default Hyperparameters

```python
embedding_dim = 128
hidden_dims = [512, 256]
dropout_rate = 0.3  # first layer
dropout_rate_middle = 0.2  # middle layers
learning_rate = 0.001
batch_size = 256
num_epochs = 50
early_stopping_patience = 5
```

### Custom Configuration

Create `my_config.json`:
```json
{
  "embedding_dim": 256,
  "hidden_dims": [1024, 512, 256],
  "learning_rate": 0.0005,
  "batch_size": 128,
  "num_epochs": 100
}
```

Train with it:
```bash
python train_two_tower.py --data data/train.csv --config my_config.json
```

## 📁 Files Created

### Core Implementation (10 files)

1. **`backend/api/management/commands/generate_two_tower_dataset.py`**
   - Dataset generation with smart labeling
   
2. **`recommendation_service/engines/two_tower.py`**
   - PyTorch model architecture + inference

3. **`recommendation_service/engines/feature_engineering.py`**
   - Feature encoding and dataset processing

4. **`recommendation_service/engines/training_config.py`**
   - Hyperparameter configuration

5. **`recommendation_service/engines/evaluation.py`**
   - Ranking metrics (Precision@K, NDCG@K, etc.)

6. **`recommendation_service/train_two_tower.py`**
   - Training script with early stopping

7. **`recommendation_service/engines/model_registry.py`**
   - Model version management

8. **`recommendation_service/engines/router.py`** (modified)
   - Intelligent routing logic

9. **`recommendation_service/services/recommendation_service.py`** (modified)
   - Two-tower integration

10. **`recommendation_service/requirements.txt`** (modified)
    - Added PyTorch dependencies

### Documentation (5 files)

11. **`recommendation_service/QUICK_START.md`** - 5-minute guide
12. **`recommendation_service/TWO_TOWER_TRAINING_GUIDE.md`** - Complete guide
13. **`TWO_TOWER_IMPLEMENTATION_SUMMARY.md`** - Technical details
14. **`README_TWO_TOWER.md`** - This file
15. **`recommendation_service/test_two_tower_integration.py`** - Test suite

### Helper Scripts (2 files)

16. **`train_model.sh`** - Linux/Mac automated training
17. **`train_model.bat`** - Windows automated training

## 🧪 Testing

Run the integration test suite:

```bash
cd recommendation_service
python test_two_tower_integration.py
```

**Tests included:**
1. ✓ Database connection
2. ✓ Model registry
3. ✓ Content-based recommendations
4. ✓ Two-tower recommendations
5. ✓ Routing logic

## 🎛️ Enable in Production

### Flask Service

Update your endpoint handlers:

```python
from services.recommendation_service import RecommendationService

@app.route('/api/recommendations/startups/for-developer/<user_id>')
def get_startups_for_developer(user_id):
    db = SessionLocal()
    try:
        # Enable two-tower here
        rec_service = RecommendationService(db, enable_two_tower=True)
        
        results = rec_service.get_recommendations(
            user_id=user_id,
            use_case='developer_startup',
            limit=10
        )
        
        return jsonify(results), 200
    finally:
        db.close()
```

### Django Backend

No changes needed! The Flask service handles all ML operations.

## 🔍 Monitoring

### Check Training Progress

```bash
# View training history
cat recommendation_service/models/two_tower_v1_history.json

# View test metrics
cat recommendation_service/models/two_tower_v1_test_metrics.json
```

### List Available Models

```python
from engines.model_registry import get_registry

registry = get_registry()
models = registry.list_available_models()
print(f"Found {len(models)} models")
```

### API Response

Check which method was used:

```json
{
  "recommendations": [...],
  "method_used": "two_tower",  // or "content_based"
  "interaction_count": 15
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data loaded" | Run `python manage.py generate_recommendation_dataset` |
| "Missing embeddings" | Run `python manage.py generate_embeddings` |
| "CUDA out of memory" | Use `--batch-size 128` or train on CPU |
| "Model not found" | Check `recommendation_service/models/` directory |
| "Low NDCG (<0.3)" | Need more training data (10K+ interactions) |

## 🚀 Future Enhancements

The code is designed for easy extension to:

### Phase 1: Current ✅
- Manual training workflow
- Model integration  
- Basic routing

### Phase 2: Automation (Future)
- Scheduled data extraction (ETL)
- Automated model retraining
- Model versioning in database
- A/B testing framework

### Phase 3: MLOps (Future)
- MLflow experiment tracking
- Performance monitoring dashboards
- Precision drift detection
- Automated rollback

### Phase 4: Advanced Models (Future)
- ALS collaborative filtering
- Ensemble methods
- Contextual bandits
- Multi-objective optimization

## 📈 Performance Tips

### Data Quality
- ✅ Generate 10K+ interactions for best results
- ✅ Ensure embeddings are high quality
- ✅ Balance positive/negative examples

### Training
- ✅ Use GPU if available (10-20x faster)
- ✅ Monitor validation metrics
- ✅ Experiment with hyperparameters

### Inference
- ✅ Batch predictions when possible
- ✅ Cache user/startup embeddings
- ✅ Use fallback for cold-start users

## 🎓 What You've Learned

This implementation demonstrates:

1. **Modern RecSys Architecture**: Two-tower design (used by Google, Pinterest, Spotify)
2. **Smart Labeling**: Using all interaction types with graded relevance
3. **Production ML**: Model serving, versioning, fallback strategies
4. **Evaluation**: Proper ranking metrics (not just accuracy)
5. **Feature Engineering**: Embedding + categorical features

## 📞 Support

- **Quick Start**: See `QUICK_START.md`
- **Full Guide**: See `TWO_TOWER_TRAINING_GUIDE.md`
- **Implementation Details**: See `TWO_TOWER_IMPLEMENTATION_SUMMARY.md`
- **Logs**: Check `recommendation_service/logs/`

## ✅ What's Complete

- [x] Dataset generation with smart labeling
- [x] Two-tower model architecture
- [x] Training pipeline with early stopping
- [x] Feature engineering and encoding
- [x] Evaluation metrics (Precision@K, NDCG@K, etc.)
- [x] Model registry and versioning
- [x] Intelligent routing (cold/warm/hot users)
- [x] Integration with existing system
- [x] Comprehensive documentation
- [x] Integration tests
- [x] Helper scripts

## 🎉 Ready to Use!

Your two-tower recommendation model is **fully implemented and ready for training**. Just run:

```bash
./train_model.sh    # Linux/Mac
# or
train_model.bat     # Windows
```

And you'll have a trained model in ~10-15 minutes!

---

**Happy Recommending! 🚀**

