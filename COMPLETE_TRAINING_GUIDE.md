# Complete Two-Tower Model Training & Deployment Guide

## 🎯 Full Production Pipeline

### Phase 1: Data Preparation ✅ DONE

```bash
cd backend

# 1. Generate synthetic data (if not done)
python manage.py generate_recommendation_dataset --users 350 --startups 250 --interactions 7500

# 2. Generate embeddings
python manage.py generate_embeddings

# 3. Generate training dataset
python manage.py generate_two_tower_dataset \
    --output ../recommendation_service/data/two_tower_train.csv \
    --negative-samples 2 \
    --use-case developer_startup
```

**Output**: `recommendation_service/data/two_tower_train.csv` (~3,360 samples)

---

### Phase 2: Model Training ✅ DONE (Basic) - NOW DO FULL

```bash
cd ../recommendation_service

# QUICK TEST (5-10 epochs) - Already done ✓
python train_standalone.py --data data/two_tower_train.csv --epochs 10 --batch-size 256

# FULL PRODUCTION TRAINING (50 epochs with early stopping)
python train_standalone.py --data data/two_tower_train.csv --epochs 50 --batch-size 256
```

**Expected Results**:
- Training time: 5-10 minutes (CPU), 1-2 minutes (GPU)
- Best NDCG@10: 0.45-0.55
- Model saved: `models/two_tower_v1_best.pth`

---

### Phase 3: Model Evaluation 📊 TODO

Create evaluation script:

```bash
cd recommendation_service
python evaluate_model.py --model models/two_tower_v1_best.pth --data data/two_tower_train.csv
```

**Metrics to Track**:
- Precision@10, Recall@10
- NDCG@10, NDCG@20
- Hit Rate@10
- MAP (Mean Average Precision)

---

### Phase 4: Model Integration 🔌 TODO

#### Option A: Use Inference Module (Recommended)

```python
from inference_two_tower import TwoTowerInference

# Initialize
inference = TwoTowerInference("models/two_tower_v1_best.pth")

# Get recommendations
results = inference.recommend(
    user_id="user-uuid",
    limit=10,
    filters={'type': 'collaboration'}
)

print(f"Recommended: {results['item_ids']}")
print(f"Scores: {results['scores']}")
```

#### Option B: Update Flask Service

Update `recommendation_service/app.py`:

```python
from inference_two_tower import TwoTowerInference

# Initialize at startup
two_tower_inference = TwoTowerInference("models/two_tower_v1_best.pth")

@app.route('/api/recommendations/startups/for-developer/<user_id>')
def get_startups_for_developer(user_id):
    # Use two-tower for users with 5+ interactions
    interaction_count = get_interaction_count(user_id)
    
    if interaction_count >= 5:
        # Use two-tower
        results = two_tower_inference.recommend(user_id, limit=10)
    else:
        # Use content-based (cold start)
        results = content_based_recommender.recommend(user_id, limit=10)
    
    return jsonify(results)
```

---

### Phase 5: Production Deployment 🚀 TODO

#### Step 1: Test Integration

```bash
cd recommendation_service
python inference_two_tower.py  # Runs test_inference()
```

Expected output:
```
Testing Two-Tower inference...
Loading model from models/two_tower_v1_best.pth
Model loaded successfully on cpu
Testing with user: john_doe (uuid-here)
Got 5 recommendations:
  1. Startup xxx: score=0.723
  2. Startup yyy: score=0.698
  ...
✓ Inference test successful!
```

#### Step 2: Update Router Logic

```python
# In recommendation_service/services/recommendation_service.py

def __init__(self, db_session, enable_two_tower: bool = True):
    self.db = db_session
    self.router = RecommendationRouter(enable_two_tower=enable_two_tower)
    self.content_based = ContentBasedRecommender(db_session)
    
    # Use inference module instead of full two_tower
    if enable_two_tower:
        from inference_two_tower import TwoTowerInference
        self.two_tower = TwoTowerInference("models/two_tower_v1_best.pth")
    else:
        self.two_tower = None
```

#### Step 3: Deploy & Monitor

```bash
# Start Flask service
cd recommendation_service
python app.py

# Test endpoint
curl "http://localhost:5000/api/recommendations/startups/for-developer/<user-id>?limit=10"
```

---

## 📋 Complete Checklist

### Phase 1: Data ✅
- [x] Generate synthetic data
- [x] Generate embeddings  
- [x] Generate training dataset
- [x] Verify dataset statistics

### Phase 2: Training ✅ (Partial)
- [x] Quick training test (10 epochs) ✓
- [ ] **Full training (50 epochs)** ← DO THIS NOW
- [ ] Save training metrics
- [ ] Validate model performance

### Phase 3: Evaluation 📊
- [ ] Create evaluation script
- [ ] Compute ranking metrics
- [ ] Compare with content-based baseline
- [ ] Save evaluation results

### Phase 4: Integration 🔌
- [x] Create inference module ✓
- [ ] Test inference locally
- [ ] Update Flask endpoints
- [ ] Add routing logic
- [ ] Test end-to-end flow

### Phase 5: Production 🚀
- [ ] Deploy model to production
- [ ] Monitor recommendations
- [ ] Track metrics (CTR, engagement)
- [ ] A/B test vs content-based

---

## 🚀 Next Steps (In Order)

### Step 1: Full Training (10 minutes)

```bash
cd recommendation_service
python train_standalone.py --data data/two_tower_train.csv --epochs 50 --batch-size 256
```

### Step 2: Test Inference

```bash
python inference_two_tower.py
```

### Step 3: Integrate into Flask

Update `app.py` to use `TwoTowerInference` for warm/hot users.

### Step 4: Deploy & Monitor

Start service and monitor recommendation quality.

---

## 📊 Expected Performance

| Metric | Cold Start (Content) | Two-Tower (Warm/Hot) |
|--------|---------------------|---------------------|
| **NDCG@10** | 0.35-0.45 | 0.45-0.55 |
| **Precision@10** | 0.20-0.30 | 0.25-0.35 |
| **Recall@10** | 0.25-0.35 | 0.30-0.45 |
| **Hit Rate@10** | 0.50-0.70 | 0.60-0.80 |

---

## 🔧 Troubleshooting

### Issue: Low Metrics After Training

**Solutions**:
1. Generate more training data (10K+ interactions)
2. Generate better embeddings
3. Tune hyperparameters (learning rate, layers)
4. Train for more epochs

### Issue: Inference Fails

**Solutions**:
1. Check model file exists: `ls models/two_tower_v1_best.pth`
2. Verify feature dimensions match (502 user, 471 startup)
3. Check database has embeddings

### Issue: Slow Inference

**Solutions**:
1. Use GPU if available
2. Batch predictions (32-64 startups at once)
3. Cache user embeddings
4. Pre-compute startup embeddings

---

## 💡 Pro Tips

### Training
- **Use GPU**: 10-20x faster training
- **Monitor validation loss**: Early stopping prevents overfitting
- **Save checkpoints**: Keep best model based on validation

### Feature Engineering  
- **Save encoders**: Pickle encoders during training, load for inference
- **Handle missing data**: Use zeros for missing embeddings
- **Normalize features**: Already done in training script

### Production
- **Warm-up time**: First prediction takes ~1-2 seconds (model loading)
- **Batch inference**: Process multiple users together for efficiency
- **Cache results**: Cache recommendations for 1-24 hours
- **Monitor metrics**: Track precision drift over time

---

## 📁 File Structure

```
recommendation_service/
├── data/
│   └── two_tower_train.csv          # Training dataset
├── models/
│   ├── two_tower_v1_best.pth        # Trained model ✓
│   └── (future) encoders.pkl        # Feature encoders
├── train_standalone.py              # Training script ✓
├── inference_two_tower.py           # Inference module ✓
├── app.py                           # Flask service (update needed)
└── COMPLETE_TRAINING_GUIDE.md       # This file

backend/
├── api/management/commands/
│   └── generate_two_tower_dataset.py  # Dataset generator ✓
└── db.sqlite3                       # Database with interactions
```

---

## 🎓 What You've Accomplished

✅ **Dataset Generation**: 3,360 samples with smart labeling  
✅ **Model Architecture**: 827K parameter two-tower network  
✅ **Training Pipeline**: Standalone script with early stopping  
✅ **Basic Training**: 10 epochs, validation loss 0.68  
✅ **Inference Module**: Ready for production integration  

---

## 🎯 What's Left

⏳ **Full Training**: 50 epochs for better performance (10 min)  
⏳ **Testing**: Verify inference works end-to-end  
⏳ **Integration**: Connect to Flask endpoints  
⏳ **Deployment**: Enable in production  
⏳ **Monitoring**: Track metrics and performance  

---

## 🚀 Quick Commands

```bash
# Full training (do this now!)
cd recommendation_service
python train_standalone.py --data data/two_tower_train.csv --epochs 50

# Test inference
python inference_two_tower.py

# Start Flask service (after integration)
python app.py

# Generate new dataset (when you have more data)
cd ../backend
python manage.py generate_two_tower_dataset --output ../recommendation_service/data/train.csv
```

---

## 📞 Need Help?

Check these files for details:
- **Training issues**: `TRAINING_SUCCESS.md`
- **Architecture details**: `TWO_TOWER_TRAINING_GUIDE.md`
- **Quick start**: `QUICK_START.md`
- **Implementation**: `TWO_TOWER_IMPLEMENTATION_SUMMARY.md`

---

**Ready to complete the training? Run the full 50 epochs now!** 🚀

