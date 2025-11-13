# Two-Tower Model - Quick Start Guide

Get up and running with the Two-Tower recommendation model in 5 steps.

## Prerequisites

```bash
# Install PyTorch and ML dependencies
cd recommendation_service
pip install torch numpy scikit-learn tqdm
```

## Step 1: Generate Data (if needed)

```bash
cd backend
python manage.py generate_recommendation_dataset --users 350 --startups 250 --interactions 7500
python manage.py generate_embeddings
```

## Step 2: Generate Training Dataset

```bash
python manage.py generate_two_tower_dataset \
    --output ../recommendation_service/data/two_tower_train.csv \
    --negative-samples 2 \
    --use-case developer_startup
```

**Expected output:** CSV file with ~15,000-20,000 samples

## Step 3: Train the Model

```bash
cd ../recommendation_service
python train_two_tower.py \
    --data data/two_tower_train.csv \
    --epochs 50 \
    --batch-size 256
```

**Training time:** 
- CPU: ~10-15 minutes
- GPU: ~2-3 minutes

**Expected metrics:**
- NDCG@10: 0.45-0.55
- Precision@10: 0.25-0.35

## Step 4: Test Integration

```bash
python test_two_tower_integration.py
```

This runs 5 tests to verify everything is working.

## Step 5: Enable in Production

Update your Flask app initialization:

```python
# In app.py or your endpoint handlers
rec_service = RecommendationService(db, enable_two_tower=True)
```

Start the Flask service:
```bash
python app.py
```

## Verify It's Working

Make a recommendation request:

```bash
curl "http://localhost:5000/api/recommendations/startups/for-developer/<user-id>?limit=10"
```

Check the response for `"method_used": "two_tower"` (for users with 5+ interactions).

## Troubleshooting

### "No data loaded"
→ Run Step 1 first to generate synthetic data

### "Missing embeddings"  
→ Run `python manage.py generate_embeddings`

### "No trained model found"
→ Complete Step 3 to train the model

### "CUDA out of memory"
→ Use smaller batch size: `--batch-size 128`

## Next Steps

- Read the full guide: `TWO_TOWER_TRAINING_GUIDE.md`
- Monitor training: Check `models/two_tower_v1_history.json`
- Tune hyperparameters: Adjust learning rate, architecture
- Set up periodic retraining: Coming soon!

## Architecture Overview

```
User Features              Startup Features
     ↓                           ↓
[User Tower]              [Startup Tower]
  (512→256→128)            (468→256→128)
     ↓                           ↓
[User Embedding]          [Startup Embedding]
     └──────── dot product ──────┘
                  ↓
            sigmoid(score)
                  ↓
           [0.0 to 1.0]
```

## Key Files

- `train_two_tower.py` - Training script
- `engines/two_tower.py` - Model architecture  
- `engines/feature_engineering.py` - Feature processing
- `engines/evaluation.py` - Metrics computation
- `models/two_tower_v1.pth` - Trained model (after Step 3)

## Questions?

Check the logs: `logs/recommendation_service.log`

---

**Happy Training! 🚀**

