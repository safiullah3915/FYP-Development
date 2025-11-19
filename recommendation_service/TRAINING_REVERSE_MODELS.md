# Training Reverse Recommendation Models

## Overview

This document provides a comprehensive guide for training reverse recommendation models for the startup-centric recommendation system. Reverse recommendations enable startups to find matched developers and investors, as opposed to the forward direction where developers/investors find startups.

## Architecture

The reverse recommendation system uses completely separate models from forward recommendations:

- **Two-Tower Reverse Model**: Deep learning model for warm/hot startups (20+ interactions)
- **ALS Reverse Model**: Collaborative filtering for warm startups (5-19 interactions)
- **Ranker Reverse Model**: Neural ranker for reranking reverse recommendations

### Use Cases

- `startup_developer`: Startup → Developer recommendations
- `startup_investor`: Startup → Investor recommendations

## Dataset Generation

### 1. Two-Tower Dataset

Generate training dataset for the two-tower reverse model:

```bash
cd backend
python manage.py generate_two_tower_dataset \
    --use-case startup_developer \
    --output ../recommendation_service/data/two_tower_reverse_developer.csv \
    --negative-samples 5 \
    --min-interactions 1
```

For investors:

```bash
python manage.py generate_two_tower_dataset \
    --use-case startup_investor \
    --output ../recommendation_service/data/two_tower_reverse_investor.csv \
    --negative-samples 5 \
    --min-interactions 1
```

**Key Points:**
- Uses `StartupInteraction` model for reverse interactions
- Maps startup features as "user" features and user features as "item" features
- Generates hard negatives from recommendation sessions

### 2. ALS Dataset

Generate sparse interaction matrix for ALS reverse model:

```bash
python manage.py generate_als_dataset \
    --use-case startup_developer \
    --output-dir ../recommendation_service/data \
    --min-interactions 1
```

This generates:
- `als_interactions_reverse.npz`: Sparse matrix (Startups × Users)
- `als_reverse_user_mapping.json`: Startup ID to index mapping
- `als_reverse_item_mapping.json`: User ID to index mapping

### 3. Ranker Dataset

Generate ranking dataset with positive/negative pairs:

```bash
python manage.py generate_ranker_dataset \
    --use-case startup_developer \
    --output ../recommendation_service/data/ranker_reverse_developer.csv \
    --neg-ratio 2
```

**Key Points:**
- Extracts positive interactions from `StartupInteraction` (contact, apply_received)
- Generates hard negatives from recommendation sessions
- Includes exposure bias correction weights

## Model Training

### 1. Two-Tower Reverse Model

Train the two-tower reverse model:

```bash
cd recommendation_service
python train_two_tower.py \
    --data data/two_tower_reverse_developer.csv \
    --use-case startup_developer \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --embedding-dim 128 \
    --output-dir models
```

The model will be saved as:
- `models/two_tower_reverse_v1_best.pth`: Best model checkpoint
- `models/two_tower_reverse_v1_latest.pth`: Latest checkpoint
- `models/two_tower_reverse_v1_config.json`: Model configuration
- `models/two_tower_reverse_v1_encoder.pkl`: Feature encoder

**Note:** For reverse models, feature dimensions are swapped:
- Entity (Startup) features use startup feature encoder
- Item (User) features use user feature encoder

### 2. ALS Reverse Model

Train the ALS reverse model using SVD:

```bash
python train_als_reverse.py \
    --data data/als_interactions_reverse.npz \
    --user-mapping data/als_reverse_user_mapping.json \
    --item-mapping data/als_reverse_item_mapping.json \
    --use-case startup_developer \
    --factors 128 \
    --iterations 10 \
    --output-dir models \
    --model-name als_reverse_v1
```

This generates:
- `models/als_reverse_v1_user_factors.npy`: Startup embeddings
- `models/als_reverse_v1_item_factors.npy`: User embeddings
- `models/als_reverse_v1_user_mapping.json`: Startup ID mapping
- `models/als_reverse_v1_item_mapping.json`: User ID mapping
- `models/als_reverse_v1_config.json`: Model metadata

### 3. Ranker Reverse Model

Train the neural ranker for reverse recommendations:

```bash
python train_ranker.py \
    --data data/ranker_reverse_developer.csv \
    --use-case startup_developer \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.001 \
    --output-dir models
```

The model will be saved as:
- `models/ranker_reverse_v1.pth`: Final model
- `models/ranker_reverse_v1_best.pth`: Best validation model

## Model Evaluation

### Two-Tower Model

Evaluate the trained two-tower model:

```bash
python train_two_tower.py \
    --evaluate \
    --model models/two_tower_reverse_v1_best.pth \
    --config models/two_tower_reverse_v1_config.json \
    --data data/two_tower_reverse_developer.csv
```

### ALS Model

The ALS model includes diagnostic metrics during training:
- Explained variance ratio
- Reconstruction MSE

### Ranker Model

The ranker model tracks validation loss during training. Lower validation loss indicates better ranking performance.

## Integration with Flask Service

The reverse models are automatically loaded by the Flask recommendation service:

1. **Model Loading**: Models are loaded at startup in `app.py`
   - `two_tower_reverse_model`: Loaded from `models/two_tower_reverse_v1_best.pth`
   - `ranker_reverse_model`: Loaded from `models/ranker_reverse_v1.pth`
   - `als_reverse_model`: Loaded from ALS reverse artifacts

2. **Endpoint Usage**:
   - `/api/recommendations/developers/for-startup/<startup_id>`: Uses reverse models for `startup_developer` use case
   - `/api/recommendations/investors/for-startup/<startup_id>`: Uses reverse models for `startup_investor` use case

3. **Routing Logic**:
   - Cold start (< 5 interactions): Content-based recommendations
   - Warm (5-19 interactions): Two-Tower Reverse or ALS Reverse
   - Hot (20+ interactions): Two-Tower Reverse (preferred) or ALS Reverse

## Interaction Tracking

Reverse recommendations use the `StartupInteraction` model to track:
- `view`: Startup viewed developer/investor profile
- `click`: Startup clicked on developer/investor profile
- `contact`: Startup contacted developer/investor
- `apply_received`: Developer applied to startup position

These interactions are used for:
- Training dataset generation
- Model evaluation
- Feedback loop for continuous improvement

## Troubleshooting

### Common Issues

1. **No interactions found**: Ensure `StartupInteraction` records exist in the database
2. **Model not loading**: Check that model files exist in the `models/` directory
3. **Feature dimension mismatch**: Verify dataset was generated with correct use case
4. **Low recommendation quality**: Check interaction data quality and model training metrics

### Performance Tips

- Use GPU for two-tower and ranker training
- Increase batch size if memory allows
- Monitor training metrics to prevent overfitting
- Use validation split for early stopping

## Next Steps

1. Monitor model performance in production
2. Collect more interaction data for retraining
3. Experiment with hyperparameters
4. Implement A/B testing for model variants
5. Set up automated retraining pipeline

## References

- Forward recommendation training: See main training documentation
- Dataset generation commands: `backend/api/management/commands/`
- Model architecture: `recommendation_service/engines/`

