#!/bin/bash
# Quick script to train Two-Tower model end-to-end

echo "================================================"
echo "Two-Tower Model Training Pipeline"
echo "================================================"
echo ""

# Step 1: Generate dataset
echo "[Step 1/4] Generating training dataset..."
cd backend || exit 1
python manage.py generate_two_tower_dataset \
    --output ../recommendation_service/data/two_tower_train.csv \
    --negative-samples 2 \
    --use-case developer_startup \
    --min-interactions 1

if [ $? -ne 0 ]; then
    echo "Error: Dataset generation failed"
    exit 1
fi

# Step 2: Train model
echo ""
echo "[Step 2/4] Training Two-Tower model..."
cd ../recommendation_service || exit 1
python train_two_tower.py \
    --data data/two_tower_train.csv \
    --output-dir models \
    --model-name two_tower_v1 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --embedding-dim 128

if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

# Step 3: Evaluate model
echo ""
echo "[Step 3/4] Evaluating trained model..."
python train_two_tower.py \
    --evaluate \
    --model models/two_tower_v1.pth \
    --data data/two_tower_train.csv

if [ $? -ne 0 ]; then
    echo "Error: Model evaluation failed"
    exit 1
fi

# Step 4: Test integration
echo ""
echo "[Step 4/4] Testing integration..."
python test_two_tower_integration.py

if [ $? -ne 0 ]; then
    echo "Warning: Some integration tests failed (check output above)"
fi

echo ""
echo "================================================"
echo "Training Complete! 🎉"
echo "================================================"
echo ""
echo "Model saved to: recommendation_service/models/two_tower_v1.pth"
echo "Encoder saved to: recommendation_service/models/two_tower_v1_encoder.pkl"
echo ""
echo "To enable in production, update your Flask app initialization:"
echo "  rec_service = RecommendationService(db, enable_two_tower=True)"
echo ""
echo "Start Flask service:"
echo "  cd recommendation_service && python app.py"
echo ""

