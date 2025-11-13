#!/bin/bash
# Unified Training Script for All Recommendation Models
# This script generates datasets and trains both ALS and Two-Tower models

echo "======================================================================"
echo " UNIFIED MODEL TRAINING PIPELINE"
echo "======================================================================"
echo ""

# Configuration
BACKEND_DIR="backend"
REC_SERVICE_DIR="recommendation_service"

# Step 1: Generate Two-Tower Dataset
echo "[1/4] Generating Two-Tower dataset..."
cd "$BACKEND_DIR" || exit 1
python manage.py generate_two_tower_dataset
if [ $? -ne 0 ]; then
    echo "Error: Two-Tower dataset generation failed"
    exit 1
fi
cd ..
echo "✓ Two-Tower dataset generated"
echo ""

# Step 2: Generate ALS Dataset
echo "[2/4] Generating ALS dataset..."
cd "$BACKEND_DIR" || exit 1
python manage.py generate_als_dataset
if [ $? -ne 0 ]; then
    echo "Error: ALS dataset generation failed"
    exit 1
fi
cd ..
echo "✓ ALS dataset generated"
echo ""

# Step 3: Train ALS Model
echo "[3/4] Training ALS model..."
cd "$REC_SERVICE_DIR" || exit 1
python train_als.py --data data/als_interactions.npz --factors 128 --iterations 20
if [ $? -ne 0 ]; then
    echo "Error: ALS training failed"
    exit 1
fi
cd ..
echo "✓ ALS model trained"
echo ""

# Step 4: Train Two-Tower Model
echo "[4/4] Training Two-Tower model..."
cd "$REC_SERVICE_DIR" || exit 1
python train_standalone.py --data data/two_tower_train.csv --epochs 10 --batch-size 256
if [ $? -ne 0 ]; then
    echo "Error: Two-Tower training failed"
    exit 1
fi
cd ..
echo "✓ Two-Tower model trained"
echo ""

echo "======================================================================"
echo " ALL MODELS TRAINED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Models saved to: $REC_SERVICE_DIR/models/"
echo ""
echo "Next steps:"
echo "  1. Start Flask service: cd $REC_SERVICE_DIR && python app.py"
echo "  2. Models will load automatically on startup"
echo "  3. Smart routing will be active:"
echo "     - Cold start (< 5 interactions) → Content-Based"
echo "     - Warm (5-19 interactions) → ALS"
echo "     - Hot (20+ interactions) → Ensemble (ALS + Two-Tower)"
echo ""


