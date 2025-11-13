@echo off
REM Quick script to train Two-Tower model end-to-end (Windows)

echo ================================================
echo Two-Tower Model Training Pipeline
echo ================================================
echo.

REM Step 1: Generate dataset
echo [Step 1/4] Generating training dataset...
cd backend
python manage.py generate_two_tower_dataset --output ../recommendation_service/data/two_tower_train.csv --negative-samples 2 --use-case developer_startup --min-interactions 1

if errorlevel 1 (
    echo Error: Dataset generation failed
    exit /b 1
)

REM Step 2: Train model
echo.
echo [Step 2/4] Training Two-Tower model...
cd ..\recommendation_service
python train_two_tower.py --data data/two_tower_train.csv --output-dir models --model-name two_tower_v1 --epochs 50 --batch-size 256 --lr 0.001 --embedding-dim 128

if errorlevel 1 (
    echo Error: Model training failed
    exit /b 1
)

REM Step 3: Evaluate model
echo.
echo [Step 3/4] Evaluating trained model...
python train_two_tower.py --evaluate --model models/two_tower_v1.pth --data data/two_tower_train.csv

if errorlevel 1 (
    echo Error: Model evaluation failed
    exit /b 1
)

REM Step 4: Test integration
echo.
echo [Step 4/4] Testing integration...
python test_two_tower_integration.py

if errorlevel 1 (
    echo Warning: Some integration tests failed (check output above)
)

echo.
echo ================================================
echo Training Complete! 🎉
echo ================================================
echo.
echo Model saved to: recommendation_service\models\two_tower_v1.pth
echo Encoder saved to: recommendation_service\models\two_tower_v1_encoder.pkl
echo.
echo To enable in production, update your Flask app initialization:
echo   rec_service = RecommendationService(db, enable_two_tower=True)
echo.
echo Start Flask service:
echo   cd recommendation_service ^&^& python app.py
echo.

