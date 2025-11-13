@echo off
REM Unified Training Script for All Recommendation Models
REM This script generates datasets and trains both ALS and Two-Tower models

echo ======================================================================
echo  UNIFIED MODEL TRAINING PIPELINE
echo ======================================================================
echo.

REM Configuration
set BACKEND_DIR=backend
set REC_SERVICE_DIR=recommendation_service

REM Step 1: Generate Two-Tower Dataset
echo [1/4] Generating Two-Tower dataset...
cd %BACKEND_DIR%
python manage.py generate_two_tower_dataset
if %errorlevel% neq 0 (
    echo Error: Two-Tower dataset generation failed
    exit /b 1
)
cd ..
echo [OK] Two-Tower dataset generated
echo.

REM Step 2: Generate ALS Dataset
echo [2/4] Generating ALS dataset...
cd %BACKEND_DIR%
python manage.py generate_als_dataset
if %errorlevel% neq 0 (
    echo Error: ALS dataset generation failed
    exit /b 1
)
cd ..
echo [OK] ALS dataset generated
echo.

REM Step 3: Train ALS Model
echo [3/4] Training ALS model...
cd %REC_SERVICE_DIR%
python train_als.py --data data/als_interactions.npz --factors 128 --iterations 20
if %errorlevel% neq 0 (
    echo Error: ALS training failed
    exit /b 1
)
cd ..
echo [OK] ALS model trained
echo.

REM Step 4: Train Two-Tower Model
echo [4/5] Training Two-Tower model...
cd %REC_SERVICE_DIR%
python train_standalone.py --data data/two_tower_train.csv --epochs 10 --batch-size 256
if %errorlevel% neq 0 (
    echo Error: Two-Tower training failed
    exit /b 1
)
cd ..
echo [OK] Two-Tower model trained
echo.

REM Step 5: Train Ranker Model (Optional)
echo [5/5] Training Ranker model...
cd %BACKEND_DIR%
python manage.py generate_ranker_dataset --output ../recommendation_service/data/ranker_train.csv
if %errorlevel% neq 0 (
    echo Warning: Ranker dataset generation failed (may be due to insufficient data)
    echo Skipping ranker training - rule-based ranker will be used
    cd ..
    goto :skip_ranker
)
cd ..%REC_SERVICE_DIR%
python train_ranker.py --data data/ranker_train.csv --epochs 20 --batch-size 128
if %errorlevel% neq 0 (
    echo Warning: Ranker training failed
    echo Rule-based ranker will be used
)
cd ..
echo [OK] Ranker model trained
echo.
:skip_ranker

echo ======================================================================
echo  ALL MODELS TRAINED SUCCESSFULLY!
echo ======================================================================
echo.
echo Models saved to: %REC_SERVICE_DIR%\models\
echo.
echo Next steps:
echo   1. Start Flask service: cd %REC_SERVICE_DIR% ^&^& python app.py
echo   2. Models will load automatically on startup
echo   3. Smart routing will be active:
echo      - Cold start (^< 5 interactions) - Content-Based
echo      - Warm (5-19 interactions) - ALS
echo      - Hot (20+ interactions) - Ensemble (ALS + Two-Tower)
echo   4. Ranker will reorder all personalized recommendations
echo.


