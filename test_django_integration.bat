@echo off
REM Test script for Django-Flask integration
REM Make sure both Django (port 8000) and Flask (port 5000) are running!

echo ==============================================
echo Testing Django-Flask Integration
echo ==============================================
echo.

REM Configuration
set DJANGO_URL=http://localhost:8000
set FLASK_URL=http://localhost:5000

echo Checking services...
echo.

REM Check Flask
curl -s %FLASK_URL%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Flask service is running on %FLASK_URL%
) else (
    echo [ERROR] Flask service is NOT running!
    echo    Start it with: cd recommendation_service ^&^& python app.py
    exit /b 1
)

REM Check Django
curl -s %DJANGO_URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Django service is running on %DJANGO_URL%
) else (
    echo [ERROR] Django service is NOT running!
    echo    Start it with: cd backend ^&^& python manage.py runserver
    exit /b 1
)

echo.
echo ==============================================
echo New Django Endpoints Available:
echo ==============================================
echo.
echo 1. GET /api/recommendations/personalized/startups
echo    - Uses Two-Tower model for warm users (5+ interactions)
echo.
echo 2. GET /api/recommendations/personalized/developers/^<startup_id^>
echo    - Get developer recommendations for a startup
echo.
echo 3. GET /api/recommendations/personalized/investors/^<startup_id^>
echo    - Get investor recommendations for a startup
echo.

echo ==============================================
echo To Test (You need an auth token):
echo ==============================================
echo.
echo Step 1: Login to get token
echo.
echo curl -X POST http://localhost:8000/auth/login ^
echo   -H "Content-Type: application/json" ^
echo   -d "{\"username\": \"YOUR_USERNAME\", \"password\": \"YOUR_PASSWORD\"}"
echo.
echo Step 2: Use token to get recommendations
echo.
echo curl -H "Authorization: Bearer YOUR_TOKEN" ^
echo   "http://localhost:8000/api/recommendations/personalized/startups?limit=10"
echo.
echo ==============================================
echo Integration Complete!
echo ==============================================
echo.
echo Your Two-Tower model is now accessible through Django!
echo.

