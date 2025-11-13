@echo off
REM Automated Service Startup Script
REM Starts Django backend and Flask recommendation service

echo ======================================================================
echo  STARTING ALL SERVICES
echo ======================================================================
echo.

REM Check if we're in the right directory
if not exist "backend" (
    echo Error: backend directory not found
    echo Please run this script from the project root
    pause
    exit /b 1
)

if not exist "recommendation_service" (
    echo Error: recommendation_service directory not found
    echo Please run this script from the project root
    pause
    exit /b 1
)

echo Starting services in new windows...
echo.

REM Start Django in new window
echo [1/2] Starting Django Backend (port 8000)...
start "Django Backend" cmd /k "cd backend && python manage.py runserver"

REM Wait a bit for Django to start
timeout /t 3 /nobreak > nul

REM Start Flask in new window
echo [2/2] Starting Flask Recommendation Service (port 5000)...
start "Flask Recommendation Service" cmd /k "cd recommendation_service && python app.py"

echo.
echo ======================================================================
echo  SERVICES STARTING...
echo ======================================================================
echo.
echo Two new windows have been opened:
echo   1. Django Backend (http://localhost:8000)
echo   2. Flask Recommendation Service (http://localhost:5000)
echo.
echo Waiting 10 seconds for services to initialize...
timeout /t 10 /nobreak

echo.
echo ======================================================================
echo  RUNNING TESTS
echo ======================================================================
echo.

REM Run tests
python test_complete_recommendation_flow.py

echo.
echo ======================================================================
echo  DONE
echo ======================================================================
echo.
echo Services are running in separate windows.
echo Close those windows to stop the services.
echo.
pause

