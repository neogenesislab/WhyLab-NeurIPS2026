@echo off
setlocal

echo ========================================================
echo        WhyLab Autonomous System Launcher v1.0
echo ========================================================

:: 1. Setup Environment
echo [1/4] Running System Setup...
python setup.py
if %errorlevel% neq 0 (
    echo [ERROR] Setup failed. Please check the error messages above.
    pause
    exit /b %errorlevel%
)

:: 2. Start Backend API
echo [2/4] Starting Backend API Server (Port 4001)...
start "WhyLab Backend API" cmd /k "uvicorn api.main:app --host 0.0.0.0 --port 4001 --reload"

:: 3. Start Frontend Dashboard
echo [3/4] Starting Frontend Dashboard (Port 4000)...
cd dashboard
start "WhyLab Dashboard" cmd /k "npx next dev -p 4000"
cd ..

:: 4. Open Browser
echo [4/4] Opening Dashboard in Browser...
timeout /t 5 >nul
start http://localhost:4000/WhyLab/dashboard

echo ========================================================
echo        System Launched Successfully!
echo ========================================================
echo - Backend: http://localhost:4001/docs
echo - Dashboard: http://localhost:4000/WhyLab/dashboard
echo.
echo Press any key to start Autopilot Mode (Optional)...
pause >nul

:: Optional: Start Autopilot
echo.
echo [Optional] Starting Autopilot Mode...
curl -X POST http://localhost:4001/system/autopilot/start
echo Autopilot signal sent.
pause
