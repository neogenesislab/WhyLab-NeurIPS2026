@echo off
setlocal

echo [WhyLab] Starting System...

:: 1. Start Backend API (Port 4001)
start "WhyLab API Server (4001)" cmd /k "python api/main.py"

:: Wait for API to initialize
timeout /t 3 >nul

:: 2. Start Frontend Dashboard (Port 4000)
cd dashboard
start "WhyLab Dashboard (4000)" cmd /k "npm run dev"

echo [WhyLab] Systems are launching!
echo Backend: http://localhost:4001
echo Frontend: http://localhost:4000
echo.
echo Please wait a moment for the dashboard to compile...
