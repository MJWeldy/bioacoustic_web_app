@echo off
echo Starting Bioacoustics Active Learning Web Application...

REM Check environment
conda env list | findstr "bioacoustics-web-app" >nul
if %errorlevel% neq 0 (
    echo Error: bioacoustics-web-app environment not found
    echo Run setup.bat first
    pause
    exit /b 1
)

REM Activate environment
call conda activate bioacoustics-web-app
echo ✓ Activated bioacoustics-web-app environment

REM Create logs directory
if not exist logs mkdir logs

echo Starting FastAPI backend...
start "Bioacoustics Backend" cmd /c "cd backend && python main.py > ../logs/backend.log 2>&1"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Checking frontend dependencies...
cd frontend
if not exist node_modules (
    echo Frontend dependencies not found. Installing...
    call npm install
    if %errorlevel% neq 0 (
        echo Failed to install frontend dependencies
        pause
        exit /b 1
    )
    echo ✓ Frontend dependencies installed
) else (
    echo ✓ Frontend dependencies found
)
cd ..

echo Starting React frontend...
start "Bioacoustics Frontend" cmd /c "cd frontend && npm start > ../logs/frontend.log 2>&1"

echo.
echo Services started successfully!
echo Frontend UI: http://localhost:3000
echo Backend API: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
echo Logs:
echo   Backend: logs\backend.log
echo   Frontend: logs\frontend.log
echo.
echo Press any key to stop services...
pause >nul

echo Stopping services...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Bioacoustics Backend*" >nul 2>&1
taskkill /f /im node.exe /fi "WINDOWTITLE eq Bioacoustics Frontend*" >nul 2>&1
echo Services stopped.