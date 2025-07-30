@echo off
REM Windows installation script for Bioacoustics Active Learning Web App
REM This script uses environment.yml with fallback to cloning active_learning environment

echo ===================================
echo Bioacoustics Active Learning Web App
echo Windows Installation Script
echo ===================================

REM Function to check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo Make sure to add conda to your PATH during installation
    pause
    exit /b 1
)
echo ✓ Found conda

REM Check if active_learning environment exists
conda env list | findstr "active_learning" >nul
if %errorlevel% neq 0 (
    echo Error: active_learning environment not found
    echo Please run the original notebook setup first
    pause
    exit /b 1
)
echo ✓ Found active_learning environment

REM Check if target environment already exists
conda env list | findstr "bioacoustics-web-app" >nul
if %errorlevel% equ 0 (
    echo Warning: bioacoustics-web-app environment already exists
    set /p choice="Do you want to remove it and recreate? (y/N): "
    if /i "%choice%"=="y" (
        echo Removing existing environment...
        call conda env remove -n bioacoustics-web-app -y
    ) else (
        echo Installation cancelled.
        pause
        exit /b 0
    )
)

REM Create environment using environment.yml with fallback to cloning
echo Creating bioacoustics-web-app environment...

REM First try to create from environment.yml
echo Attempting to create environment from environment.yml...
call conda env create -f environment.yml
if %errorlevel% equ 0 (
    echo ✓ Environment created from environment.yml
) else (
    echo ⚠️  Environment creation from yml failed, falling back to cloning approach...
    
    REM Check if active_learning environment exists
    conda env list ^| findstr "active_learning" ^>nul
    if %errorlevel% neq 0 (
        echo Error: active_learning environment not found and environment.yml failed
        echo Please either:
        echo 1. Fix conda package issues and try again
        echo 2. Set up the original active_learning environment first
        pause
        exit /b 1
    )
    
    REM Clone the environment as fallback
    echo Cloning active_learning environment to bioacoustics-web-app...
    call conda create --name bioacoustics-web-app --clone active_learning
    if %errorlevel% neq 0 (
        echo Error: Both environment.yml and cloning methods failed
        pause
        exit /b 1
    )
    echo ✓ Environment cloned successfully ^(fallback method^)
)

REM Install any missing web packages (if environment was cloned)
echo Checking for missing web framework packages...
call conda activate bioacoustics-web-app

REM Check if FastAPI is already installed
python -c "import fastapi" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Web framework packages already available
) else (
    echo Installing missing web framework packages...
    pip install fastapi^>=0.104.0 uvicorn[standard]^>=0.24.0 python-multipart^>=0.0.6 pydantic^>=2.5.0
    if %errorlevel% neq 0 (
        echo Error: Failed to install web packages
        pause
        exit /b 1
    )
    echo ✓ Web framework packages installed
)

REM Check for Node.js and npm
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is required but not found.
    echo Please install Node.js from https://nodejs.org/
    echo Then run this script again.
    pause
    exit /b 1
)
echo ✓ Found Node.js

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo npm is required but not found.
    echo Please install Node.js from https://nodejs.org/
    echo Then run this script again.
    pause
    exit /b 1
)
echo ✓ Found npm

REM Install React dependencies
echo Installing React dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Error: Failed to install React dependencies
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ React dependencies installed

REM Create Windows run script
echo Creating Windows run script...
(
echo @echo off
echo echo Starting Bioacoustics Active Learning Web Application...
echo.
echo REM Check environment
echo conda env list ^| findstr "bioacoustics-web-app" ^>nul
echo if %%errorlevel%% neq 0 ^(
echo     echo Error: bioacoustics-web-app environment not found
echo     echo Run setup.bat first
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Activate environment
echo call conda activate bioacoustics-web-app
echo echo ✓ Activated bioacoustics-web-app environment
echo.
echo REM Create logs directory
echo if not exist logs mkdir logs
echo.
echo echo Starting FastAPI backend...
echo start "Backend" cmd /c "cd backend && python main.py > ../logs/backend.log 2>&1"
echo.
echo echo Waiting for backend to start...
echo timeout /t 5 /nobreak ^>nul
echo.
echo echo Starting React frontend...
echo start "Frontend" cmd /c "cd frontend && npm start > ../logs/frontend.log 2>&1"
echo.
echo echo Services started successfully!
echo echo Frontend UI: http://localhost:3000
echo echo Backend API: http://localhost:8000
echo echo API docs: http://localhost:8000/docs
echo echo.
echo echo Press any key to stop services...
echo pause ^>nul
echo.
echo echo Stopping services...
echo taskkill /f /im python.exe /fi "WINDOWTITLE eq Backend*" ^>nul 2^>^&1
echo taskkill /f /im node.exe /fi "WINDOWTITLE eq Frontend*" ^>nul 2^>^&1
) > run_dev.bat

echo ✓ Created Windows run script

echo.
echo ===================================
echo Setup completed successfully!
echo ===================================
echo.
echo Created new environment: bioacoustics-web-app
echo Original active_learning environment preserved
echo.
echo To start the application:
echo   run_dev.bat
echo.
echo Access points:
echo - Frontend: http://localhost:3000
echo - Backend: http://localhost:8000
pause