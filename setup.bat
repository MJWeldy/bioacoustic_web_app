@echo off
REM Windows setup script for Bioacoustics Web App  
REM Works for both first-time users and users with existing active_learning environment
REM Handles Windows-specific conda initialization and package management

echo ===================================
echo Setting up Bioacoustics Web App
echo Windows Installation Script
echo ===================================

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo Make sure to add conda to your PATH during installation
    echo.
    echo For Windows, recommended installation:
    echo 1. Download from: https://docs.conda.io/en/latest/miniconda.html
    echo 2. Run installer and check "Add to PATH" option
    echo 3. Restart Command Prompt and try again
    pause
    exit /b 1
)
echo ✓ Found conda

REM Initialize conda for batch script
call conda info --envs >nul 2>&1
if %errorlevel% neq 0 (
    echo Initializing conda for Windows batch scripts...
    call conda init cmd.exe
    echo Please restart Command Prompt and try again
    pause
    exit /b 1
)

REM Check if active_learning environment exists (optional for cloning)
conda env list | findstr "active_learning" >nul
if %errorlevel% neq 0 (
    echo ℹ️  No active_learning environment found ^(will use environment.yml^)
) else (
    echo ✓ Found existing active_learning environment
)

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
    goto :install_missing_packages
) else (
    echo ⚠️  Environment creation from environment.yml failed
    
    REM Check if active_learning environment exists for cloning fallback
    conda env list | findstr "active_learning" >nul
    if %errorlevel% neq 0 (
        echo Error: Environment creation from environment.yml failed and no active_learning environment to clone from
        echo Please either:
        echo 1. Fix conda package conflicts and try running setup.bat again
        echo 2. Install conda packages manually
        echo 3. Set up an active_learning environment first, then re-run setup.bat
        pause
        exit /b 1
    )
    
    echo Falling back to cloning active_learning environment...
    call conda create --name bioacoustics-web-app --clone active_learning
    if %errorlevel% neq 0 (
        echo Error: Both environment.yml and cloning methods failed
        pause
        exit /b 1
    )
    echo ✓ Environment cloned successfully ^(fallback method^)
)

:install_missing_packages
REM Install any missing web packages (if environment was cloned)
echo Checking for missing web framework packages...
call conda activate bioacoustics-web-app

REM Check if FastAPI is already installed
python -c "import fastapi" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Web framework packages already available
) else (
    echo Installing missing web framework packages...
    pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 python-multipart>=0.0.6 pydantic>=2.5.0
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
    echo Node.js and npm are required but not found.
    echo.
    echo Install options for Windows:
    echo 1. Official installer: https://nodejs.org/
    echo 2. Chocolatey: choco install nodejs
    echo 3. Scoop: scoop install nodejs
    echo 4. Via conda: conda install -c conda-forge nodejs
    echo.
    set /p choice="Would you like to install Node.js via conda? (Y/n): "
    if /i not "%choice%"=="n" (
        echo Installing Node.js via conda...
        call conda install -c conda-forge nodejs -y
        if %errorlevel% neq 0 (
            echo Error: Failed to install Node.js via conda
            echo Please install Node.js manually from https://nodejs.org/
            pause
            exit /b 1
        )
        echo ✓ Node.js installed via conda
    ) else (
        echo Please install Node.js manually and run this script again.
        pause
        exit /b 1
    )
) else (
    echo ✓ Found Node.js
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo npm is required but not found.
    echo Please install Node.js from https://nodejs.org/
    echo Then run this script again.
    pause
    exit /b 1
)

REM Get Node.js and npm versions
for /f "tokens=*" %%i in ('node --version') do set node_version=%%i
for /f "tokens=*" %%i in ('npm --version') do set npm_version=%%i
echo ✓ Node.js %node_version% and npm %npm_version% found

REM Install React dependencies
echo Installing React dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Error: Failed to install React dependencies
    echo Try running: cd frontend ^&^& npm install
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ React dependencies installed

REM Create enhanced Windows run script with health checks
echo Creating enhanced Windows run script...
(
echo @echo off
echo echo Starting Bioacoustics Active Learning Web Application...
echo.
echo REM Check environment exists
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
echo REM Verify key packages are available
echo echo Verifying environment...
echo python -c "import fastapi, uvicorn, tensorflow, librosa, polars" ^>nul 2^>^&1
echo if %%errorlevel%% neq 0 ^(
echo     echo Error: Missing required Python packages
echo     echo Run setup.bat to fix the environment
echo     pause
echo     exit /b 1
echo ^)
echo echo ✓ Python packages verified
echo.
echo REM Create logs directory
echo if not exist logs mkdir logs
echo.
echo REM Kill any existing processes on the ports
echo echo Checking for existing processes...
echo netstat -ano ^| findstr ":3000" ^>nul 2^>^&1
echo if %%errorlevel%% equ 0 ^(
echo     echo Warning: Port 3000 is in use. Attempting to free it...
echo     for /f "tokens=5" %%p in ^('netstat -ano ^| findstr ":3000"'^) do taskkill /pid %%p /f ^>nul 2^>^&1
echo ^)
echo netstat -ano ^| findstr ":8000" ^>nul 2^>^&1
echo if %%errorlevel%% equ 0 ^(
echo     echo Warning: Port 8000 is in use. Attempting to free it...
echo     for /f "tokens=5" %%p in ^('netstat -ano ^| findstr ":8000"'^) do taskkill /pid %%p /f ^>nul 2^>^&1
echo ^)
echo.
echo echo Starting FastAPI backend...
echo start "Bioacoustics-Backend" cmd /c "cd backend ^&^& python main.py ^> ../logs/backend.log 2^>^&1"
echo.
echo echo Waiting for backend to initialize...
echo timeout /t 10 /nobreak ^>nul
echo.
echo REM Test backend connectivity
echo echo Testing backend connectivity...
echo powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:8000/api/dataset/status' -TimeoutSec 5 | Out-Null; Write-Host '✓ Backend is responding' } catch { Write-Host 'Warning: Backend may still be starting up' }"
echo.
echo REM Check and install frontend dependencies if needed
echo echo Checking frontend dependencies...
echo cd frontend
echo if not exist node_modules ^(
echo     echo Frontend dependencies not found. Installing...
echo     call npm install
echo     if %%errorlevel%% neq 0 ^(
echo         echo Failed to install frontend dependencies
echo         cd ..
echo         pause
echo         exit /b 1
echo     ^)
echo     echo ✓ Frontend dependencies installed
echo ^) else ^(
echo     echo ✓ Frontend dependencies found
echo ^)
echo.
echo echo Starting React frontend...
echo start "Bioacoustics-Frontend" cmd /c "npm start ^> ../logs/frontend.log 2^>^&1"
echo cd ..
echo.
echo echo ==========================================
echo echo 🎉 Services started successfully!
echo echo ==========================================
echo echo.
echo echo Access your application:
echo echo   • Frontend UI:    http://localhost:3000
echo echo   • Backend API:    http://localhost:8000  
echo echo   • API Docs:       http://localhost:8000/docs
echo echo.
echo echo Logs available at:
echo echo   • Backend:        logs\backend.log
echo echo   • Frontend:       logs\frontend.log
echo echo.
echo echo Press any key to stop all services...
echo pause ^>nul
echo.
echo echo Stopping services...
echo taskkill /f /fi "WINDOWTITLE eq Bioacoustics-Backend*" ^>nul 2^>^&1
echo taskkill /f /fi "WINDOWTITLE eq Bioacoustics-Frontend*" ^>nul 2^>^&1
echo echo Services stopped.
) > run_dev.bat

echo ✓ Created enhanced run script with health checks

echo.
echo ===================================
echo 🎉 Setup completed successfully!
echo ===================================
echo.
echo Environment details:
echo   • Created: bioacoustics-web-app
echo   • Original active_learning environment preserved
echo   • Platform: Windows %OS%
echo.
echo To start the application:
echo   run_dev.bat
echo.
echo Access points:
echo   • Frontend: http://localhost:3000
echo   • Backend:  http://localhost:8000  
echo   • API Docs: http://localhost:8000/docs
echo.
echo For testing and validation:
echo   health_check.sh     # System health check ^(if using WSL/Git Bash^)
echo   cd test_audio ^&^& quick_test.sh  # Quick functionality test ^(if using WSL/Git Bash^)
echo.
pause