#!/bin/bash

# macOS setup script for Bioacoustics Web App
# Works for both first-time users and users with existing active_learning environment
# Handles macOS-specific conda initialization and package management

echo "==================================="
echo "Setting up Bioacoustics Web App"
echo "macOS Installation Script"
echo "==================================="

# Function to initialize conda on macOS (handles multiple installation paths)
init_conda() {
    # Try common macOS conda locations
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        echo "✓ Using Miniconda3 (~/miniconda3)"
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using Anaconda3 (~/anaconda3)"
    elif [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/miniconda3/etc/profile.d/conda.sh
        echo "✓ Using system Miniconda3"
    elif [ -f /usr/local/anaconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using system Anaconda3"
    elif [ -f /opt/homebrew/anaconda3/etc/profile.d/conda.sh ]; then
        source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using Homebrew Anaconda3"
    elif [ -f /opt/homebrew/miniconda3/etc/profile.d/conda.sh ]; then
        source /opt/homebrew/miniconda3/etc/profile.d/conda.sh
        echo "✓ Using Homebrew Miniconda3"
    elif command -v conda &> /dev/null; then
        # Try to find conda base dynamically
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            echo "✓ Using conda from PATH ($CONDA_BASE)"
        else
            echo "Error: Could not initialize conda"
            echo "Please install conda and ensure it's in your PATH"
            echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
    else
        echo "Error: conda not found"
        echo "Please install Anaconda or Miniconda:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        echo ""
        echo "For macOS, recommended installation methods:"
        echo "1. Official installer: https://docs.conda.io/en/latest/miniconda.html"
        echo "2. Homebrew: brew install miniconda"
        exit 1
    fi
}

# Function to check if active_learning environment exists (optional for cloning)
check_source_env() {
    if conda env list | grep -q "active_learning"; then
        echo "✓ Found existing active_learning environment"
        return 0
    else
        echo "ℹ️  No active_learning environment found (will use environment.yml)"
        return 1
    fi
}

# Function to check if target environment already exists
check_target_env() {
    if conda env list | grep -q "bioacoustics-web-app"; then
        echo "Warning: bioacoustics-web-app environment already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n bioacoustics-web-app -y
        else
            echo "Installation cancelled."
            exit 0
        fi
    fi
}

# Function to create environment using environment.yml with fallback to cloning
create_environment() {
    echo "Creating bioacoustics-web-app environment..."
    
    init_conda
    
    # First try to create from environment.yml
    echo "Attempting to create environment from environment.yml..."
    if conda env create -f environment.yml; then
        echo "✓ Environment created from environment.yml"
        return 0
    else
        echo "⚠️  Environment creation from environment.yml failed"
        
        # Check if active_learning environment exists for cloning fallback
        if conda env list | grep -q "active_learning"; then
            echo "Falling back to cloning active_learning environment..."
            conda create --name bioacoustics-web-app --clone active_learning
            
            if [ $? -eq 0 ]; then
                echo "✓ Environment cloned successfully (fallback method)"
                return 0
            else
                echo "Error: Both environment.yml and cloning methods failed"
                exit 1
            fi
        else
            echo "Error: Environment creation from environment.yml failed and no active_learning environment to clone from"
            echo "Please either:"
            echo "1. Fix conda package conflicts and try running setup_macos.sh again"
            echo "2. Install conda packages manually"
            echo "3. Set up an active_learning environment first, then re-run setup_macos.sh"
            exit 1
        fi
    fi
}

# Function to install any missing packages (if environment was cloned)
install_missing_packages() {
    echo "Checking for missing web framework packages..."
    
    init_conda
    conda activate bioacoustics-web-app
    
    # Check if FastAPI is already installed
    if python -c "import fastapi" 2>/dev/null; then
        echo "✓ Web framework packages already available"
        return 0
    fi
    
    echo "Installing missing web framework packages..."
    # Install FastAPI and related packages
    pip install \
        fastapi>=0.104.0 \
        uvicorn[standard]>=0.24.0 \
        python-multipart>=0.0.6 \
        pydantic>=2.5.0
    
    if [ $? -eq 0 ]; then
        echo "✓ Web framework packages installed"
    else
        echo "Error: Failed to install web packages"
        exit 1
    fi
}

# Function to check Node.js on macOS
check_nodejs() {
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        echo "✓ Node.js $(node --version) and npm $(npm --version) found"
    else
        echo "Node.js and npm are required but not found."
        echo ""
        echo "Install options for macOS:"
        echo "1. Official installer: https://nodejs.org/"
        echo "2. Homebrew: brew install node"
        echo "3. MacPorts: sudo port install nodejs18"
        echo "4. Via conda: conda install -c conda-forge nodejs"
        echo ""
        read -p "Would you like to install Node.js via conda? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            # Try conda installation first
            init_conda
            conda activate bioacoustics-web-app
            conda install -c conda-forge nodejs -y
            
            if command -v node &> /dev/null && command -v npm &> /dev/null; then
                echo "✓ Node.js installed via conda"
            else
                echo "Conda installation failed. Trying Homebrew..."
                if command -v brew &> /dev/null; then
                    brew install node
                    if command -v node &> /dev/null; then
                        echo "✓ Node.js installed via Homebrew"
                    else
                        echo "Error: Failed to install Node.js via Homebrew"
                        exit 1
                    fi
                else
                    echo "Error: Neither conda nor Homebrew installation worked"
                    echo "Please install Node.js manually from https://nodejs.org/"
                    exit 1
                fi
            fi
        else
            echo "Please install Node.js manually and run this script again."
            exit 1
        fi
    fi
}

# Function to install React dependencies
install_react() {
    echo "Installing React dependencies..."
    cd frontend
    npm install
    if [ $? -eq 0 ]; then
        echo "✓ React dependencies installed"
        cd ..
    else
        echo "Error: Failed to install React dependencies"
        echo "Try running: cd frontend && npm install"
        exit 1
    fi
}

# Function to create enhanced run script with health checks
create_run_script() {
    cat > run_dev.sh << 'EOF'
#!/bin/bash

echo "Starting Bioacoustics Active Learning Web Application..."

# Initialize conda for macOS (multiple path support)
init_conda_macos() {
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    elif [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/miniconda3/etc/profile.d/conda.sh
    elif [ -f /usr/local/anaconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/anaconda3/etc/profile.d/conda.sh
    elif [ -f /opt/homebrew/anaconda3/etc/profile.d/conda.sh ]; then
        source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
    elif [ -f /opt/homebrew/miniconda3/etc/profile.d/conda.sh ]; then
        source /opt/homebrew/miniconda3/etc/profile.d/conda.sh
    else
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "Error: Could not initialize conda"
            exit 1
        fi
    fi
}

init_conda_macos

# Check environment exists
if ! conda env list | grep -q "bioacoustics-web-app"; then
    echo "Error: bioacoustics-web-app environment not found"
    echo "Run ./setup_macos.sh first"
    exit 1
fi

# Activate environment
conda activate bioacoustics-web-app
echo "✓ Activated bioacoustics-web-app environment"

# Verify key packages are available
echo "Verifying environment..."
python -c "import fastapi, uvicorn, tensorflow, librosa, polars" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required Python packages"
    echo "Run ./setup_macos.sh to fix the environment"
    exit 1
fi
echo "✓ Python packages verified"

# Create logs directory
mkdir -p logs

# Cleanup function
cleanup() {
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting FastAPI backend..."
cd backend
python main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to start and test connectivity
echo "Waiting for backend to initialize..."
for i in {1..30}; do
    sleep 2
    if curl -s http://localhost:8000/api/dataset/status > /dev/null; then
        echo "✓ Backend is responding"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Backend failed to start properly"
        echo "Check logs/backend.log for details"
        exit 1
    fi
done

# Check and install frontend dependencies if needed
echo "Checking frontend dependencies..."
cd ../frontend
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
    echo "Frontend dependencies not found. Installing..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Failed to install frontend dependencies"
        exit 1
    fi
    echo "✓ Frontend dependencies installed"
else
    echo "✓ Frontend dependencies found"
fi

# Start frontend
echo "Starting React frontend..."
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "🎉 Services started successfully!"
echo "=========================================="
echo ""
echo "Access your application:"
echo "  • Frontend UI:    http://localhost:3000"
echo "  • Backend API:    http://localhost:8000"
echo "  • API Docs:       http://localhost:8000/docs"
echo ""
echo "Logs available at:"
echo "  • Backend:        logs/backend.log"
echo "  • Frontend:       logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF
    chmod +x run_dev.sh
    echo "✓ Created enhanced run script with health checks"
}

# Main setup function
main() {
    echo "Detecting macOS environment..."
    echo "  macOS version: $(sw_vers -productVersion)"
    echo "  Architecture: $(uname -m)"
    echo ""
    
    # Run setup steps
    init_conda
    check_source_env  # This just provides info now, doesn't exit
    check_target_env
    create_environment
    install_missing_packages
    check_nodejs
    install_react
    create_run_script
    
    echo ""
    echo "==================================="
    echo "🎉 Setup completed successfully!"
    echo "==================================="
    echo ""
    echo "Environment details:"
    echo "  • Created: bioacoustics-web-app"
    echo "  • Original active_learning environment preserved"
    echo "  • Platform: macOS $(sw_vers -productVersion)"
    echo ""
    echo "To start the application:"
    echo "  ./run_dev.sh"
    echo ""
    echo "Access points:"
    echo "  • Frontend: http://localhost:3000"
    echo "  • Backend:  http://localhost:8000"
    echo "  • API Docs: http://localhost:8000/docs"
    echo ""
    echo "For testing and validation:"
    echo "  ./health_check.sh     # System health check"
    echo "  cd test_audio && ./quick_test.sh  # Quick functionality test"
}

main