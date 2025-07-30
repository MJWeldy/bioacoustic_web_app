#!/bin/bash

# macOS installation script for Bioacoustics Active Learning Web App
# This script handles macOS-specific conda initialization

echo "==================================="
echo "Bioacoustics Active Learning Web App"
echo "macOS Installation Script"
echo "==================================="

# Function to initialize conda on macOS
init_conda_macos() {
    # Try common macOS conda locations
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        echo "✓ Using Miniconda3"
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using Anaconda3"
    elif [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/miniconda3/etc/profile.d/conda.sh
        echo "✓ Using system Miniconda3"
    elif [ -f /usr/local/anaconda3/etc/profile.d/conda.sh ]; then
        source /usr/local/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using system Anaconda3"
    elif [ -f /opt/homebrew/anaconda3/etc/profile.d/conda.sh ]; then
        source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
        echo "✓ Using Homebrew Anaconda3"
    elif command -v conda &> /dev/null; then
        # Try to find conda base
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            echo "✓ Using conda from PATH"
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
        exit 1
    fi
}

# Function to check Node.js on macOS
check_nodejs_macos() {
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        echo "✓ Found Node.js $(node --version) and npm $(npm --version)"
    else
        echo "Node.js and npm are required but not found."
        echo ""
        echo "Install options for macOS:"
        echo "1. Official installer: https://nodejs.org/"
        echo "2. Homebrew: brew install node"
        echo "3. MacPorts: sudo port install nodejs18"
        echo ""
        read -p "Would you like to install Node.js via Homebrew? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if command -v brew &> /dev/null; then
                brew install node
                if command -v node &> /dev/null; then
                    echo "✓ Node.js installed successfully"
                else
                    echo "Failed to install Node.js via Homebrew"
                    exit 1
                fi
            else
                echo "Homebrew not found. Please install Node.js manually."
                exit 1
            fi
        else
            echo "Please install Node.js manually and run this script again."
            exit 1
        fi
    fi
}

# Main installation process
main() {
    echo "Detecting macOS version: $(sw_vers -productVersion)"
    echo "Architecture: $(uname -m)"
    
    # Initialize conda
    init_conda_macos
    
    # Check if active_learning environment exists
    if ! conda env list | grep -q "active_learning"; then
        echo "Error: active_learning environment not found"
        echo "Please run the original notebook setup first"
        exit 1
    fi
    echo "✓ Found active_learning environment"
    
    # Check if target environment already exists
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
    
    # Clone environment
    echo "Cloning active_learning environment to bioacoustics-web-app..."
    conda create --name bioacoustics-web-app --clone active_learning
    
    if [ $? -eq 0 ]; then
        echo "✓ Environment cloned successfully"
    else
        echo "Error: Failed to clone environment"
        exit 1
    fi
    
    # Install web packages
    echo "Installing web framework packages..."
    conda activate bioacoustics-web-app
    pip install \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        python-multipart==0.0.6 \
        pydantic==2.5.0
    
    if [ $? -eq 0 ]; then
        echo "✓ Web framework packages installed"
    else
        echo "Error: Failed to install web packages"
        exit 1
    fi
    
    # Check Node.js
    check_nodejs_macos
    
    # Install React dependencies
    echo "Installing React dependencies..."
    cd frontend
    npm install
    if [ $? -eq 0 ]; then
        echo "✓ React dependencies installed"
        cd ..
    else
        echo "Error: Failed to install React dependencies"
        exit 1
    fi
    
    # Create run script with proper conda initialization
    cat > run_dev.sh << 'EOF'
#!/bin/bash

echo "Starting Bioacoustics Active Learning Web Application..."

# Initialize conda for macOS
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
else
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Error: Could not initialize conda"
        exit 1
    fi
fi

# Check environment
if ! conda env list | grep -q "bioacoustics-web-app"; then
    echo "Error: bioacoustics-web-app environment not found"
    echo "Run ./setup_macos.sh first"
    exit 1
fi

# Activate environment
conda activate bioacoustics-web-app
echo "✓ Activated bioacoustics-web-app environment"

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
cd ..

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting React frontend..."
cd frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"
cd ..

echo ""
echo "Services started successfully!"
echo "Frontend UI: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""
echo "Logs:"
echo "  Backend: logs/backend.log"
echo "  Frontend: logs/frontend.log"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF
    chmod +x run_dev.sh
    
    echo ""
    echo "==================================="
    echo "Setup completed successfully!"
    echo "==================================="
    echo ""
    echo "Created new environment: bioacoustics-web-app"
    echo "Original active_learning environment preserved"
    echo ""
    echo "To start the application:"
    echo "./run_dev.sh"
    echo ""
    echo "Access points:"
    echo "- Frontend: http://localhost:3000"
    echo "- Backend: http://localhost:8000"
}

main