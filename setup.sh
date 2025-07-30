#!/bin/bash

# Script to set up the web app by cloning the existing active_learning environment
# This preserves the original environment while creating a dedicated web app environment

echo "==================================="
echo "Setting up Bioacoustics Web App"
echo "Cloning active_learning environment"
echo "==================================="

# Function to check conda initialization
init_conda() {
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    else
        # Try to find conda
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "Error: Could not find conda initialization script"
            exit 1
        fi
    fi
}

# Function to check if active_learning environment exists
check_source_env() {
    if ! conda env list | grep -q "active_learning"; then
        echo "Error: active_learning environment not found"
        echo "Please run the original notebook setup first, or try ./install-fallback.sh"
        exit 1
    fi
    echo "✓ Found active_learning environment"
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
        echo "⚠️  Environment creation from yml failed, falling back to cloning approach..."
        
        # Check if active_learning environment exists
        if ! conda env list | grep -q "active_learning"; then
            echo "Error: active_learning environment not found and environment.yml failed"
            echo "Please either:"
            echo "1. Fix conda package issues and try again"
            echo "2. Set up the original active_learning environment first"
            exit 1
        fi
        
        # Clone the environment as fallback
        echo "Cloning active_learning environment to bioacoustics-web-app..."
        conda create --name bioacoustics-web-app --clone active_learning
        
        if [ $? -eq 0 ]; then
            echo "✓ Environment cloned successfully (fallback method)"
            return 0
        else
            echo "Error: Both environment.yml and cloning methods failed"
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

# Function to check Node.js
check_nodejs() {
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        echo "✓ Node.js $(node --version) and npm $(npm --version) found"
    else
        echo "Node.js and npm are required. Installing via conda..."
        init_conda
        conda activate bioacoustics-web-app
        conda install -c conda-forge nodejs -y
        
        if command -v node &> /dev/null && command -v npm &> /dev/null; then
            echo "✓ Node.js installed successfully"
        else
            echo "Error: Please install Node.js manually from https://nodejs.org/"
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
        exit 1
    fi
}

# Function to create run script for existing environment
create_run_script() {
    cat > run_dev.sh << 'EOF'
#!/bin/bash

echo "Starting Bioacoustics Active Learning Web Application..."

# Initialize conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
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
    echo "Run ./setup_existing_env.sh first"
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

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting React frontend..."
cd ../frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

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
    echo "✓ Created run script"
}

# Main setup
main() {
    check_source_env
    check_target_env
    clone_environment
    install_missing_packages
    check_nodejs
    install_react
    create_run_script
    
    echo ""
    echo "==================================="
    echo "Setup completed!"
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