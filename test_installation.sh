#!/bin/bash

# Test script to verify conda environment installation
# This script checks that all major dependencies are available

echo "Testing Bioacoustics Web App Installation..."
echo "============================================"

# Function to test conda environment
test_conda_env() {
    echo "1. Testing conda environment..."
    
    if ! conda env list | grep -q "bioacoustics-web-app"; then
        echo "❌ Conda environment 'bioacoustics-web-app' not found"
        echo "   Run ./setup.sh to create the environment"
        return 1
    fi
    
    echo "✓ Conda environment exists"
    
    # Activate environment for testing
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bioacoustics-web-app
    
    if [ $? -eq 0 ]; then
        echo "✓ Environment activation successful"
    else
        echo "❌ Failed to activate environment"
        return 1
    fi
}

# Function to test Python dependencies
test_python_deps() {
    echo "2. Testing Python dependencies..."
    
    # Test core dependencies
    python -c "import fastapi; print('✓ FastAPI:', fastapi.__version__)" 2>/dev/null || echo "❌ FastAPI not available"
    python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)" 2>/dev/null || echo "❌ TensorFlow not available"
    python -c "import librosa; print('✓ Librosa:', librosa.__version__)" 2>/dev/null || echo "❌ Librosa not available"
    python -c "import polars; print('✓ Polars:', polars.__version__)" 2>/dev/null || echo "❌ Polars not available"
    python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "❌ NumPy not available"
    python -c "import soundfile; print('✓ SoundFile:', soundfile.__version__)" 2>/dev/null || echo "❌ SoundFile not available"
    python -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "❌ Matplotlib not available"
    python -c "import sklearn; print('✓ Scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "❌ Scikit-learn not available"
}

# Function to test Node.js and npm
test_node_deps() {
    echo "3. Testing Node.js dependencies..."
    
    if command -v node &> /dev/null; then
        echo "✓ Node.js: $(node --version)"
    else
        echo "❌ Node.js not available"
        return 1
    fi
    
    if command -v npm &> /dev/null; then
        echo "✓ npm: $(npm --version)"
    else
        echo "❌ npm not available"
        return 1
    fi
    
    # Check if React dependencies are installed
    if [ -d "frontend/node_modules" ]; then
        echo "✓ Frontend dependencies installed"
    else
        echo "❌ Frontend dependencies not installed"
        echo "   Run: cd frontend && npm install"
        return 1
    fi
}

# Function to test backend modules
test_backend_modules() {
    echo "4. Testing backend modules..."
    
    # Test if our custom modules can be imported
    cd backend
    python -c "from modules import config; print('✓ Config module')" 2>/dev/null || echo "❌ Config module import failed"
    python -c "from modules import database; print('✓ Database module')" 2>/dev/null || echo "❌ Database module import failed"
    python -c "from modules import utilities; print('✓ Utilities module')" 2>/dev/null || echo "❌ Utilities module import failed"
    python -c "from modules import classifier; print('✓ Classifier module')" 2>/dev/null || echo "❌ Classifier module import failed"
    python -c "from modules import display_web; print('✓ Display web module')" 2>/dev/null || echo "❌ Display web module import failed"
    cd ..
}

# Function to test file structure
test_file_structure() {
    echo "5. Testing file structure..."
    
    # Check required files exist
    files=(
        "environment.yml"
        "install.sh"
        "run_dev.sh"
        "backend/main.py"
        "backend/requirements.txt"
        "frontend/package.json"
        "frontend/src/App.js"
        "frontend/src/components/DatasetBuilder.js"
        "frontend/src/components/ActiveLearning.js"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            echo "✓ $file"
        else
            echo "❌ Missing: $file"
        fi
    done
}

# Function to test if ports are available
test_ports() {
    echo "6. Testing port availability..."
    
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port 8000 is in use (backend may already be running)"
    else
        echo "✓ Port 8000 available"
    fi
    
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port 3000 is in use (frontend may already be running)"
    else
        echo "✓ Port 3000 available"
    fi
}

# Main test function
main() {
    echo "Running installation tests..."
    echo ""
    
    test_conda_env || exit 1
    test_python_deps
    test_node_deps || exit 1
    test_backend_modules
    test_file_structure
    test_ports
    
    echo ""
    echo "============================================"
    echo "Installation test completed!"
    echo ""
    echo "If all tests passed, you can start the application with:"
    echo "  ./run_dev.sh"
    echo ""
    echo "If there were errors, try:"
    echo "  ./setup.sh    # Reinstall environment"
    echo "  conda activate bioacoustics-web-app  # Activate environment"
}

# Run tests
main