#!/bin/bash

# Simple test script to verify setup works
echo "Testing Bioacoustics Web App Setup..."

# Test 1: Check conda
if command -v conda &> /dev/null; then
    echo "✓ Conda available"
else
    echo "✗ Conda not found"
    exit 1
fi

# Test 2: Check if environment exists
if conda env list | grep -q "bioacoustics-web-app"; then
    echo "✓ bioacoustics-web-app environment found"
else
    echo "✗ bioacoustics-web-app environment not found"
    echo "Run ./setup.sh first"
    exit 1
fi

# Test 3: Initialize conda and activate environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "✗ Could not initialize conda"
        exit 1
    fi
fi

conda activate bioacoustics-web-app
echo "✓ Activated bioacoustics-web-app environment"

# Test 4: Check Python packages
echo "Testing Python packages..."
python -c "import fastapi; print('✓ FastAPI available')" || echo "✗ FastAPI not available"
python -c "import uvicorn; print('✓ Uvicorn available')" || echo "✗ Uvicorn not available" 
python -c "import tensorflow; print('✓ TensorFlow available')" || echo "✗ TensorFlow not available"
python -c "import librosa; print('✓ Librosa available')" || echo "✗ Librosa not available"
python -c "import polars; print('✓ Polars available')" || echo "✗ Polars not available"

# Test 5: Check Node.js
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    echo "✓ Node.js $(node --version) and npm $(npm --version) available"
else
    echo "✗ Node.js or npm not available"
fi

# Test 6: Check frontend dependencies
if [ -d "frontend/node_modules" ]; then
    echo "✓ Frontend dependencies installed"
else
    echo "✗ Frontend dependencies not installed"
fi

echo ""
echo "Setup test completed!"