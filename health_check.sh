#!/bin/bash

# Health check script for Bioacoustics Web App
echo "========================================"
echo "Bioacoustics Web App Health Check"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results counter
TOTAL_TESTS=0
PASSED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    printf "%-40s" "$test_name"
    
    if eval "$test_command" &> /dev/null; then
        echo -e "[${GREEN}PASS${NC}]"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "[${RED}FAIL${NC}]"
        return 1
    fi
}

run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    printf "%-40s" "$test_name"
    
    result=$(eval "$test_command" 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "[${GREEN}PASS${NC}] $result"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "[${RED}FAIL${NC}] $result"
        return 1
    fi
}

echo "System Requirements:"
echo "----------------------------------------"
run_test_with_output "Conda available" "conda --version | head -1"
run_test_with_output "Python available" "python --version"

echo ""
echo "Environment Check:"
echo "----------------------------------------"
run_test "bioacoustics-web-app env exists" "conda env list | grep -q 'bioacoustics-web-app'"

# Try to activate environment for package checks
if conda env list | grep -q "bioacoustics-web-app"; then
    # Initialize conda
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    else
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi
    
    conda activate bioacoustics-web-app 2>/dev/null
    
    echo ""
    echo "Python Packages:"
    echo "----------------------------------------"
    run_test_with_output "FastAPI" "python -c 'import fastapi; print(fastapi.__version__)'"
    run_test_with_output "Uvicorn" "python -c 'import uvicorn; print(uvicorn.__version__)'"
    run_test_with_output "TensorFlow" "python -c 'import tensorflow as tf; print(tf.__version__)'"
    run_test_with_output "Librosa" "python -c 'import librosa; print(librosa.__version__)'"
    run_test_with_output "Polars" "python -c 'import polars as pl; print(pl.__version__)'"
    run_test "NumPy" "python -c 'import numpy'"
    run_test "SciKit-Learn" "python -c 'import sklearn'"
    run_test "Pandas" "python -c 'import pandas'"
    run_test "SoundFile" "python -c 'import soundfile'"
else
    echo -e "${YELLOW}Skipping package checks - environment not found${NC}"
fi

echo ""
echo "Node.js and Frontend:"
echo "----------------------------------------"
run_test_with_output "Node.js" "node --version"
run_test_with_output "npm" "npm --version"
run_test "Frontend dependencies" "[ -d frontend/node_modules ]"

echo ""
echo "Port Availability:"
echo "----------------------------------------"
# Check if ports are free
if command -v netstat &> /dev/null; then
    run_test "Port 3000 available" "! netstat -tuln | grep -q ':3000 '"
    run_test "Port 8000 available" "! netstat -tuln | grep -q ':8000 '"
elif command -v ss &> /dev/null; then
    run_test "Port 3000 available" "! ss -tuln | grep -q ':3000 '"
    run_test "Port 8000 available" "! ss -tuln | grep -q ':8000 '"
else
    echo -e "${YELLOW}Cannot check ports - netstat/ss not available${NC}"
fi

echo ""
echo "File Structure:"
echo "----------------------------------------"
run_test "Backend main.py exists" "[ -f backend/main.py ]"
run_test "Frontend package.json exists" "[ -f frontend/package.json ]"
run_test "Environment.yml exists" "[ -f environment.yml ]"
run_test "Setup script exists" "[ -f setup.sh ]"
run_test "Run script exists" "[ -f run_dev.sh ]"

echo ""
echo "========================================"
echo "Health Check Summary"
echo "========================================"
echo "Tests passed: $PASSED_TESTS/$TOTAL_TESTS"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "Status: ${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo "Your setup looks good. Try running:"
    echo "  ./run_dev.sh"
    exit 0
elif [ $PASSED_TESTS -gt $((TOTAL_TESTS / 2)) ]; then
    echo -e "Status: ${YELLOW}Most tests passed, but some issues found${NC}"
    echo ""
    echo "Try running setup to fix issues:"
    echo "  ./setup.sh"
    exit 1
else
    echo -e "Status: ${RED}Many tests failed - setup required${NC}"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    echo ""
    echo "If setup fails, check SETUP.md for manual installation steps."
    exit 1
fi