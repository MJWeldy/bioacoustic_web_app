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
    echo "Run ./setup.sh first"
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
