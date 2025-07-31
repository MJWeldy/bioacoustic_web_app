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
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate bioacoustics-web-app environment"
    exit 1
fi
echo "✓ Activated bioacoustics-web-app environment"

# Verify key packages are available
echo "Verifying environment..."
python -c "import fastapi, uvicorn, tensorflow, librosa, polars" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required Python packages"
    echo "Run ./setup.sh to fix the environment"
    exit 1
fi
echo "✓ Required packages verified"

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

# Wait for backend to start and verify it's running
echo "Waiting for backend to start..."
sleep 5

# Check if backend process is still running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start. Check logs/backend.log"
    exit 1
fi

# Test backend connectivity
if command -v curl &> /dev/null; then
    for i in {1..10}; do
        if curl -s http://localhost:8000/api/dataset/status > /dev/null; then
            echo "✓ Backend is responding"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "Warning: Backend not responding to health check"
        fi
        sleep 1
    done
fi

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

# Wait for frontend to start
echo "Waiting for frontend to start..."
sleep 10

# Check if frontend process is still running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "Error: Frontend failed to start. Check logs/frontend.log"
    cleanup
    exit 1
fi

# Test frontend connectivity (if curl is available)
if command -v curl &> /dev/null; then
    for i in {1..15}; do
        if curl -s http://localhost:3000 > /dev/null; then
            echo "✓ Frontend is responding"
            break
        fi
        if [ $i -eq 15 ]; then
            echo "Warning: Frontend not responding to health check"
        fi
        sleep 2
    done
fi

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
