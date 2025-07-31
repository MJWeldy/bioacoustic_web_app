#!/bin/bash

# Reset script - cleans up processes and ports for fresh start

echo "========================================"
echo "Bioacoustics Web App Reset"
echo "========================================"

# Function to kill processes safely
safe_kill() {
    local process_name="$1"
    local pids=$(pgrep -f "$process_name" 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo "Stopping $process_name processes..."
        pkill -f "$process_name" 2>/dev/null
        sleep 2
        
        # Check if any are still running and force kill if necessary
        local remaining=$(pgrep -f "$process_name" 2>/dev/null)
        if [ -n "$remaining" ]; then
            echo "Force stopping remaining $process_name processes..."
            pkill -9 -f "$process_name" 2>/dev/null
        fi
        echo "✓ Stopped $process_name"
    fi
}

# Kill backend processes
safe_kill "python main.py"
safe_kill "uvicorn"

# Kill frontend processes  
safe_kill "npm start"
safe_kill "react-scripts start"
safe_kill "node.*react-scripts"

# Wait for ports to be released
echo "Waiting for ports to be released..."
sleep 3

# Check port status
check_port() {
    local port=$1
    if command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":$port "; then
            echo "⚠️  Port $port still in use"
            return 1
        else
            echo "✓ Port $port is free"
            return 0
        fi
    elif command -v ss &> /dev/null; then
        if ss -tuln | grep -q ":$port "; then
            echo "⚠️  Port $port still in use"
            return 1
        else
            echo "✓ Port $port is free"
            return 0
        fi
    else
        echo "ℹ️  Cannot check port $port status"
        return 0
    fi
}

echo ""
echo "Port Status:"
echo "----------------------------------------"
check_port 3000
check_port 8000

# Clean up log files
if [ -d "logs" ]; then
    echo ""
    echo "Cleaning up logs..."
    rm -f logs/backend.log logs/frontend.log
    echo "✓ Logs cleared"
fi

echo ""
echo "========================================"
echo "Reset Complete!"
echo "========================================"
echo ""
echo "You can now run:"
echo "  ./run_dev.sh    # Start the application"
echo "  ./health_check.sh  # Check system status"