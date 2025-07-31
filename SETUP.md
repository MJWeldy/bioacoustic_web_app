# Bioacoustics Web App Setup Guide

This guide will help you set up the Bioacoustics Active Learning Web Application on your system.

## Prerequisites

1. **Conda/Miniconda/Anaconda** - Required for Python environment management
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Or use existing Anaconda installation

2. **Node.js** (optional) - Will be installed automatically by conda if not present
   - If you prefer to install manually: https://nodejs.org/

## Quick Start

### For First-Time Users

1. **Clone or download this repository**
2. **Navigate to the project directory**
3. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
4. **Start the application:**
   ```bash
   ./run_dev.sh
   ```

That's it! The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### For Users with Existing `active_learning` Environment

If you already have an `active_learning` conda environment (from previous bioacoustics work), the setup script will:

1. First try to create the environment from `environment.yml`
2. If that fails, it will clone your existing `active_learning` environment
3. Install any missing web framework packages

## Manual Setup (if automatic setup fails)

### 1. Create Conda Environment

```bash
# Option A: From environment.yml (recommended)
conda env create -f environment.yml

# Option B: Clone existing environment (if you have active_learning)
conda create --name bioacoustics-web-app --clone active_learning

# Option C: Create from scratch
conda create -n bioacoustics-web-app python=3.10
```

### 2. Activate Environment and Install Packages

```bash
conda activate bioacoustics-web-app

# Install Python packages
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 python-multipart>=0.0.6
pip install tensorflow>=2.13.0 tensorflow-hub>=0.14.0
pip install librosa>=0.10.0 soundfile>=0.12.0
pip install polars>=0.20.0 numpy>=1.24.0 scikit-learn>=1.3.0

# Install Node.js (if not already available)
conda install -c conda-forge nodejs
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

## Troubleshooting

### Common Issues

1. **"bioacoustics-web-app environment not found"**
   - Run `./setup.sh` first
   - Check if conda is properly initialized: `conda info`

2. **"Something is already running on port 3000/8000"**
   - Kill existing processes: `pkill -f "npm start" && pkill -f "python main.py"`
   - Or use different ports by modifying the scripts

3. **Frontend fails to start**
   - Check if Node.js is installed: `node --version`
   - Install frontend dependencies: `cd frontend && npm install`

4. **Backend import errors**
   - Verify Python packages: `python -c "import fastapi, tensorflow, librosa"`
   - Re-run setup: `./setup.sh`

5. **Conda initialization issues**
   - Initialize conda: `conda init bash` (then restart terminal)
   - Or source conda manually: `source ~/miniconda3/etc/profile.d/conda.sh`

### Testing Your Setup

Run the verification script to check if everything is properly installed:

```bash
./test_setup.sh
```

### Log Files

If something goes wrong, check the log files:
- Backend logs: `logs/backend.log`
- Frontend logs: `logs/frontend.log`

### Getting Help

1. **Check the logs** in the `logs/` directory
2. **Run the test script** to identify missing components
3. **Verify conda environment** with `conda list` after activating
4. **Check port availability** with `netstat -tulpn | grep -E ":(3000|8000)"`

## Environment Details

The setup creates a conda environment with:
- Python 3.10
- TensorFlow for machine learning
- FastAPI + Uvicorn for the web backend
- Librosa for audio processing
- Polars for efficient data handling
- Node.js + React for the frontend

## Development

- **Start development servers**: `./run_dev.sh`
- **Stop servers**: Press `Ctrl+C` in the terminal running `run_dev.sh`
- **Rebuild environment**: Remove existing environment and re-run setup:
  ```bash
  conda env remove -n bioacoustics-web-app
  ./setup.sh
  ```

## Platform-Specific Notes

- **Linux**: Should work out of the box
- **macOS**: Use `setup_macos.sh` if available, or `setup.sh`
- **Windows**: Use `setup.bat` or run setup.sh in Git Bash/WSL