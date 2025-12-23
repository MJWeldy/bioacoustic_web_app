# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

---

## Quick Reference for Claude Code

### Python Script Execution
**CRITICAL**: Always use the bioacoustics-web-app conda environment with full path:

```bash
# For Claude Code (direct python execution)
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python <script-name>.py

# Common commands
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python main.py
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --verbose
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python -m pytest tests/ -v

# Check package availability
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python -c "import polars; print('✓ Available')"
```

### Testing Framework
**93 test functions** covering all backend components with comprehensive mocking:

```bash
# Run all tests
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --verbose

# Test categories
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --unit
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --integration
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --fast

# Specific modules
/home/matt/miniconda3/envs/bioacoustics-web-app/bin/python run_tests.py --module database
```

### Required Dependencies
- **httpx compatibility**: `pip install "httpx<0.27"` for FastAPI TestClient
- **Main packages**: All packages in environment.yml (polars, fastapi, tensorflow, etc.)

---

## Development Guidelines

### Code Style Standards
- **Descriptive names**: Use clear, meaningful variable and function names
- **Function names**: Active verbs describing what the function does
- **Naming convention**: `lower_case_with_underscores` (e.g., `get_data`, `process_audio`)

### Development Philosophy (TDD)
1. **Generate failing tests** based on expected inputs and outputs
2. **Ask for review** of new tests before implementing
3. **Execute tests** to confirm they fail (TDD requirement)
4. **Write minimal code** to make tests pass
5. **Refactor** for readability without changing behavior
6. **Repeat cycle** until feature is complete

### Git Workflow
- Write and run tests before committing
- Make regular commits when features are working
- Create descriptive commit messages

---

## Project Overview

**Bioacoustics Active Learning Web Application** - Replaces Jupyter notebook workflow with modern FastAPI + React interface for building datasets, performing active learning annotation, training models, and evaluating performance in bioacoustic classification.

**Core Architecture**: FastAPI backend + React frontend + Python audio processing modules

### Key Components

**Backend** (`backend/`):
- `main.py`: FastAPI application with REST endpoints and state management
- `modules/database.py`: Three-table architecture (files → clips → annotations) using Polars
- `modules/utilities.py`: Audio processing, embedding generation, model integration
- `modules/classifier.py`: TensorFlow training with custom BCE loss
- `modules/config.py`: Backend model configurations (BirdNET, PERCH, etc.)

**Frontend** (`frontend/src/`):
- `App.js`: Tabbed interface with state management
- `components/`: DatasetBuilder, ActiveLearning, ModelTraining, Evaluation, DatabaseViewer

**Database Architecture**:
```
files (metadata) → clips (segments) → annotations (labels)
file_id           clip_id (FK)       clip_id (FK)
file_path         clip_start         class_name
duration_sec      clip_end           label_value
sampling_rate     annotation_status  annotator_id
```

---

## Development Commands

### Environment Setup
```bash
# Initial setup
./setup.sh          # Linux/macOS
setup.bat            # Windows

# Start development servers
./run_dev.sh         # Linux/macOS
run_dev.bat          # Windows

# System validation
./health_check.sh    # 22 comprehensive tests
./test_setup.sh      # Quick package verification
./reset.sh           # Clean up stuck processes/ports
```

### Backend Development
```bash
# For human users (manual activation required)
conda activate bioacoustics-web-app

# Start backend
cd backend && python main.py

# API testing
curl http://localhost:8000/docs  # FastAPI documentation
curl http://localhost:8000/      # Health check

# Debug imports
python -c "from modules import config; print('✓ Modules working')"

# Database testing
python -c "from backend.modules import database; print('✓ Database working')"

# TensorFlow check
python -c "import tensorflow; print('TF version:', tensorflow.__version__)"
```

### Frontend Development
```bash
cd frontend

# Install and start
npm install
npm start

# Build and test
npm run build
npm test
```

### Package Management
```bash
# Install new packages (requires activation)
conda activate bioacoustics-web-app
conda install <package>  # or pip install <package>
```

---

## Key Development Workflows

### Adding New Backend Models
1. **Configuration** (`config.py`): Add model parameters
2. **Processing** (`utilities.py`): Implement embedding generation
3. **Frontend** (`DatasetBuilder.js`): Add to model dropdown
4. **API**: No changes needed (configuration-driven)

### Extending Annotation Interface
**Backend**: Update `display_web.py`, `main.py` endpoints, `database.py` schema
**Frontend**: Modify `ActiveLearning.js` components and interactions

### Multiclass Annotation System
**Features**: Target class selection, additional positive classes, label strength tracking
**API Endpoints**:
- `/api/active-learning/annotate`: Target class annotation
- `/api/active-learning/annotate-class`: Specific class annotation
- `/api/active-learning/annotate-other-classes`: Mark classes as absent

### Model Training Pipeline
**Features**: Priority-based label loading, custom BCE loss, multiclass support
**Data Formats**: Metadata JSON → Modern filenames → Legacy support

---

## Technical Implementation

### Performance Considerations
**Backend**: Polars DataFrames, TensorFlow caching, embedding pre-computation, async endpoints
**Frontend**: Component-level state, efficient spectrogram rendering, progress polling

### File Format Conventions
**Audio Export**: `originalname_clipstart-class1+class2.wav`
**Database Files**: `audio_database.parquet`, `metadata.json`, `embeddings.pkl`

### Cross-Platform Compatibility
- Conda environment with `environment.yml`
- Platform-specific scripts (`.sh`/`.bat`)
- Automatic path handling

### Error Handling
**Backend**: HTTP exceptions with detailed messages
**Frontend**: Toast notifications with graceful degradation
**Database**: Transaction-like operations with rollback

---

## Development Tips

### Debugging
**Backend**: Check `logs/backend.log`, use `/docs` endpoint, verify imports
**Frontend**: Hot-reload, browser dev tools, network tab debugging
**Database**: Version detection migrations, test with new/existing datasets

### Audio Processing Notes
- Sample rates converted automatically per model
- Embedding generation: 2-3 minute startup time
- Format support via `librosa` and `soundfile`

---

## Architecture Patterns

### State Management
- **Backend**: Global `app_state` dictionary for session persistence
- **Frontend**: Component-level React state with prop passing
- **Updates**: Polling-based status checks for long operations

### Data Flow
1. **Dataset Building**: Audio files → Embeddings → Three-table database
2. **Active Learning**: Database → Filtered clips → Annotations → Updated database
3. **Model Training**: Annotated clips → Feature vectors → TensorFlow model
4. **Evaluation**: Test clips + Model → Predictions → Performance metrics