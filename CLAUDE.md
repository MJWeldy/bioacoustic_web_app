# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Bioacoustics Active Learning Web Application** that replaces a Jupyter notebook workflow with a modern FastAPI + React web interface. The application provides a complete pipeline for building datasets, performing active learning annotation, training models, and evaluating performance in bioacoustic classification tasks.

**Core Architecture**: FastAPI backend serving RESTful APIs + React frontend with tabbed interface + Python modules for audio processing and machine learning.

## Common Development Commands

### Environment Management
```bash
# Initial setup (Linux/macOS)
./setup.sh

# Initial setup (Windows)
setup.bat

# Start development servers
./run_dev.sh     # Linux/macOS
run_dev.bat      # Windows

# Health checks and diagnostics
./health_check.sh    # Comprehensive system validation (22 tests)
./test_setup.sh      # Quick package verification
./reset.sh           # Clean up processes/ports if stuck
```

### Backend Development
```bash
# Activate environment
conda activate bioacoustics-web-app

# Start backend only (manual)
cd backend && python main.py

# Test backend endpoints
curl http://localhost:8000/docs  # FastAPI documentation
curl http://localhost:8000/      # Health check

# Debug imports
python -c "from modules import config; print('✓ Modules working')"
```

### Frontend Development
```bash
# Start frontend only (manual)
cd frontend && npm start

# Install dependencies
cd frontend && npm install

# Build for production
cd frontend && npm run build

# Run tests
cd frontend && npm test
```

### Database Operations
```bash
# Test database functionality
python -c "from backend.modules import database; print('✓ Database module working')"

# Check TensorFlow
python -c "import tensorflow; print('TF version:', tensorflow.__version__)"
```

## High-Level Architecture

### Backend Structure (`backend/`)

**Main Application** (`main.py`):
- FastAPI application with all REST endpoints
- Global application state management for datasets, models, and active sessions
- Cross-Origin Resource Sharing (CORS) for React frontend integration
- Real-time progress tracking for long-running operations (dataset building, training)

**Core Modules** (`backend/modules/`):
- `database.py`: **Three-table architecture** (files → clips → annotations) using Polars for performance
- `utilities.py`: Audio processing, embedding generation, and backend model integration
- `classifier.py`: TensorFlow model training with custom BCE loss for weak/strong labels
- `config.py`: Backend model configurations (BirdNET, PERCH, PNW_Cnet, etc.)
- `display_web.py`: Web-compatible annotation interface (replaces Jupyter widgets)

### Frontend Structure (`frontend/src/`)

**Main Application** (`App.js`):
- Tabbed interface with global state management
- Toast notifications for user feedback
- Responsive design with consistent styling

**Core Components** (`components/`):
- `DatasetBuilder.js`: Dataset creation interface with progress tracking
- `ActiveLearning.js`: Interactive annotation with spectrogram visualization and audio playback
- `ModelTraining.js`: Model training interface with real-time progress monitoring
- `Evaluation.js`: Model evaluation with metrics visualization
- `DatabaseViewer.js`: Database exploration and management interface

### Key Architectural Patterns

**State Management**:
- Backend: Global `app_state` dictionary for session persistence
- Frontend: Component-level React state with prop passing
- Real-time updates: Polling-based status checks for long operations

**Data Flow**:
1. **Dataset Building**: Audio files → Embeddings → Three-table database
2. **Active Learning**: Database → Filtered clips → User annotations → Updated database
3. **Model Training**: Annotated clips → Feature vectors → TensorFlow model → Saved classifier
4. **Evaluation**: Test clips + Trained model → Predictions → Performance metrics

**Database Architecture** (Critical for understanding):
```
files (metadata) → clips (segments) → annotations (human labels)
     ↓                    ↓                     ↓
file_id             clip_id (FK)        clip_id (FK)
file_path           clip_start          class_name
duration_sec        clip_end            label_value
sampling_rate       annotation_status   annotator_id
                    confidence_scores   timestamp
```

## Development Workflows

### Adding New Backend Models

1. **Configuration** (`config.py`): Add model parameters (sample_rate, context_frames, etc.)
2. **Processing Logic** (`utilities.py`): Implement embedding generation for new model
3. **Frontend Integration** (`DatasetBuilder.js`): Add model option to dropdown
4. **API Updates**: No changes needed - model selection is configuration-driven

### Extending Annotation Interface

**Backend Changes**:
- `display_web.py`: Core annotation logic and filtering
- `main.py`: New API endpoints for annotation features
- `database.py`: Schema changes for new annotation types

**Frontend Changes**:
- `ActiveLearning.js`: UI components and user interactions
- Ensure real-time label updates and progress tracking integration

### Multiclass Annotation System

The application implements a sophisticated multiclass annotation system:

**Key Features**:
- **Target Class Selection**: Primary class being annotated (drives score filtering)
- **Additional Positive Classes**: Multi-select interface for marking other present classes
- **Label Strength Tracking**: Strong vs weak labels for training optimization
- **Export Preservation**: Complete annotation state maintained through export/import cycle

**API Endpoints**:
- `/api/active-learning/annotate`: Annotate target class
- `/api/active-learning/annotate-class`: Annotate specific class by index
- `/api/active-learning/annotate-other-classes`: Mark non-target classes as absent

### Model Training Pipeline

**Enhanced Features**:
- **Priority-Based Label Loading**: Metadata → Modern filenames → Legacy formats
- **Custom BCE Loss**: Weighted loss function for strong/weak labels (default weak weight: 0.05)
- **Multiclass Support**: Full multiclass training with partial annotation handling
- **Real-time Monitoring**: Progress tracking, loss visualization, macro cMAP scores

**Training Data Formats**:
1. **Metadata JSON** (preferred): Complete binary vectors with strength information
2. **Modern Filenames**: `originalname_clipstart-class1+class2.wav` format
3. **Legacy Support**: Underscore separators and embedded class names

## Technical Implementation Notes

### Performance Considerations

**Backend**:
- Polars DataFrame operations for large dataset handling
- TensorFlow model caching and GPU utilization
- Embedding pre-computation and storage
- Asynchronous API endpoints for non-blocking operations

**Frontend**:
- Component-level state management (no global state library needed)
- Efficient spectrogram rendering and audio streaming
- Progress polling for long operations (dataset building, training)

### File Format Conventions

**Audio Export Naming**:
```
Site_001_Rep_C_165.0-YRWA_song_1+BRMA_call_1.wav
└─── original name ──┘└─ start time ─┘└─ classes ─┘
```

**Database Files**:
- `audio_database.parquet`: Main three-table database
- `metadata.json`: Dataset configuration and class mappings
- `embeddings.pkl`: Pre-computed embedding vectors

### Cross-Platform Compatibility

**Environment**: Uses conda environment with `environment.yml` for consistent package versions across platforms
**Scripts**: Platform-specific setup and run scripts (`.sh` for Unix, `.bat` for Windows)
**Paths**: Automatic path handling in backend for cross-platform file operations

### Error Handling Patterns

**Backend**: HTTP exception handling with detailed error messages and debug logging
**Frontend**: Toast notifications for user feedback with graceful degradation
**Database**: Transaction-like operations with rollback capabilities for failed operations

## Development Tips

### Debugging Backend Issues
- Check `logs/backend.log` for FastAPI server logs
- Use `/docs` endpoint for interactive API testing
- Verify module imports with test commands in environment

### Frontend Development
- React development server hot-reloads on changes
- Browser developer tools for component state inspection
- Network tab for API request/response debugging

### Database Schema Changes
- Database migrations handled through version detection in `database.py`
- Legacy format support maintained for backward compatibility
- Always test with both new and existing datasets

### Audio Processing
- Sample rate conversions handled automatically per backend model
- Embedding generation is CPU/GPU intensive - expect 2-3 minutes startup time
- Audio file format support depends on `librosa` and `soundfile` capabilities