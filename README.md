TODO
Proposed Tabs:
  1. Dataset Builder - Upload audio, configure parameters, generate embeddings
  2. Active Learning - Review clips, annotate, select uncertain samples
  3. Model Training - Configure training parameters, monitor progress
  4. Evaluation - View metrics, confusion matrices, performance charts


Please go ahead and build a prototype using FastAPI and React. Some elements of the original code will need to change; however, for now please stay close to the original implementation. 

The Dataset Builder tab should include the following functions
- A button that allows me to select a folder of audio files
- An interface for me to create the class_map for the classifier
- A backend model selection (BirdNET, Perch, PNW_Cnet, ...) 
- A button that selects a location to save the embeddings and the audio database
- The option to load a pretrained classifer and populate scores. If there is no pretrained classifier available we will use embedding search (using the Euclidian distance of a clip embedding to a query embedding as the initial score) to start reviewing audio. 

The Active Learning tab should include the following functions
- A button that loads a dataset built in the Dataset Builder tab. Loading the dataset should automatically recognize the backend model that was selected
- The gui interface for active learning that is currently generated when I run dis.annotate() in the active_learning_loop.ipynb file
- There should be a button to load a pretrained classifier and update the model scores
- There should be an option to select the spectrogram color mode in addition to the other buttons found in the display module. 
- There should an option to save the Audio database
- There should be an option to write the annotated clips in audio database to a folder

The Model Training tab should include the following functions
- Select a folder of wav files that will be used for model training
- Load and embed the clips 
- Read the clip labels from the file names
- Create train and test partitions of the data allowing me to select a random state and test data size 
- Train a new classifier using the tc.fit_w_tape function, allow me to control access to the function arguments 

The Evaluation tab should include the following functions
- Allow me to select a folder containing annotated audio clips for evaluation
- load the labels from the clips
- Calculate and display AUC, Average Precision, and a confusion matrix for single class models. For multiclass models display macro averaged AUC, mean Average Precision, and a multiclass confusion matrix. Also allow me to see class specific AUC, Average Precision  







# Bioacoustics Active Learning Web Application

A web-based application for building datasets and performing active learning in bioacoustic classification. This application provides an interface equivalent to the original Jupyter notebook workflow but runs as a modern web application with FastAPI backend and React frontend.

## Project Status

✅ **Fully Functional** - Complete web application with all core features working  
✅ **Cross-Platform Compatible** - Tested on Linux, macOS, and Windows  
✅ **Well Documented** - Comprehensive setup, usage, and troubleshooting documentation  
✅ **Clean Architecture** - Organized FastAPI backend + React frontend structure  

**Recent Achievements:**
- ✅ **Fixed Critical Backend Issues**: Resolved all 500 errors in multiclass annotation system
- ✅ **Enhanced Setup Process**: Now works for first-time users without existing environments
- ✅ **Comprehensive Diagnostics**: Added health check, test, and reset utilities  
- ✅ **Robust Error Handling**: Improved validation and error reporting across all endpoints
- ✅ **First-Time User Support**: Complete setup from scratch using environment.yml
- ✅ **Process Management**: Reliable startup/shutdown with health monitoring
- ✅ **Clean Frontend**: Eliminated ESLint warnings and compilation issues
- ✅ **Complete Documentation**: Setup guides, troubleshooting, and user manuals

## Features

### 🎯 **Dataset Builder Tab**
- **Audio File Management**: Browse and select folders containing WAV/MP3 files
- **Custom Classification**: Create and edit class maps with names and numerical values
- **Backend Model Selection**: Choose from BirdNET, Perch, PNW_Cnet, and other models
- **Flexible Storage**: Select custom locations for embeddings and database files
- **Pretrained Integration**: Load existing classifiers for initial score population
- **Automated Processing**: Generate embeddings and create databases with progress tracking

### 🎵 **Active Learning Tab**
- **Dataset Loading**: Import datasets created with Dataset Builder
- **Interactive Annotation Interface**:
  - Dynamic spectrogram visualization with multiple color schemes
  - Integrated audio playback with automatic clip loading
  - Score-based filtering with adjustable range sliders
  - Intuitive navigation controls (Previous/Next/Jump to clip)
- **Flexible Annotations**: Support for Present (1), Not Present (0), and Uncertain (3) labels
- **Enhanced Export System**: Export annotated clips with complete label strength preservation
  - **Smart Filenames**: `filename_clipstart-annotation_slug-strong.wav` format
  - **Metadata Export**: Complete binary label vectors and strength information in JSON
  - **Multiclass Support**: Full annotation state preservation for all classes
- **Data Management**: Save annotations to database and export with complete metadata
- **Progress Tracking**: Real-time statistics and completion monitoring

### 🤖 **Model Training Tab**
- **Enhanced Data Loading**: Intelligent loading system with priority-based label extraction
  - **Priority 1**: Metadata binary vectors (multiclass-compatible)
  - **Priority 2**: Enhanced filename parsing (strong labels only)
  - **Priority 3**: Legacy filename parsing (backward compatibility)
- **Automatic Embedding**: Load and embed audio clips using selected backend models
- **Label Strength Integration**: Automatic detection and use of weak/strong label information
- **Data Partitioning**: Create train/test splits preserving label strength information
- **Advanced Loss Function**: Custom BCE loss with weak negative weighting (default 0.05)
- **Training Monitoring**: Real-time progress tracking and loss visualization
- **Multiclass Support**: Full support for training on exported Active Learning datasets

### 📊 **Evaluation Tab**
- **Evaluation Dataset Loading**: Select folders containing annotated audio clips for model evaluation
- **Automatic Label Detection**: Extract ground truth labels from audio file names
- **Comprehensive Metrics**: Calculate and display performance metrics:
  - **Single Class Models**: AUC, Average Precision, Confusion Matrix
  - **Multi-Class Models**: Macro-averaged AUC, Mean Average Precision, Multi-class Confusion Matrix
  - **Class-Specific Analysis**: Individual class AUC and Average Precision scores
- **Visual Analytics**: Interactive charts and confusion matrix visualizations
- **Results Export**: Save evaluation metrics and visualizations

### 🗃️ **Database Viewer Tab**  
- **Database Exploration**: Browse and examine audio databases created by Dataset Builder
- **Annotation Review**: View existing annotations and their statistics
- **Data Filtering**: Filter clips by annotation status, score ranges, and metadata
- **Bulk Operations**: Perform batch annotation updates and data cleanup
- **Export Options**: Export filtered datasets or annotation summaries
- **Database Statistics**: View comprehensive dataset statistics and class distributions

### 🔧 **Technical Features**
- **Cross-Platform Compatibility**: Runs on Linux, macOS, and Windows
- **Modern Web Interface**: Responsive React frontend with tabbed navigation
- **RESTful API**: Comprehensive FastAPI backend with automatic documentation
- **Flexible Architecture**: Easily extensible for new models and annotation types

## Architecture

### Backend (FastAPI)
- **main.py**: Main FastAPI application with all REST API endpoints
- **modules/**: Self-contained Python modules package
  - **display_web.py**: Web-compatible annotation interface (replaces ipywidgets)
  - **database.py**: Polars-based audio database management
  - **classifier.py**: TensorFlow model training and evaluation
  - **utilities.py**: Audio processing and embedding utilities
  - **config.py**: Configuration for different backend models

### Frontend (React)
- **App.js**: Main application with tabbed interface and global styling
- **components/DatasetBuilder.js**: Interface for creating new datasets
- **components/ActiveLearning.js**: Interactive annotation interface
- Modern React with hooks, responsive design, and real-time updates

## Cross-Platform Compatibility

### ✅ **Linux** (Primary Development Platform)
- **Status**: Fully tested and supported
- **Installation**: `./setup.sh`
- **Startup**: `./run_dev.sh`

### ✅ **macOS** (Intel and Apple Silicon)
- **Status**: Fully compatible
- **Installation**: `./setup_macos.sh`
- **Startup**: `./run_dev.sh` (auto-generated)
- **Special considerations**: Handles multiple conda installation paths

### ✅ **Windows** (10/11)
- **Status**: Compatible with Windows scripts
- **Installation**: `setup.bat`
- **Startup**: `run_dev.bat`
- **Requirements**: Windows 10+ with conda and Node.js

## Installation and Setup

### 🚀 Quick Start (New Users)

**Prerequisites:**
- **Conda/Miniconda/Anaconda** - Download from: https://docs.conda.io/en/latest/miniconda.html

**Installation:**
```bash
git clone https://github.com/MJWeldy/bioacoustic_web_app.git
cd bioacoustic_web_app
./setup.sh      # Works for first-time users!
./run_dev.sh    # Start the application
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 🔧 Setup Methods

The setup process **automatically handles both scenarios**:

#### ✅ **First-Time Users (Recommended)**
- **No existing environment needed!** 
- Creates environment from `environment.yml` with all required packages
- Includes TensorFlow, FastAPI, Librosa, Polars, Node.js, and all dependencies

#### ✅ **Existing Users** 
- If you have an `active_learning` environment, it will be cloned as fallback
- Preserves your existing setup while creating dedicated web app environment

### Platform-Specific Installation

#### Linux
```bash
./setup.sh
```

#### macOS
```bash
./setup_macos.sh  # or use ./setup.sh
```

#### Windows
```cmd
setup.bat
```

### 🧪 Verification & Diagnostics

**Test your installation:**
```bash
./health_check.sh    # Comprehensive system validation (22 tests)
./test_setup.sh      # Quick package verification
```

**Troubleshooting:**
```bash
./reset.sh           # Clean up processes/ports if stuck
./health_check.sh    # Diagnose issues
```

### What the Installation Does
1. **Environment Creation**: Uses `environment.yml` (primary) or clones existing environment (fallback)
2. **Package Installation**: Installs all Python and Node.js dependencies automatically
3. **Verification**: Tests all components and provides clear status feedback
4. **Script Generation**: Creates optimized run scripts with health checks

### Running the Application

#### Linux/macOS
```bash
./run_dev.sh
```

#### Windows
```cmd
run_dev.bat
```

#### Manual startup (all platforms)
```bash
# Activate conda environment first
conda activate bioacoustics-web-app

# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend  
cd frontend && npm start
```

### Access Points
- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs

## Usage

### Dataset Builder
1. **Audio Folder**: Specify path to folder containing WAV/MP3 files
2. **Class Map**: Define your classification classes with names and numerical values
3. **Backend Model**: Choose embedding model (PERCH, BirdNET, PNWCnet, etc.)
4. **Save Path**: Choose where to store generated embeddings and database
5. **Pretrained Classifier** (optional): Load existing model to populate initial scores
6. Click **Create Dataset** to process audio files and generate embeddings

### Active Learning
1. **Load Dataset**: Enter path to dataset created with Dataset Builder
2. **Load Classifier** (optional): Load pretrained model to update scores
3. **Adjust Filters**: Use score range slider to filter clips for review
4. **Review Clips**: 
   - View spectrograms with customizable color schemes
   - Listen to audio clips with automatic playback
   - Navigate between clips using Previous/Next buttons
5. **Annotate**: Click annotation buttons (Present/Not Present/Uncertain)
6. **Save Progress**: Regularly save annotations to database
7. **Export Results**: Export annotated clips as WAV files

### Model Training
1. **Select Training Data**: Choose folder containing annotated WAV files
2. **Configure Embedding**: Select backend model for feature extraction
3. **Set Parameters**: Configure random state, test size, and training parameters
4. **Create Data Splits**: Generate train/test partitions
5. **Start Training**: Begin model training with progress monitoring
6. **Monitor Progress**: View real-time training metrics and loss curves
7. **Save Model**: Export trained classifier for use in other tabs

### Evaluation
1. **Load Test Dataset**: Select folder with annotated audio clips for evaluation
2. **Load Model**: Choose trained classifier to evaluate
3. **Run Evaluation**: Execute evaluation on test dataset
4. **View Metrics**: Examine AUC, Average Precision, and confusion matrices
5. **Analyze Results**: Review class-specific performance metrics
6. **Export Results**: Save evaluation reports and visualizations

### Database Viewer
1. **Load Database**: Select audio database created with Dataset Builder
2. **Explore Data**: Browse clips, annotations, and metadata
3. **Apply Filters**: Filter by annotation status, scores, or other criteria
4. **Review Statistics**: View dataset composition and class distributions
5. **Bulk Operations**: Perform batch annotation updates if needed
6. **Export Data**: Export filtered datasets or summary statistics

## API Endpoints

### Dataset Builder
- `POST /api/dataset/create`: Create new dataset with embeddings
- `GET /api/dataset/status`: Get current dataset status

### Active Learning
- `POST /api/active-learning/load-dataset`: Load existing dataset
- `POST /api/active-learning/load-classifier`: Load pretrained classifier
- `POST /api/active-learning/get-clips`: Get filtered clips for annotation
- `POST /api/active-learning/annotate`: Annotate a clip
- `POST /api/active-learning/save-database`: Save annotation database
- `POST /api/active-learning/export-clips`: Export annotated clips
- `POST /api/spectrogram`: Generate spectrogram visualization
- `GET /api/audio/{file_path}`: Stream audio clips

### Model Training
- `POST /api/training/load-data`: Load annotated audio files for training
- `POST /api/training/create-partitions`: Create train/test data splits
- `POST /api/training/start`: Start model training with specified parameters
- `GET /api/training/status`: Get current training progress and metrics
- `POST /api/training/stop`: Stop current training process
- `GET /api/training/history`: Get training history and loss curves

### Evaluation
- `POST /api/evaluation/load-dataset`: Load evaluation dataset
- `POST /api/evaluation/run`: Run model evaluation on loaded dataset
- `GET /api/evaluation/metrics`: Get computed evaluation metrics
- `GET /api/evaluation/confusion-matrix`: Get confusion matrix data
- `POST /api/evaluation/export-results`: Export evaluation results

### Database Viewer
- `POST /api/database/load`: Load and examine audio database
- `GET /api/database/stats`: Get database statistics and summaries
- `POST /api/database/filter`: Filter database entries by criteria
- `POST /api/database/bulk-annotate`: Perform bulk annotation operations
- `POST /api/database/export`: Export filtered database contents

## Configuration

The application supports the same backend models as the original notebook:

- **PERCH**: Google's PERCH model v8 for bird vocalizations
- **BirdNET_2.4**: BirdNET 2.4 model 
- **PNWCnet**: Pacific Northwest focused model
- **PNWCnet_EXPANDED**: Expanded version with broader frequency range

Model-specific parameters (sample rates, context frames, etc.) are automatically configured based on the selected backend.

## File Structure

```
bioacoustics_web_app/
├── setup.sh                  # Linux installation script
├── setup_macos.sh            # macOS installation script  
├── setup.bat                 # Windows installation script
├── run_dev.sh               # Linux/macOS startup script
├── run_dev.bat              # Windows startup script
├── test_installation.sh     # Verify installation (Linux/macOS)
├── README.md                # Project documentation
├── DEVELOPMENT_LOG.md       # Complete development history
├── active_learning_loop.ipynb # Original Jupyter notebook
├── backend/
│   ├── main.py               # FastAPI application with all endpoints
│   ├── requirements.txt      # Python dependencies (legacy)
│   └── modules/              # Complete Python modules package
│       ├── __init__.py       # Package initialization
│       ├── config.py         # Backend model configurations
│       ├── database.py       # Polars-based audio database
│       ├── utilities.py      # Audio processing and embeddings
│       ├── classifier.py     # TensorFlow model training/evaluation
│       ├── display.py        # Original Jupyter display module
│       └── display_web.py    # Web-compatible annotation interface
├── frontend/
│   ├── package.json          # Node.js dependencies and scripts
│   ├── package-lock.json     # Dependency lock file
│   ├── public/
│   │   └── index.html        # HTML template
│   ├── node_modules/         # Installed dependencies
│   └── src/
│       ├── index.js          # React application entry point
│       ├── index.css         # Global styles
│       ├── App.js            # Main application with tabs
│       ├── App.css           # Application-specific styles
│       └── components/
│           ├── DatasetBuilder.js  # Dataset creation interface
│           ├── ActiveLearning.js  # Interactive annotation interface
│           ├── ModelTraining.js   # Model training interface
│           ├── Evaluation.js      # Model evaluation interface
│           └── DatabaseViewer.js  # Database exploration interface
├── logs/                     # Runtime logs (created automatically)
│   ├── backend.log           # FastAPI server logs
│   └── frontend.log          # React development server logs
├── checkpoints/              # Model checkpoints directory
├── embedding_checkpoints/    # Pre-trained embedding models
└── data/                     # Example data directory
    └── test/                 # Test dataset
        ├── audio/            # Audio files
        ├── annotated/        # Exported annotated clips
        └── *.keras           # Trained classifier models
```

## Migration from Jupyter Notebook

This web application provides equivalent and extended functionality compared to the original `active_learning_loop.ipynb` notebook:

**Core Functionality Migration:**
- **Dataset Creation**: Replaces manual embedding generation and database setup
- **Interactive Annotation**: Replaces `dis.annotate()` widget interface with web-based UI
- **Score Filtering**: Equivalent score range filtering with improved sliders
- **Multiple Review Modes**: Supports random, sorted, and CDE review modes
- **Export Functionality**: Same export capabilities as original

**Enhanced Features:**
- **Model Training Tab**: Advanced training with weak/strong label support and custom BCE loss
- **Evaluation Tab**: Comprehensive metrics display with confusion matrices and class-specific analysis  
- **Database Viewer**: New functionality for exploring and managing audio databases
- **Enhanced Export/Import**: Complete label preservation workflow for active learning
- **Multiclass Active Learning**: Full support for multiclass annotation and training workflows
- **Cross-Platform Support**: Web interface accessible from any browser
- **Improved Usability**: Modern UI/UX replacing Jupyter widget dependencies

## Enhanced Active Learning Workflow

This application implements a complete active learning pipeline with advanced label preservation and multiclass support:

### **Export System (Active Learning → Model Training)**

**Smart Filename Convention:**
```
filename_clipstart-annotation_slug-strong.wav
```
- **Simple & Unambiguous**: Only strong labels in filenames to avoid multiclass conflicts
- **Backward Compatible**: Works with legacy filename parsing systems

**Enhanced Metadata Export:**
```json
{
  "export_info": {
    "export_date": "2025-06-10T15:30:04.744248",
    "total_clips_exported": 25,
    "positive_clips": 15,
    "negative_clips": 8,
    "uncertain_clips": 2
  },
  "class_map": {"NOWA_song": 0, "CEWA": 1, "WIWA": 2},
  "clips": [
    {
      "filename": "audio_10.0-NOWA_song-strong.wav",
      "annotation_slug": "NOWA_song",
      "labels": [1, 0, 0],              // Binary vector for all classes
      "label_strengths": [1, 1, 0],     // Strength vector: 1=strong, 0=weak
      "scores": [0.85, 0.12, 0.03]      // Prediction scores for all classes
    }
  ]
}
```

### **Intelligent Import System (Model Training)**

**Priority-Based Loading:**
1. **🥇 Metadata Binary Vectors**: Most accurate, full multiclass support
2. **🥈 Enhanced Filenames**: Strong labels only, single-class compatible  
3. **🥉 Legacy Filenames**: Backward compatibility with existing datasets

**Label Strength Integration:**
- **Strong Labels** (confident annotations): Full weight (1.0) in loss function
- **Weak Labels** (uncertain annotations): Reduced weight (0.05) in loss function
- **Mixed Training**: Seamlessly combines strong and weak labels in same training run

### **Advanced Training Features**

**Custom BCE Loss Function:**
```python
def bce_loss(y_true, logits, is_labeled_mask, weak_neg_weight=0.05):
    # Strong labels get full weight (1.0)
    # Weak labels get reduced weight (0.05)
    weights = (1.0 - is_labeled_mask) * weak_neg_weight + is_labeled_mask
    return tf.reduce_mean(raw_bce * weights)
```

**Multiclass Active Learning:**
- Export preserves complete annotation state for all classes
- Training handles partial annotations (some classes reviewed, others not)
- Uncertain annotations become weak positive labels with reduced influence

## Development and Customization

### Adding New Backend Models
1. Add model configuration to `backend/modules/config.py`
2. Implement processing logic in `backend/modules/utilities.py`
3. Add model option to frontend `src/components/DatasetBuilder.js`
4. Update API documentation and type definitions

### Extending Annotation Interface
- Modify `backend/modules/display_web.py` for new annotation logic
- Update frontend `src/components/ActiveLearning.js` for UI changes
- Add new API endpoints in `backend/main.py` as needed
- Update React state management and component props

### Performance Optimization
- **Backend**: Consider caching embeddings and pre-loading models
- **Frontend**: Implement virtualization for large clip lists
- **Database**: Use indexed queries for faster filtering
- **Audio**: Optimize clip loading and spectrogram generation

### Testing
```bash
# Test the installation
./test_installation.sh

# Check services are running
curl http://localhost:8000/docs  # API documentation
curl http://localhost:3000       # Frontend application
```

## Troubleshooting

### Environment Issues
- **conda not found**: Install Anaconda/Miniconda and restart terminal
- **Environment activation fails**: Run `conda init` and restart terminal  
- **Permission denied on scripts**: Run `chmod +x setup.sh run_dev.sh`
- **active_learning environment missing**: Set up original notebook first

### Common Issues and Solutions

#### Backend Won't Start
```bash
# Check if environment is activated
conda list | grep fastapi

# Test module imports
python -c "from modules import config; print('✓ Modules working')"

# Check for port conflicts
lsof -i :8000
```

#### Frontend Build Errors
```bash
# Clear and reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### TensorFlow/GPU Issues
- **Startup takes 2-3 minutes**: Normal GPU initialization time
- **CUDA warnings**: Expected with GPU setup, not errors
- **Memory errors**: Reduce batch sizes or disable GPU

#### Import/Module Errors
- **Cannot import modules**: Ensure all files copied to `backend/modules/`
- **Path issues**: Verify `backend/modules/__init__.py` exists

### Environment Management
```bash
# Check if environment exists
conda env list

# Remove and recreate environment
conda env remove -n bioacoustics-web-app
./setup.sh

# Manual debugging
conda activate bioacoustics-web-app
python -c "import tensorflow; print('TF version:', tensorflow.__version__)"
```

### Performance Notes
- **Backend startup**: Initial startup takes 2-3 minutes (TensorFlow GPU initialization)
- **Memory usage**: ~4GB RAM with loaded models
- **Disk space**: ~500MB for dependencies + model storage

### Platform-Specific Issues

#### Windows
- **PowerShell Execution Policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Path separators**: Use backslashes `\` in file paths
- **Audio codecs**: May need additional codec installation for some audio formats
- **Long paths**: Enable long path support in Windows if needed

#### macOS  
- **Permission issues**: Run `chmod +x setup_macos.sh run_dev.sh` if needed
- **Homebrew dependencies**: Install Homebrew for easier package management
- **Apple Silicon**: Works on both Intel and M1/M2 Macs
- **Conda paths**: Script auto-detects various conda installation locations

#### Linux
- **Audio libraries**: May need `sudo apt install ffmpeg libsndfile1` on Ubuntu/Debian
- **Permissions**: Ensure execute permissions on shell scripts
- **GPU drivers**: Install appropriate NVIDIA drivers for GPU acceleration

For detailed troubleshooting, see `DEVELOPMENT_LOG.md` which documents all issues encountered during development and their solutions.

## License

This project extends the original bioacoustics active learning framework into a web application format.
