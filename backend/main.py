from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, Any
import os
import json
import pickle
import numpy as np
from pathlib import Path
import polars as pl
import platform
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import threading
import time
from datetime import datetime

# Import our modules
from modules import config as cfg
from modules import database as db
from modules import utilities as u
from modules import classifier as tc
from modules import validation_db as vdb

app = FastAPI(title="Bioacoustics Active Learning", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for Pydantic validation errors
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"DEBUG: FastAPI validation error: {exc}")
    print(f"DEBUG: Request URL: {request.url}")
    print(f"DEBUG: Request method: {request.method}")
    print(f"DEBUG: Request headers: {dict(request.headers)}")
    try:
        body = await request.body()
        print(f"DEBUG: Request body: {body}")
    except Exception as e:
        print(f"DEBUG: Could not read request body: {e}")
    return HTTPException(status_code=422, detail=f"Validation error: {exc.errors()}")

# Global state - in production use proper state management
app_state = {
    "audio_db": None,
    "embeddings": None,
    "classifier_model": None,
    "class_map": None,
    "backend_model": None,
    "dataset_path": None,
    "save_path": None,
    "review_mode": "random",  # Default review mode
    "validation_db": None  # Validation database instance
}

# Evaluation-specific state
eval_state = {
    "eval_embeddings": None,
    "eval_labels": None,
    "eval_classifier": None,
    "eval_class_map": None,
    "eval_dataset_path": None
}

# Building status state
building_state = {
    "status": "idle",  # idle, building, completed, error
    "message": "",
    "progress": 0,
    "total_files": 0,
    "processed_files": 0
}

# Training status state
training_state = {
    "status": "idle",  # idle, training, completed, error, stopping
    "message": "",
    "logs": [],
    "results": None,
    "stop_requested": False
}

# Spectrogram cache configuration (module-level for global access)
import tempfile
import hashlib
SPECTROGRAM_CACHE_DIR = Path(tempfile.gettempdir()) / "bioacoustic_spectrogram_cache"
MAX_SPECTROGRAM_CACHE_SIZE_GB = 1
MAX_SPECTROGRAM_CACHE_SIZE_BYTES = MAX_SPECTROGRAM_CACHE_SIZE_GB * 1024 * 1024 * 1024

def get_spectrogram_cache_key(file_path: str, start: float, end: float, color_mode: str,
                              freq_scale: str = "mel", n_mels: int = 256, n_fft: int = 2048,
                              hop_length: int = 128, window_size: Optional[int] = None,
                              fmin: Optional[float] = None, fmax: Optional[float] = None,
                              bandpass_min: Optional[float] = None, bandpass_max: Optional[float] = None) -> str:
    """Generate unique cache key for a spectrogram including all parameters"""
    # Version 3: Spectrograms with customizable parameters
    cache_string = f"v3_{file_path}_{start}_{end}_{color_mode}_{freq_scale}_{n_mels}_{n_fft}_{hop_length}_{window_size}_{fmin}_{fmax}_{bandpass_min}_{bandpass_max}"
    return hashlib.md5(cache_string.encode()).hexdigest() + ".png"

def get_spectrogram_cache_size() -> int:
    """Get total size of spectrogram cache directory in bytes"""
    total_size = 0
    if SPECTROGRAM_CACHE_DIR.exists():
        for file in SPECTROGRAM_CACHE_DIR.iterdir():
            if file.is_file():
                total_size += file.stat().st_size
    return total_size

def cleanup_old_spectrogram_files():
    """Remove oldest spectrogram cache files if cache exceeds size limit"""
    if not SPECTROGRAM_CACHE_DIR.exists():
        return

    current_size = get_spectrogram_cache_size()
    if current_size <= MAX_SPECTROGRAM_CACHE_SIZE_BYTES:
        return

    # Get all cache files sorted by access time (oldest first)
    cache_files = [(f, f.stat().st_atime) for f in SPECTROGRAM_CACHE_DIR.iterdir() if f.is_file()]
    cache_files.sort(key=lambda x: x[1])

    # Remove oldest files until under 80% limit to avoid frequent cleanup
    for file_path, _ in cache_files:
        if current_size <= MAX_SPECTROGRAM_CACHE_SIZE_BYTES * 0.8:
            break
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            current_size -= file_size
            print(f"Spectrogram cache cleanup: Removed {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"Spectrogram cache cleanup error: {e}")

# Pydantic models
class ClassMapItem(BaseModel):
    name: str
    value: int

class DatasetConfig(BaseModel):
    audio_folder: str
    class_map: List[ClassMapItem]
    backend_model: str
    save_path: str
    is_evaluation_dataset: bool = False

class FilterConfig(BaseModel):
    score_min: float = 0.0
    score_max: float = 1.0
    annotation_filter: Optional[List[int]] = None

class AnnotationRequest(BaseModel):
    clip_id: str
    annotation: int  # 0: not present, 1: present, 3: uncertain

class MultiClassAnnotationRequest(BaseModel):
    clip_id: str
    annotation: int  # 0: not present, 1: present, 3: uncertain
    class_index: int  # Specific class to annotate

class SpectrogramRequest(BaseModel):
    file_path: str
    clip_start: float
    clip_end: float
    color_mode: str = "viridis"  # viridis, gray_r, plasma, inferno, etc.
    freq_scale: str = "mel"  # mel or linear
    n_mels: int = 256  # number of mel bins (only used if freq_scale="mel")
    n_fft: int = 2048  # FFT window size
    hop_length: int = 128  # samples between successive frames
    window_size: Optional[int] = None  # window size in samples (if None, uses n_fft)
    fmin: Optional[float] = None  # minimum frequency (Hz)
    fmax: Optional[float] = None  # maximum frequency (Hz)
    bandpass_min: Optional[float] = None  # bandpass filter min frequency (Hz)
    bandpass_max: Optional[float] = None  # bandpass filter max frequency (Hz)

class TrainingParams(BaseModel):
    model_config = {'protected_namespaces': ()}

    n_steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 0.001
    model_type: int = 2
    verbose: bool = True
    weak_neg_weight: float = 0.05
    enable_early_stopping: bool = True
    enable_lr_reduction: bool = True
    lr_redux: float = 0.5
    patience: int = 5000
    lr_reduce_patience: int = 1000
    metric_for_tracking: str = 'cmap'  # 'cmap' or 'auc'

class ModelTrainingConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    training_audio_folder: str
    metadata_path: str
    test_data_mode: str = "split"  # "split" or "folder"
    test_split: Optional[float] = None
    test_audio_folder: Optional[str] = None
    random_state: int = 42
    model_save_path: str
    training_params: TrainingParams

def build_dataset_thread(config: DatasetConfig):
    """Thread function to build dataset in background"""
    try:
        building_state["status"] = "building"
        building_state["message"] = "Finding audio files..."
        building_state["progress"] = 5
        
        # Set backend configuration
        os.environ["BACKEND"] = config.backend_model
        app_state["backend_model"] = config.backend_model
        app_state["class_map"] = {item.name: item.value for item in config.class_map}
        app_state["save_path"] = config.save_path
        
        # Find audio files using Path for cross-platform compatibility
        audio_folder = Path(config.audio_folder)
        if not audio_folder.exists():
            building_state["status"] = "error"
            building_state["message"] = "Audio folder not found"
            return
        
        # Use Path.glob for cross-platform file discovery
        files = list(audio_folder.glob("**/*.wav"))
        files.extend(list(audio_folder.glob("**/*.mp3")))
        files.extend(list(audio_folder.glob("**/*.WAV")))  # Case variations
        files.extend(list(audio_folder.glob("**/*.MP3")))
        files = [str(f) for f in files]
        
        if not files:
            building_state["status"] = "error"
            building_state["message"] = "No audio files found"
            return
        
        building_state["total_files"] = len(files)
        building_state["message"] = f"Processing {len(files)} audio files..."
        building_state["progress"] = 10
        
        # Create embeddings and labels for evaluation datasets
        embeddings_path = Path(config.save_path) / "embeddings.pkl"
        if embeddings_path.exists():
            building_state["message"] = "Loading existing embeddings..."
            building_state["progress"] = 80
            with open(embeddings_path, "rb") as f:
                embeddings_data = pickle.load(f)
                # Handle both old format (just embeddings) and new format (dict with embeddings and labels)
                if isinstance(embeddings_data, dict) and 'embeddings' in embeddings_data:
                    embeddings = embeddings_data['embeddings']
                    labels = embeddings_data.get('labels', None)
                else:
                    embeddings = embeddings_data
                    labels = None
        else:
            building_state["message"] = "Generating embeddings..."
            building_state["progress"] = 20
            
            embeddings = u.load_and_preprocess(files)
            labels = None
            
            building_state["progress"] = 60
            
            # For evaluation datasets, extract labels from filenames
            if config.is_evaluation_dataset:
                building_state["message"] = "Extracting labels from filenames..."
                class_map_dict = {item.name: item.value for item in config.class_map}
                labels = []
                for file_path in files:
                    file_label = u.get_label(class_map_dict, file_path)
                    labels.append(file_label)
                labels = np.array(labels)
            
            building_state["progress"] = 70
            building_state["message"] = "Saving embeddings..."
            
            os.makedirs(config.save_path, exist_ok=True)
            
            # Save embeddings and labels together for evaluation datasets
            if config.is_evaluation_dataset and labels is not None:
                embeddings_data = {
                    'embeddings': embeddings,
                    'labels': labels
                }
                with open(embeddings_path, "wb") as f:
                    pickle.dump(embeddings_data, f)
            else:
                with open(embeddings_path, "wb") as f:
                    pickle.dump(embeddings, f)
        
        app_state["embeddings"] = embeddings
        
        building_state["message"] = "Creating audio database..."
        building_state["progress"] = 85
        
        # Create audio database with number of classes
        class_map = {item.name: item.value for item in config.class_map}
        num_classes = len(class_map)
        audio_db = db.Audio_DB(num_classes=num_classes)
        audio_db.class_map = class_map
        embedding_index = 0  # Track embedding indices as clips are created
        
        for i, file_path in enumerate(files):
            building_state["processed_files"] = i + 1
            building_state["progress"] = 85 + (10 * i // len(files))
            
            import soundfile as sf
            f = sf.SoundFile(file_path)
            duration_sec = f.frames / f.samplerate
            file_name = Path(file_path).stem
            
            # Add file and clips using new structure
            audio_db.add_file_and_clips(
                file_name=file_name,
                file_path=str(file_path),
                duration_sec=duration_sec,
                sampling_rate=cfg.TARGET_SR,
                window_size=cfg.WINDOW,
                embedding_start_index=embedding_index
            )
            
            # Calculate how many clips were created for this file
            clips_in_file = int(np.ceil(duration_sec / cfg.WINDOW))
            embedding_index += clips_in_file
        
        app_state["audio_db"] = audio_db
        
        building_state["message"] = "Saving database..."
        building_state["progress"] = 95
        
        # Save database (will save all three tables)
        db_path = Path(config.save_path) / "audio_database.parquet"
        audio_db.save_db(str(db_path))
        
        building_state["message"] = f"Saving metadata to {config.save_path}/metadata.json..."
        building_state["progress"] = 98
        
        # Create and save metadata
        metadata = {
            "dataset_info": {
                "creation_date": datetime.now().isoformat(),
                "dataset_type": "evaluation" if config.is_evaluation_dataset else "active_learning",
                "backend_model": config.backend_model,
                "audio_folder": config.audio_folder,
                "save_path": config.save_path,
                "has_labels": config.is_evaluation_dataset and labels is not None
            },
            "class_map": {item.name: item.value for item in config.class_map},
            "statistics": {
                "total_files": len(files),
                "total_clips": len(audio_db.clips_df),
                "window_size": cfg.WINDOW,
                "sample_rate": cfg.TARGET_SR
            },
            "file_paths": {
                "embeddings": str(embeddings_path),
                "database": str(db_path),
                "metadata": str(Path(config.save_path) / "metadata.json")
            }
        }
        
        # Save metadata to JSON file
        metadata_path = Path(config.save_path) / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"✗ Error saving metadata to {metadata_path}: {e}")
            # Continue without failing the whole process
        
        # Complete
        dataset_type = "evaluation" if config.is_evaluation_dataset else "active_learning"
        labels_info = f" with labels extracted from filenames" if config.is_evaluation_dataset else ""
        
        building_state["status"] = "completed"
        building_state["message"] = f"{dataset_type.title()} dataset created with {len(files)} files and {len(audio_db.clips_df)} clips{labels_info}"
        building_state["progress"] = 100
        building_state["files_count"] = len(files)
        building_state["clips_count"] = len(audio_db.clips_df)
        building_state["is_evaluation_dataset"] = config.is_evaluation_dataset
        building_state["has_labels"] = config.is_evaluation_dataset and labels is not None
        
    except Exception as e:
        building_state["status"] = "error"
        building_state["message"] = f"Error creating dataset: {str(e)}"
        building_state["progress"] = 0

# Dataset Builder endpoints
@app.post("/api/dataset/create")
async def create_dataset(config: DatasetConfig):
    """Create a new dataset with embeddings and database"""
    try:
        # Check if already building
        if building_state["status"] == "building":
            raise HTTPException(status_code=400, detail="Dataset creation already in progress")
        
        # Reset building state completely
        building_state.update({
            "status": "building",
            "message": "Starting dataset creation...", 
            "progress": 0,
            "total_files": 0,
            "processed_files": 0,
            "files_count": 0,
            "clips_count": 0,
            "is_evaluation_dataset": False,
            "has_labels": False
        })
        
        # Start building in background thread
        thread = threading.Thread(target=build_dataset_thread, args=(config,))
        thread.daemon = True
        thread.start()
        
        # Give the thread a moment to start and update the initial status
        import time
        time.sleep(0.1)
        
        return {
            "status": "started",
            "message": "Dataset creation started in background"
        }
        
    except Exception as e:
        building_state["status"] = "error"
        building_state["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/building-status")
async def get_building_status():
    """Get current building status"""
    return {
        "status": building_state["status"],
        "message": building_state["message"],
        "progress": building_state["progress"],
        "total_files": building_state["total_files"],
        "processed_files": building_state["processed_files"],
        "files_count": building_state.get("files_count", 0),
        "clips_count": building_state.get("clips_count", 0),
        "is_evaluation_dataset": building_state.get("is_evaluation_dataset", False),
        "has_labels": building_state.get("has_labels", False)
    }

@app.get("/api/dataset/status")
async def get_dataset_status():
    """Get current dataset status"""
    if app_state["audio_db"] is None:
        return {"loaded": False}
    
    # Try to load metadata if dataset path is available
    metadata = None
    if app_state["dataset_path"]:
        metadata_path = Path(app_state["dataset_path"]) / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass
    
    status = {
        "loaded": True,
        "clips_count": len(app_state["audio_db"].clips_df),
        "backend_model": app_state["backend_model"],
        "class_map": app_state["class_map"],
        "has_classifier": app_state["classifier_model"] is not None
    }
    
    # Add metadata information if available
    if metadata:
        dataset_info = metadata.get("dataset_info", {})
        status.update({
            "creation_date": dataset_info.get("creation_date", ""),
            "dataset_type": dataset_info.get("dataset_type", ""),
            "original_audio_folder": dataset_info.get("audio_folder", ""),
            "has_labels": dataset_info.get("has_labels", False),
            "metadata": metadata
        })
    
    return status

# Multiclass class selection endpoint
@app.get("/api/active-learning/classes")
async def get_available_classes():
    """Get available classes for multiclass selection"""
    if app_state["class_map"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    # Convert class map to list of options sorted by value
    classes = [{"name": name, "value": value} for name, value in app_state["class_map"].items()]
    classes.sort(key=lambda x: x["value"])

    return {
        "classes": classes,
        "current_class": app_state.get("current_class_index", 0)
    }


@app.post("/api/active-learning/select-class")
async def select_class(class_index: int):
    """Select a class for active learning and update scores/annotations"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    if app_state["class_map"] is None:
        raise HTTPException(status_code=400, detail="No class map available")
    
    num_classes = len(app_state["class_map"])
    if class_index < 0 or class_index >= num_classes:
        raise HTTPException(status_code=400, detail=f"Class index {class_index} out of range (0-{num_classes-1})")
    
    try:
        # Update the database to show scores and annotations for the selected class
        app_state["audio_db"].update_class_scores_and_annotations(class_index)
        app_state["current_class_index"] = class_index
        
        # Get class name for response
        class_names = list(app_state["class_map"].keys())
        class_name = class_names[class_index] if class_index < len(class_names) else f"Class {class_index}"
        
        return {
            "status": "success",
            "message": f"Selected class: {class_name}",
            "class_index": class_index,
            "class_name": class_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ReviewModeRequest(BaseModel):
    review_mode: str

@app.post("/api/active-learning/set-review-mode")
async def set_review_mode(request: ReviewModeRequest):
    """Set the review mode for clip selection"""
    valid_modes = ["random", "top_down", "top_10+score_quantiles", "review_annotated"]

    if request.review_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid review mode. Must be one of: {', '.join(valid_modes)}"
        )

    app_state["review_mode"] = request.review_mode

    return {
        "status": "success",
        "message": f"Review mode set to: {request.review_mode}",
        "review_mode": request.review_mode
    }

@app.get("/api/active-learning/label-statistics")
async def get_label_statistics():
    """Get statistics about strong vs weak labels"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        stats = app_state["audio_db"].get_label_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database Viewer endpoints
@app.get("/api/database/info")
async def get_database_info():
    """Get basic information about the loaded database"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Use legacy df for backwards compatibility, but also include new structure info
        audio_db = app_state["audio_db"]
        df = audio_db.df if audio_db.df is not None else audio_db.clips_df
        
        # Get basic info
        info = {
            "total_rows": len(df),
            "columns": df.columns,
            "schema": {col: str(df.schema[col]) for col in df.columns},
            "num_classes": getattr(audio_db, "num_classes", 1),
            "class_map": app_state.get("class_map", {}),
            "new_structure": {
                "files_count": len(audio_db.files_df),
                "clips_count": len(audio_db.clips_df),
                "annotations_count": len(audio_db.annotations_df)
            }
        }
        
        return {"status": "success", "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/data")
async def get_database_data(
    limit: int = 100,
    offset: int = 0,
    columns: str = None,
    filter_column: str = None,
    filter_value: str = None
):
    """Get database data with optional filtering and pagination"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Use legacy df for database viewer compatibility
        audio_db = app_state["audio_db"]
        df = audio_db.df if audio_db.df is not None else audio_db.clips_df
        
        # Apply column selection
        if columns:
            selected_columns = [col.strip() for col in columns.split(",")]
            # Validate columns exist
            valid_columns = [col for col in selected_columns if col in df.columns]
            if valid_columns:
                df = df.select(valid_columns)
        
        # Apply filtering
        if filter_column and filter_value:
            if filter_column in df.columns:
                # Handle different data types
                if df.schema[filter_column] in [pl.Int32, pl.Int64]:
                    try:
                        filter_val = int(filter_value)
                        df = df.filter(pl.col(filter_column) == filter_val)
                    except ValueError:
                        pass
                elif df.schema[filter_column] in [pl.Float32, pl.Float64]:
                    try:
                        filter_val = float(filter_value)
                        df = df.filter(pl.col(filter_column) == filter_val)
                    except ValueError:
                        pass
                else:  # String columns
                    df = df.filter(pl.col(filter_column).str.contains(filter_value))
        
        # Apply pagination
        total_rows = len(df)
        df_page = df.slice(offset, limit)
        
        # Convert to dict format, handling list columns
        data = []
        for row in df_page.to_dicts():
            # Convert list columns to string representation for display
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, list):
                    processed_row[key] = str(value)
                elif hasattr(value, 'isoformat'):  # datetime
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            data.append(processed_row)
        
        return {
            "status": "success",
            "data": data,
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "columns": df.columns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/column-stats")
async def get_column_statistics(column: str):
    """Get statistics for a specific column"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Use legacy df for database viewer compatibility
        audio_db = app_state["audio_db"]
        df = audio_db.df if audio_db.df is not None else audio_db.clips_df
        
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
        
        col_data = df[column]
        stats = {}
        
        # Basic stats
        stats["total_count"] = len(col_data)
        stats["null_count"] = col_data.null_count()
        stats["data_type"] = str(df.schema[column])
        
        # Type-specific stats
        if df.schema[column] in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            stats["min"] = col_data.min()
            stats["max"] = col_data.max()
            stats["mean"] = col_data.mean()
            stats["std"] = col_data.std()
        elif df.schema[column] == pl.Utf8:
            stats["unique_count"] = col_data.n_unique()
            # Get value counts for string columns
            value_counts = col_data.value_counts().sort("counts", descending=True).limit(10)
            stats["top_values"] = value_counts.to_dicts()
        elif df.schema[column] == pl.List:
            # For list columns, get some sample values
            sample_values = col_data.drop_nulls().slice(0, 5).to_list()
            stats["sample_values"] = [str(val) for val in sample_values]
            
        return {"status": "success", "column": column, "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New API endpoints demonstrating three-table structure capabilities
@app.get("/api/database/files")
async def get_files():
    """Get all files in the database"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        files_df = audio_db.get_files_df()
        
        return {
            "status": "success", 
            "files": files_df.to_dicts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/clips-with-annotations")
async def get_clips_with_annotations(class_name: str = None):
    """Get clips with their annotations, optionally filtered by class"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        clips_df = audio_db.get_clips_with_annotations(class_name)
        
        return {
            "status": "success", 
            "clips": clips_df.to_dicts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/review-clips")
async def get_review_clips():
    """Get clips that contain at least one label (for review)"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        
        # Get clips that have any annotations
        annotated_clip_ids = audio_db.annotations_df.select("clip_id").unique()
        
        # Join with clips and files to get full information
        review_clips = audio_db.clips_df.join(
            annotated_clip_ids,
            on="clip_id",
            how="inner"
        ).join(
            audio_db.files_df,
            on="file_id",
            how="left"
        )
        
        return {
            "status": "success",
            "message": f"Found {len(review_clips)} clips with annotations for review",
            "clips": review_clips.to_dicts()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/annotation-summary")
async def get_annotation_summary():
    """Get summary statistics about annotations across all classes"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        
        # Get annotation counts by class and label
        annotation_summary = {}
        
        for row in audio_db.annotations_df.iter_rows(named=True):
            class_name = row['class_name']
            label = row['label']
            
            if class_name not in annotation_summary:
                annotation_summary[class_name] = {
                    "present": 0,
                    "not_present": 0, 
                    "uncertain": 0,
                    "total": 0
                }
            
            annotation_summary[class_name][label] += 1
            annotation_summary[class_name]["total"] += 1
        
        # Add overall statistics
        total_files = len(audio_db.files_df)
        total_clips = len(audio_db.clips_df)
        total_annotations = len(audio_db.annotations_df)
        annotated_clips = len(audio_db.annotations_df.select("clip_id").unique())
        
        return {
            "status": "success",
            "summary": {
                "database_structure": {
                    "total_files": total_files,
                    "total_clips": total_clips,
                    "total_annotations": total_annotations,
                    "annotated_clips": annotated_clips,
                    "unannotated_clips": total_clips - annotated_clips
                },
                "annotations_by_class": annotation_summary
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Table-specific data endpoints for Database Viewer
@app.get("/api/database/table-data")
async def get_table_data(
    table: str = "clips",
    limit: int = 100,
    offset: int = 0,
    columns: str = None,
    filter_column: str = None,
    filter_value: str = None
):
    """Get data from a specific database table"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        
        # Select the appropriate table/view
        if table == "files":
            df = audio_db.files_df
        elif table == "clips": 
            df = audio_db.clips_df
        elif table == "annotations":
            df = audio_db.annotations_df
        elif table == "clips_with_files":
            df = audio_db.get_clips_with_files()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown table: {table}")
        
        # Apply column selection
        if columns:
            selected_columns = [col.strip() for col in columns.split(",")]
            available_columns = [col for col in selected_columns if col in df.columns]
            if available_columns:
                df = df.select(available_columns)
        
        # Apply filtering
        if filter_column and filter_value and filter_column in df.columns:
            df = df.filter(pl.col(filter_column).cast(pl.Utf8).str.contains(filter_value))
        
        # Get total count before pagination
        total_rows = len(df)
        
        # Apply pagination
        if offset > 0:
            df = df.slice(offset, limit)
        else:
            df = df.head(limit)
        
        # Convert to dict format for frontend
        data = df.to_dicts()
        
        return {
            "status": "success",
            "data": data,
            "total_rows": total_rows,
            "table": table
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/table-info")  
async def get_table_info(table: str = "clips"):
    """Get schema info for a specific table"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        audio_db = app_state["audio_db"]
        
        # Select the appropriate table
        if table == "files":
            df = audio_db.files_df
        elif table == "clips":
            df = audio_db.clips_df  
        elif table == "annotations":
            df = audio_db.annotations_df
        elif table == "clips_with_files":
            df = audio_db.get_clips_with_files()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown table: {table}")
        
        info = {
            "table": table,
            "total_rows": len(df),
            "columns": df.columns,
            "schema": {col: str(df.schema[col]) for col in df.columns}
        }
        
        return {"status": "success", "info": info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Active Learning endpoints
@app.post("/api/active-learning/populate-embedding-indices")
async def populate_embedding_indices():
    """Populate embedding indices for the current dataset"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Get current database
        audio_db = app_state["audio_db"]
        
        # Auto-populate embedding indices (0, 1, 2, ...)
        audio_db.auto_populate_embedding_indices()
        
        return {"status": "success", "message": f"Populated embedding indices for {len(audio_db.clips_df)} clips"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/load-dataset")
async def load_dataset(dataset_path: str):
    """Load an existing dataset for active learning"""
    print(f"DEBUG: load_dataset called with path: {dataset_path}")
    try:
        dataset_path = Path(dataset_path)
        
        # Check for new three-table format
        files_path = dataset_path / "files.parquet"
        clips_path = dataset_path / "clips.parquet"
        annotations_path = dataset_path / "annotations.parquet"
        embeddings_path = dataset_path / "embeddings.pkl"
        metadata_path = dataset_path / "metadata.json"
        
        # Use fake db_path for compatibility with existing logic
        db_path = dataset_path / "audio_database.parquet"
        
        if not (files_path.exists() and clips_path.exists() and annotations_path.exists()):
            raise HTTPException(status_code=404, detail="Three-table database files not found. Expected: files.parquet, clips.parquet, annotations.parquet")
        
        # Load metadata if available
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Load class map from metadata
            if "class_map" in metadata:
                app_state["class_map"] = metadata["class_map"]
            
            # Load backend model from metadata
            if "dataset_info" in metadata and "backend_model" in metadata["dataset_info"]:
                app_state["backend_model"] = metadata["dataset_info"]["backend_model"]
                os.environ["BACKEND"] = metadata["dataset_info"]["backend_model"]
        
        # Load database with number of classes from metadata
        class_map = metadata.get("class_map", {}) if metadata else {}
        num_classes = len(class_map) if class_map else 1
        audio_db = db.Audio_DB(num_classes=num_classes)
        if class_map:
            audio_db.class_map = class_map
        
        # Load from the three-table structure using load_db method
        # Pass any file path from the dataset directory - load_db will find the three tables
        audio_db.load_db(str(files_path))
        
        app_state["audio_db"] = audio_db
        
        # Load embeddings
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                embeddings_data = pickle.load(f)
                # Handle both old format (just embeddings) and new format (dict with embeddings and labels)
                if isinstance(embeddings_data, dict) and 'embeddings' in embeddings_data:
                    app_state["embeddings"] = embeddings_data['embeddings']
                else:
                    app_state["embeddings"] = embeddings_data
        
        app_state["dataset_path"] = str(dataset_path)
        
        message = "Dataset loaded successfully"
        if metadata:
            dataset_type = metadata.get("dataset_info", {}).get("dataset_type", "unknown")
            creation_date = metadata.get("dataset_info", {}).get("creation_date", "unknown")
            message += f" ({dataset_type} dataset created {creation_date[:10]})"
        
        return {
            "status": "success",
            "clips_count": len(audio_db.clips_df),
            "message": message,
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/load-classifier")
async def load_classifier(classifier_path: str):
    """Load a pretrained classifier"""
    try:
        if not Path(classifier_path).exists():
            raise HTTPException(status_code=404, detail="Classifier file not found")
        
        import tensorflow as tf
        classifier_model = tf.keras.models.load_model(classifier_path)
        app_state["classifier_model"] = classifier_model
        
        # Update scores if we have embeddings and class_map
        if (app_state["embeddings"] and app_state["class_map"] and 
            app_state["audio_db"]):
            
            try:
                # Handle embeddings - they are stored as list of tensors with different frame counts
                embeddings_list = app_state["embeddings"]
                print(f"DEBUG: embeddings_list type: {type(embeddings_list)}")
                print(f"DEBUG: embeddings_list length: {len(embeddings_list) if hasattr(embeddings_list, '__len__') else 'No length'}")
                
                if not isinstance(embeddings_list, list) or len(embeddings_list) == 0:
                    raise ValueError("Invalid embeddings format")
                
                print(f"DEBUG: Starting frame-wise processing for {len(embeddings_list)} embedding tensors")
                
                # Get database info to understand clip structure
                audio_db = app_state["audio_db"]
                total_clips = len(audio_db.clips_df)
                print(f"DEBUG: Database has {total_clips} clips")
                
                # Get frame-wise predictions for all embeddings and map to clips
                all_clip_predictions = []
                
                # Get clips with file information
                clips_with_files = audio_db.get_clips_with_files()
                
                # Get unique files and their clips
                file_groups = clips_with_files.group_by("file_name", maintain_order=True)
                
                for file_idx, (file_group_key, file_clips_df) in enumerate(file_groups):
                    if file_idx >= len(embeddings_list):
                        print(f"WARNING: More files in database than embeddings available")
                        break
                        
                    file_name = file_group_key[0]  # file_name is the grouping key
                    num_clips_for_file = len(file_clips_df)
                    print(f"DEBUG: File {file_name} has {num_clips_for_file} clips")
                    
                    # Get embedding tensor for this file
                    emb_tensor = embeddings_list[file_idx]
                    
                    # Convert to numpy if it's a TensorFlow tensor
                    if hasattr(emb_tensor, 'numpy'):
                        emb_array = emb_tensor.numpy()
                    else:
                        emb_array = np.array(emb_tensor)
                    
                    print(f"DEBUG: Embedding shape for {file_name}: {emb_array.shape}")
                    
                    # emb_array should have shape (num_frames, embedding_dim)
                    if emb_array.ndim == 1:
                        emb_array = emb_array.reshape(1, -1)
                    elif emb_array.ndim > 2:
                        emb_array = emb_array.reshape(-1, emb_array.shape[-1])
                    
                    # Get frame-wise predictions
                    frame_logits = classifier_model(emb_array)
                    frame_predictions = tf.sigmoid(frame_logits).numpy()
                    
                    print(f"DEBUG: Frame predictions shape: {frame_predictions.shape}")
                    
                    # Map frame predictions to clips
                    # If we have more clips than frames, repeat predictions
                    # If we have more frames than clips, average frames for each clip
                    num_frames = frame_predictions.shape[0]
                    
                    if num_frames >= num_clips_for_file:
                        # More frames than clips - average frames for each clip
                        frames_per_clip = num_frames // num_clips_for_file
                        for clip_idx in range(num_clips_for_file):
                            start_frame = clip_idx * frames_per_clip
                            end_frame = min((clip_idx + 1) * frames_per_clip, num_frames)
                            clip_pred = np.mean(frame_predictions[start_frame:end_frame], axis=0)
                            all_clip_predictions.append(clip_pred.tolist())
                    else:
                        # Fewer frames than clips - interpolate or repeat
                        for clip_idx in range(num_clips_for_file):
                            frame_idx = min(clip_idx * num_frames // num_clips_for_file, num_frames - 1)
                            clip_pred = frame_predictions[frame_idx]
                            all_clip_predictions.append(clip_pred.tolist())
                
                print(f"DEBUG: Generated {len(all_clip_predictions)} predictions for {total_clips} clips")
                
                # Ensure we have exactly the right number of predictions
                if len(all_clip_predictions) != total_clips:
                    print(f"WARNING: Prediction count mismatch. Padding or truncating...")
                    if len(all_clip_predictions) < total_clips:
                        # Pad with last prediction
                        last_pred = all_clip_predictions[-1] if all_clip_predictions else [0.0]
                        while len(all_clip_predictions) < total_clips:
                            all_clip_predictions.append(last_pred)
                    else:
                        # Truncate
                        all_clip_predictions = all_clip_predictions[:total_clips]
                
                # Populate multiclass predictions in database
                app_state["audio_db"].populate_multiclass_predictions(all_clip_predictions)
                
                # Update scores for the first class (default)
                app_state["audio_db"].update_class_scores_and_annotations(0)
                app_state["current_class_index"] = 0
                
            except Exception as embed_error:
                import traceback
                print(f"ERROR: Could not update scores with classifier: {embed_error}")
                print(f"ERROR: Full traceback: {traceback.format_exc()}")
                # Still consider classifier loading successful even if score update fails
                raise embed_error  # Temporarily re-raise to see the full error
        
        return {"status": "success", "message": "Classifier loaded and scores updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/get-clips")
async def get_clips(filter_config: FilterConfig):
    """Get filtered clips for annotation"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        # Import the WebAnnotationInterface
        from modules.display_web import WebAnnotationInterface

        # Create annotation interface with current review mode
        review_mode = app_state.get("review_mode", "random")
        annotation_interface = WebAnnotationInterface(
            app_state["audio_db"],
            review_mode=review_mode
        )

        # For review_annotated mode, get only annotated clips
        if review_mode == "review_annotated":
            # Get current class name
            class_names = app_state["audio_db"].get_class_names()
            current_class_index = getattr(app_state["audio_db"], '_current_class_index', 0)
            class_name = class_names[current_class_index]

            # Get annotated clips for the current class
            filtered_df = annotation_interface.get_annotated_clips_for_review(class_name)
        else:
            # Get filtered clips using the annotation interface
            filtered_df = annotation_interface.get_filtered_clips(
                score_min=filter_config.score_min,
                score_max=filter_config.score_max,
                annotation_filter=filter_config.annotation_filter
            )

        # For quantile mode, generate the full round and return all clips with metadata
        if review_mode == "top_10+score_quantiles":
            # Generate the round
            round_clips_indices, round_metadata = annotation_interface._generate_quantile_round_clips()

            # Build list of clips with their round metadata
            clips_with_metadata = []
            for i, clip_index in enumerate(round_clips_indices):
                clip_row = filtered_df.row(clip_index)
                clip_dict = dict(zip(filtered_df.columns, clip_row))

                # Convert types and add round metadata
                converted_dict = {}
                for key, value in clip_dict.items():
                    if hasattr(value, 'item'):
                        converted_dict[key] = value.item()
                    elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
                        converted_dict[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                    else:
                        converted_dict[key] = value

                # Add round metadata
                converted_dict["round_position"] = i + 1
                converted_dict["round_total"] = len(round_clips_indices)
                converted_dict["round_category"] = round_metadata[i]["category"]
                converted_dict["round_category_label"] = round_metadata[i]["category_label"]

                clips_with_metadata.append(converted_dict)

            return {
                "clips": clips_with_metadata,
                "total_count": len(clips_with_metadata),
                "next_clip": clips_with_metadata[0] if clips_with_metadata else None,
                "review_mode": review_mode,
                "is_quantile_round": True
            }

        # For other modes, get next clip normally
        next_clip = annotation_interface.get_next_clip()
        
        # Convert filtered dataframe to list of dicts with proper type conversion
        clips = []
        for row_dict in filtered_df.to_dicts():
            # Convert numpy/polars types to native Python types
            converted_dict = {}
            for key, value in row_dict.items():
                if hasattr(value, 'item'):  # numpy scalar
                    converted_dict[key] = value.item()
                elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):  # numpy array elements
                    converted_dict[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                else:
                    converted_dict[key] = value
            clips.append(converted_dict)
        
        return {
            "clips": clips,
            "total_count": len(clips),
            "next_clip": next_clip,
            "review_mode": review_mode
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/annotate")
async def annotate_clip(request: AnnotationRequest):
    """Annotate a clip"""
    print(f"DEBUG: Received annotation request - clip_id: {request.clip_id}, annotation: {request.annotation}")
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Check if clip_id is in new UUID format or legacy format
        if "|" in request.clip_id:
            # Legacy format: file_path|clip_start|clip_end
            parts = request.clip_id.split("|")
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Invalid legacy clip_id format")
            
            file_path, clip_start_str, clip_end_str = parts
            clip_start = float(clip_start_str)
            clip_end = float(clip_end_str)
            
            # Get the actual clip_id from the three-table structure
            clip_id = app_state["audio_db"].get_clip_id_by_legacy_info(file_path, clip_start, clip_end)
        else:
            # New UUID format - use directly
            clip_id = request.clip_id
        
        # Convert annotation value to semantic label
        annotation_map = {0: "not_present", 1: "present", 3: "uncertain", 4: "not_reviewed"}
        label = annotation_map.get(request.annotation, "not_reviewed")
        
        # Get current class information
        current_class_index = app_state.get("current_class_index", 0)
        if app_state["class_map"] is None:
            raise HTTPException(status_code=400, detail="No class map available")
        
        class_names = list(app_state["class_map"].keys())
        if current_class_index >= len(class_names):
            raise HTTPException(status_code=400, detail="Invalid class index")
        
        class_name = class_names[current_class_index]
        
        # Add annotation to the database
        print(f"DEBUG: Adding annotation - clip_id: {clip_id}, class_name: {class_name}, label: {label}")
        app_state["audio_db"].add_annotation(clip_id, class_name, label)
        
        # Auto-save the database after annotation
        dataset_path = app_state.get("dataset_path")
        if dataset_path:
            db_path = Path(dataset_path) / "audio_database.parquet"
            app_state["audio_db"].save_db(str(db_path))
            print(f"DEBUG: Auto-saved database to {db_path}")
        
        return {"status": "success", "message": "Annotation updated and saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/annotate-class")
async def annotate_specific_class(request: MultiClassAnnotationRequest):
    """Annotate a specific class for a clip"""
    print(f"DEBUG: Received multi-class annotation request - clip_id: {request.clip_id}, annotation: {request.annotation}, class_index: {request.class_index}")
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        # Check if clip_id is in new UUID format or legacy format
        if "|" in request.clip_id:
            # Legacy format: file_path|clip_start|clip_end
            parts = request.clip_id.split("|")
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Invalid legacy clip_id format")

            file_path, clip_start_str, clip_end_str = parts
            clip_start = float(clip_start_str)
            clip_end = float(clip_end_str)

            # Get the actual clip_id from the three-table structure
            clip_id = app_state["audio_db"].get_clip_id_by_legacy_info(file_path, clip_start, clip_end)
            if clip_id is None:
                raise HTTPException(status_code=404, detail="Clip not found")
        else:
            # New UUID format - use directly
            clip_id = request.clip_id

        # Convert annotation value to semantic label
        annotation_map = {0: "not_present", 1: "present", 3: "uncertain", 4: "not_reviewed"}
        label = annotation_map.get(request.annotation, "not_reviewed")

        # Get class information
        if app_state["class_map"] is None:
            raise HTTPException(status_code=400, detail="No class map available")

        class_names = list(app_state["class_map"].keys())
        if request.class_index >= len(class_names) or request.class_index < 0:
            raise HTTPException(status_code=400, detail="Invalid class index")

        class_name = class_names[request.class_index]

        # Add annotation to the database
        print(f"DEBUG: Adding annotation - clip_id: {clip_id}, class_name: {class_name}, label: {label}")
        app_state["audio_db"].add_annotation(clip_id, class_name, label)

        # Auto-save the database after annotation
        dataset_path = app_state.get("dataset_path")
        if dataset_path:
            db_path = Path(dataset_path) / "audio_database.parquet"
            app_state["audio_db"].save_db(str(db_path))
            print(f"DEBUG: Auto-saved database to {db_path}")

        return {"status": "success", "message": f"Annotation updated for class '{class_name}' and saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/annotate-other-classes")
async def annotate_other_classes_as_absent(request: AnnotationRequest):
    """Mark all other classes (except current) as 'not present' for a clip"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Check if clip_id is in new UUID format or legacy format  
        if "|" in request.clip_id:
            # Legacy format: file_path|clip_start|clip_end
            parts = request.clip_id.split("|")
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Invalid legacy clip_id format")
            
            file_path, clip_start_str, clip_end_str = parts
            clip_start = float(clip_start_str)
            clip_end = float(clip_end_str)
            
            # Get the actual clip_id from the three-table structure
            clip_id = app_state["audio_db"].get_clip_id_by_legacy_info(file_path, clip_start, clip_end)
        else:
            # New UUID format - use directly
            clip_id = request.clip_id
        
        current_class_index = app_state.get("current_class_index", 0)
        if app_state["class_map"] is None:
            raise HTTPException(status_code=400, detail="No class map available")
        
        class_names = list(app_state["class_map"].keys())
        
        # Mark all other classes as "not present"
        classes_updated = 0
        for class_idx, class_name in enumerate(class_names):
            if class_idx != current_class_index:
                app_state["audio_db"].add_annotation(clip_id, class_name, "not_present")
                classes_updated += 1
        
        # Auto-save the database after annotations
        dataset_path = app_state.get("dataset_path")
        if dataset_path:
            db_path = Path(dataset_path) / "audio_database.parquet"
            app_state["audio_db"].save_db(str(db_path))
            print(f"DEBUG: Auto-saved database after annotating {classes_updated} other classes")
        
        return {
            "status": "success", 
            "message": f"Marked {classes_updated} other classes as 'Not Present' and saved",
            "classes_updated": classes_updated,
            "current_class": current_class_index
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active-learning/clip-labels")
async def get_clip_labels(clip_id: str):
    """Get all class labels for a specific clip using the new three-table structure"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        print(f"DEBUG: get_clip_labels called with clip_id: {clip_id}")
        
        # Get annotations for this clip from the annotations table
        clip_annotations = app_state["audio_db"].annotations_df.filter(
            pl.col("clip_id") == clip_id
        )
        
        print(f"DEBUG: Found {len(clip_annotations)} annotations for clip {clip_id}")
        
        # Build class information
        class_map = app_state.get("class_map", {})
        class_names = list(class_map.keys())
        current_class_index = app_state.get("current_class_index", 0)
        
        class_labels = []
        
        # Create a dict of annotations by class name for quick lookup
        annotations_by_class = {}
        if len(clip_annotations) > 0:
            for row in clip_annotations.iter_rows():
                annotation_id, clip_id_db, class_name, label, annotated_at = row
                annotations_by_class[class_name] = label
        
        print(f"DEBUG: Annotations by class: {annotations_by_class}")
        
        # Build labels for all classes
        for i, class_name in enumerate(class_names):
            annotation_label = annotations_by_class.get(class_name)
            
            # Determine label text based on annotation
            if annotation_label == "present":
                label_text = "Present"
            elif annotation_label == "not_present":
                label_text = "Not Present"
            elif annotation_label == "uncertain":
                label_text = "Uncertain"
            else:
                label_text = "Unlabelled"
            
            class_labels.append({
                "class_name": class_name,
                "label_text": label_text,
                "is_current": i == current_class_index
            })
        
        print(f"DEBUG: Returning class_labels: {class_labels}")
        
        return {
            "status": "success",
            "class_labels": class_labels
        }
        
    except Exception as e:
        print(f"ERROR in get_clip_labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/save-database")
async def save_database():
    """Save the current database"""
    if app_state["audio_db"] is None or app_state["dataset_path"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        db_path = Path(app_state["dataset_path"]) / "audio_database.parquet"
        app_state["audio_db"].save_db(str(db_path))
        
        return {"status": "success", "message": "Database saved"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active-learning/check-export-folder")
async def check_export_folder(export_path: str):
    """Check export folder for existing clips before exporting"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        from pathlib import Path

        # Scan existing exports
        existing_clips = app_state["audio_db"]._scan_existing_exports(export_path)

        return {
            "status": "success",
            "existing_clips_count": len(existing_clips),
            "folder_exists": Path(export_path).exists()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/export-clips")
async def export_clips(export_path: str, annotation_slug: str = "export"):
    """Export annotated clips as WAV files with smart incremental updates"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        # Pass class_map to the database for metadata export
        if app_state["class_map"]:
            app_state["audio_db"].class_map = app_state["class_map"]

        # Get export statistics
        stats = app_state["audio_db"].export_wav_clips(export_path, annotation_slug)

        # Build message based on what happened
        if stats["clips_new"] > 0 and stats["clips_updated"] > 0:
            action_msg = f"Exported {stats['clips_new']} new clips and updated {stats['clips_updated']} existing clips"
        elif stats["clips_new"] > 0:
            action_msg = f"Exported {stats['clips_new']} new clips"
        elif stats["clips_updated"] > 0:
            action_msg = f"Updated {stats['clips_updated']} existing clips"
        else:
            action_msg = "No new or updated clips"

        if stats["clips_skipped"] > 0:
            action_msg += f" (skipped {stats['clips_skipped']} unchanged)"

        return {
            "status": "success",
            "positive_clips": stats["positive_clips"],
            "negative_clips": stats["negative_clips"],
            "uncertain_clips": stats["uncertain_clips"],
            "total_clips": stats["total_clips"],
            "clips_new": stats["clips_new"],
            "clips_updated": stats["clips_updated"],
            "clips_skipped": stats["clips_skipped"],
            "existing_in_folder": stats["existing_in_folder"],
            "message": action_msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/active-learning/review-clips")
async def get_review_clips():
    """Get annotated clips for review mode"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        # Import the WebAnnotationInterface
        from modules.display_web import WebAnnotationInterface

        # Get current class information
        current_class_index = app_state.get("current_class_index", 0)
        if app_state["class_map"] is None:
            raise HTTPException(status_code=400, detail="No class map available")

        class_names = list(app_state["class_map"].keys())
        if current_class_index >= len(class_names):
            raise HTTPException(status_code=400, detail="Invalid class index")

        class_name = class_names[current_class_index]

        # Create annotation interface
        review_mode = app_state.get("review_mode", "top_down")
        annotation_interface = WebAnnotationInterface(
            app_state["audio_db"],
            review_mode=review_mode
        )

        # Get annotated clips for the current class
        annotated_df = annotation_interface.get_annotated_clips_for_review(class_name)

        # Get the next clip for review
        next_clip = annotation_interface.get_next_clip()

        # Convert dataframe to list of dicts
        clips = []
        for row_dict in annotated_df.to_dicts():
            # Convert numpy/polars types to native Python types
            converted_dict = {}
            for key, value in row_dict.items():
                if hasattr(value, 'item'):  # numpy scalar
                    converted_dict[key] = value.item()
                elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
                    converted_dict[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                else:
                    converted_dict[key] = value
            clips.append(converted_dict)

        return {
            "clips": clips,
            "total_count": len(clips),
            "next_clip": next_clip,
            "review_mode": review_mode,
            "class_name": class_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/active-learning/annotation")
async def delete_annotation(clip_id: str, class_name: str = None):
    """Delete an annotation for a specific clip and class"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        # If no class_name provided, use current class
        if class_name is None:
            current_class_index = app_state.get("current_class_index", 0)
            if app_state["class_map"] is None:
                raise HTTPException(status_code=400, detail="No class map available")

            class_names = list(app_state["class_map"].keys())
            if current_class_index >= len(class_names):
                raise HTTPException(status_code=400, detail="Invalid class index")

            class_name = class_names[current_class_index]

        # Delete the annotation
        print(f"DEBUG: Deleting annotation for clip_id={clip_id}, class_name={class_name}")
        app_state["audio_db"].delete_annotation(clip_id, class_name)

        # Auto-save the database
        dataset_path = app_state.get("dataset_path")
        if dataset_path:
            db_path = Path(dataset_path) / "audio_database.parquet"
            app_state["audio_db"].save_db(str(db_path))
            print(f"DEBUG: Auto-saved database after deletion to {db_path}")

        return {"status": "success", "message": "Annotation deleted and database saved"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spectrogram")
def generate_spectrogram(request: SpectrogramRequest):
    """Generate spectrogram data for a clip with caching"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
        import io
        import base64

        # Generate cache key (includes all parameters for proper cache invalidation)
        cache_key = get_spectrogram_cache_key(
            request.file_path,
            request.clip_start,
            request.clip_end,
            request.color_mode,
            request.freq_scale,
            request.n_mels,
            request.n_fft,
            request.hop_length,
            request.window_size,
            request.fmin,
            request.fmax,
            request.bandpass_min,
            request.bandpass_max
        )

        # Create cache directory if it doesn't exist
        SPECTROGRAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file_path = SPECTROGRAM_CACHE_DIR / cache_key

        # Check cache first (FAST PATH)
        if cache_file_path.exists():
            print(f"Spectrogram cache HIT: {cache_key}")
            # Update access time for LRU cleanup
            cache_file_path.touch()

            # Read cached PNG and convert to base64
            with open(cache_file_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()

            # Calculate metadata (lightweight operation)
            buffer_s = 1.0
            buffered_start = max(0, request.clip_start - buffer_s)

            import soundfile as sf
            f = sf.SoundFile(request.file_path)
            file_duration = f.frames / f.samplerate
            buffered_end = min(file_duration, request.clip_end + buffer_s)

            # Set frequency range from request or defaults
            fmin = request.fmin if request.fmin is not None else cfg.MIN_FREQ
            nyquist = cfg.MODEL_SR // 2
            fmax = request.fmax if request.fmax is not None else min(cfg.MAX_FREQ, nyquist)

            return {
                "spectrogram": f"data:image/png;base64,{image_data}",
                "metadata": {
                    "time_start": float(buffered_start),
                    "time_end": float(buffered_end),
                    "clip_start": float(request.clip_start),
                    "clip_end": float(request.clip_end),
                    "freq_min": float(fmin),
                    "freq_max": float(fmax),
                    "freq_scale": request.freq_scale,
                    "n_mels": request.n_mels if request.freq_scale == "mel" else None,
                    "n_fft": request.n_fft,
                    "hop_length": request.hop_length,
                    "window_size": request.window_size,
                    "bandpass_min": request.bandpass_min,
                    "bandpass_max": request.bandpass_max,
                    "db_min": -80.0,  # Approximate for cached spectrograms
                    "db_max": 0.0
                }
            }

        # Cache MISS - Generate spectrogram using new flexible function
        print(f"Spectrogram cache MISS: Generating for {request.file_path} ({request.clip_start}-{request.clip_end}s)")
        print(f"  Parameters: scale={request.freq_scale}, n_mels={request.n_mels}, n_fft={request.n_fft}, hop={request.hop_length}")
        if request.bandpass_min and request.bandpass_max:
            print(f"  Bandpass filter: {request.bandpass_min}-{request.bandpass_max} Hz")

        # Generate spectrogram with all custom parameters
        from modules.display_web import create_spectrogram_with_options

        spectrogram_base64, metadata = create_spectrogram_with_options(
            audio_path=request.file_path,
            clip_start=request.clip_start,
            clip_end=request.clip_end,
            color_mode=request.color_mode,
            freq_scale=request.freq_scale,
            n_mels=request.n_mels,
            n_fft=request.n_fft,
            hop_length=request.hop_length,
            window_size=request.window_size,
            fmin=request.fmin,
            fmax=request.fmax,
            bandpass_min=request.bandpass_min,
            bandpass_max=request.bandpass_max
        )

        # Extract just the base64 data (without the data:image/png;base64, prefix)
        if spectrogram_base64.startswith("data:image/png;base64,"):
            image_data_only = spectrogram_base64.split(",", 1)[1]
        else:
            image_data_only = spectrogram_base64

        # Convert base64 back to bytes for caching
        image_bytes = base64.b64decode(image_data_only)
        buffer = io.BytesIO(image_bytes)

        # Save to cache
        try:
            with open(cache_file_path, 'wb') as f:
                f.write(buffer.getvalue())
            print(f"Spectrogram cache SAVE: {cache_key} ({cache_file_path.stat().st_size / 1024:.2f} KB)")

            # Cleanup old cache if needed
            cleanup_old_spectrogram_files()
        except Exception as e:
            print(f"Spectrogram cache save error (non-fatal): {e}")

        # Return spectrogram with metadata for frontend scales
        return {
            "spectrogram": spectrogram_base64,
            "metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Evaluation endpoints
@app.post("/api/evaluation/load-dataset")
async def load_evaluation_dataset(dataset_path: str):
    """Load evaluation dataset with embeddings and labels"""
    try:
        print(f"DEBUG: Starting evaluation dataset load for path: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        print(f"DEBUG: Checking if dataset path exists: {dataset_path}")
        if not dataset_path.exists():
            print(f"DEBUG: Dataset path not found: {dataset_path}")
            raise HTTPException(status_code=404, detail="Dataset path not found")
        
        print(f"DEBUG: Dataset path exists, loading metadata...")
        # Load metadata if available
        metadata_path = dataset_path / "metadata.json"
        metadata = None
        if metadata_path.exists():
            print(f"DEBUG: Loading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Verify this is an evaluation dataset
            dataset_type = metadata.get("dataset_info", {}).get("dataset_type", "")
            print(f"DEBUG: Dataset type from metadata: '{dataset_type}'")
            if dataset_type != "evaluation":
                print(f"DEBUG: Invalid dataset type: {dataset_type}")
                raise HTTPException(status_code=400, detail="Dataset is not an evaluation dataset")
            
            # Store class map for evaluation
            if "class_map" in metadata:
                eval_state["eval_class_map"] = metadata["class_map"]
                print(f"DEBUG: Stored class map: {metadata['class_map']}")
        else:
            print(f"DEBUG: No metadata file found at {metadata_path}")
        
        # Load embeddings file
        embeddings_path = dataset_path / "embeddings.pkl"
        print(f"DEBUG: Checking for embeddings file at {embeddings_path}")
        if not embeddings_path.exists():
            print(f"DEBUG: Embeddings file not found: {embeddings_path}")
            raise HTTPException(status_code=404, detail="Embeddings file not found")
        
        print(f"DEBUG: Loading embeddings from {embeddings_path}")
        with open(embeddings_path, "rb") as f:
            embeddings_data = pickle.load(f)
        
        print(f"DEBUG: Embeddings data type: {type(embeddings_data)}")
        print(f"DEBUG: Embeddings data keys: {list(embeddings_data.keys()) if isinstance(embeddings_data, dict) else 'not a dict'}")
        
        # Check if it's an evaluation dataset (has labels)
        if isinstance(embeddings_data, dict) and 'embeddings' in embeddings_data and 'labels' in embeddings_data:
            embeddings = embeddings_data['embeddings']
            labels = embeddings_data['labels']
            print(f"DEBUG: Found embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")
            print(f"DEBUG: Found labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
            
            # Check for file information in embeddings data
            file_info = None
            if 'file_names' in embeddings_data:
                file_info = embeddings_data['file_names']
                print(f"DEBUG: Found file_names in embeddings data: {len(file_info)} files")
            elif 'filenames' in embeddings_data:
                file_info = embeddings_data['filenames'] 
                print(f"DEBUG: Found filenames in embeddings data: {len(file_info)} files")
            elif 'files' in embeddings_data:
                file_info = embeddings_data['files']
                print(f"DEBUG: Found files in embeddings data: {len(file_info)} files")
            else:
                print(f"DEBUG: No file information found in embeddings data")
                
        else:
            print(f"DEBUG: Invalid embeddings data structure")
            raise HTTPException(status_code=400, detail="Dataset is not an evaluation dataset (no labels found)")
        
        # Load audio database to get class structure (optional for evaluation)
        db_path = dataset_path / "audio_database.parquet"
        files_path = dataset_path / "files.parquet"
        clips_path = dataset_path / "clips.parquet"
        
        print(f"DEBUG: Checking for audio database at {db_path}")
        print(f"DEBUG: Checking for files.parquet at {files_path}")  
        print(f"DEBUG: Checking for clips.parquet at {clips_path}")
        
        files_df = None
        clips_df = None
        
        # Try to load separate parquet files first
        if files_path.exists() and clips_path.exists():
            print(f"DEBUG: Loading separate parquet files")
            import polars as pl
            try:
                files_df = pl.read_parquet(files_path)
                clips_df = pl.read_parquet(clips_path)
                print(f"DEBUG: Loaded files.parquet with {len(files_df)} files")
                print(f"DEBUG: Loaded clips.parquet with {len(clips_df)} clips")
                print(f"DEBUG: Files columns: {files_df.columns}")
                print(f"DEBUG: Clips columns: {clips_df.columns}")
            except Exception as e:
                print(f"DEBUG: Error loading parquet files: {e}")
                files_df = None
                clips_df = None
        
        if db_path.exists():
            print(f"DEBUG: Audio database file found, attempting to load...")
            # Load minimal database info to get class structure
            try:
                audio_db = db.Audio_DB()
                audio_db.load_db(str(db_path))
                print(f"DEBUG: Successfully loaded audio database from {db_path}")
            except Exception as db_error:
                print(f"DEBUG: Failed to load audio database: {db_error}")
                # For evaluation, we might not need the full database structure
                # We can continue without it if we have the class_map from metadata
                if not metadata or "class_map" not in metadata:
                    raise HTTPException(status_code=500, detail=f"Failed to load audio database and no class_map in metadata: {str(db_error)}")
                print("DEBUG: Continuing without audio database since we have class_map in metadata")
                audio_db = None
        else:
            print(f"DEBUG: Audio database not found at {db_path}")
            # For evaluation datasets, the audio database is optional if we have class_map
            if not metadata or "class_map" not in metadata:
                if files_df is None or clips_df is None:
                    raise HTTPException(status_code=404, detail="No database files found and no class_map in metadata")
            print("DEBUG: Continuing without audio database since we have class_map or parquet files")
            audio_db = None
        
        eval_state["eval_embeddings"] = embeddings
        eval_state["eval_labels"] = labels
        eval_state["eval_dataset_path"] = str(dataset_path)
        eval_state["eval_file_names"] = file_info  # Store file names if available
        eval_state["eval_files_df"] = files_df  # Store files dataframe if available
        eval_state["eval_clips_df"] = clips_df  # Store clips dataframe if available
        
        message = f"Evaluation dataset loaded with {len(embeddings)} samples and {labels.shape[1]} classes"
        if metadata:
            creation_date = metadata.get("dataset_info", {}).get("creation_date", "unknown")
            backend_model = metadata.get("dataset_info", {}).get("backend_model", "unknown")
            message += f" (created {creation_date[:10]}, {backend_model} model)"
        
        return {
            "status": "success",
            "message": message,
            "samples_count": len(embeddings),
            "classes_count": labels.shape[1],
            "metadata": metadata
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error in load_evaluation_dataset: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/load-classifier")
async def load_evaluation_classifier(classifier_path: str):
    """Load classifier model for evaluation"""
    try:
        classifier_path = Path(classifier_path)
        if not classifier_path.exists():
            raise HTTPException(status_code=404, detail="Classifier file not found")
        
        # Load the classifier model
        classifier_model = tf.keras.models.load_model(str(classifier_path))
        eval_state["eval_classifier"] = classifier_model
        
        return {
            "status": "success",
            "message": f"Classifier loaded successfully",
            "model_name": classifier_path.name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clean_nan_values(obj):
    """Recursively clean NaN values from data structures for JSON serialization"""
    import math
    
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, (np.floating, float)) and math.isnan(obj):
        return None  # Convert NaN to null in JSON
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays and handle NaN values
        cleaned_array = []
        for item in obj.flatten():
            if isinstance(item, (np.floating, float)) and math.isnan(item):
                cleaned_array.append(None)
            else:
                cleaned_array.append(float(item) if isinstance(item, np.number) else item)
        return cleaned_array
    elif isinstance(obj, np.number):
        return float(obj)
    else:
        return obj

@app.post("/api/evaluation/run-evaluation")
async def run_evaluation(threshold: float = 0):
    """Run evaluation and calculate performance metrics with optional threshold for class filtering"""
    try:
        print(f"DEBUG: Running evaluation with threshold={threshold}")
        
        if eval_state["eval_embeddings"] is None or eval_state["eval_labels"] is None:
            raise HTTPException(status_code=400, detail="Evaluation dataset not loaded")
        
        if eval_state["eval_classifier"] is None:
            raise HTTPException(status_code=400, detail="Classifier not loaded")
        
        embeddings = eval_state["eval_embeddings"]
        true_labels = eval_state["eval_labels"]
        classifier = eval_state["eval_classifier"]
        
        print(f"DEBUG: Embeddings type: {type(embeddings)}, length: {len(embeddings)}")
        print(f"DEBUG: True labels shape: {true_labels.shape}")
        
        # Get predictions by processing each embedding separately
        print(f"DEBUG: Raw embeddings structure: {[np.array(emb).shape if hasattr(emb, 'shape') or isinstance(emb, list) else type(emb) for emb in embeddings[:3]]}")
        
        # Process embeddings as list of arrays, handle multi-clip files
        all_predictions = []
        
        for i, emb in enumerate(embeddings):
            emb_array = np.array(emb)
            print(f"DEBUG: Processing embedding {i}: shape {emb_array.shape}")
            
            if emb_array.ndim == 1:
                # Single clip - reshape to (1, embedding_dim)
                emb_input = emb_array.reshape(1, -1)
                logits = classifier(emb_input)
                clip_predictions = tf.sigmoid(logits).numpy()
                print(f"DEBUG: Single clip logits: {logits.numpy()}")
                print(f"DEBUG: Single clip predictions: {clip_predictions}")
                
            elif emb_array.ndim == 2:
                # Multiple clips from same file - shape (n_clips, embedding_dim)
                # Process all clips and take max prediction per class
                logits = classifier(emb_array)  # This should work since emb_array is already (n_clips, 1280)
                clip_predictions = tf.sigmoid(logits).numpy()
                print(f"DEBUG: Multi-clip logits: {logits.numpy()}")
                print(f"DEBUG: Multi-clip predictions before max: {clip_predictions}")
                # Take max prediction across clips for each class
                clip_predictions = np.max(clip_predictions, axis=0, keepdims=True)
                print(f"DEBUG: Multi-clip predictions after max: {clip_predictions}")
                
            else:
                raise ValueError(f"Unexpected embedding shape: {emb_array.shape}")
            
            all_predictions.append(clip_predictions)
            print(f"DEBUG: Embedding {i} final prediction shape: {clip_predictions.shape}")
            print(f"DEBUG: Embedding {i} final prediction values: {clip_predictions}")
        
        # Combine all predictions
        predictions = np.vstack(all_predictions)
        print(f"DEBUG: Final predictions shape: {predictions.shape}")
        print(f"DEBUG: Final predictions values: {predictions}")
        print(f"DEBUG: Final predictions range: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}")
        
        # Store detailed predictions for display
        detailed_predictions = []
        
        # Check if we need file-level evaluation
        # For now, let's see what data we have available
        dataset_path = eval_state.get("eval_dataset_path")
        file_level_evaluation = False
        metadata = None
        
        if dataset_path:
            print(f"DEBUG: Looking for file grouping information in {dataset_path}")
            # Try to load additional metadata that might contain file information
            metadata_path = Path(dataset_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                print(f"DEBUG: Metadata keys: {list(metadata.keys())}")
                
                # Check if metadata contains file information
                if 'clips' in metadata:
                    print(f"DEBUG: Found {len(metadata['clips'])} clips in metadata")
                    # Check if clips have file information
                    first_clip = metadata['clips'][0] if metadata['clips'] else {}
                    print(f"DEBUG: First clip keys: {list(first_clip.keys())}")
                    
                    # Look for file grouping indicators
                    if ('original_file' in first_clip or 'file_path' in first_clip or 
                        'filename' in first_clip):
                        # Check if we actually have multiple clips per file
                        file_groups = {}
                        for clip_info in metadata['clips']:
                            file_key = (clip_info.get('original_file') or 
                                       clip_info.get('file_path') or 
                                       clip_info.get('filename') or
                                       clip_info.get('file_name') or 
                                       'unknown')
                            
                            # Extract just the filename from full paths
                            if '/' in file_key or '\\' in file_key:
                                file_key = Path(file_key).name
                                
                            if file_key not in file_groups:
                                file_groups[file_key] = 0
                            file_groups[file_key] += 1
                        
                        # If any file has multiple clips, enable file-level evaluation
                        max_clips_per_file = max(file_groups.values()) if file_groups else 1
                        if max_clips_per_file > 1:
                            file_level_evaluation = True
                            print(f"DEBUG: File-level evaluation enabled - max clips per file: {max_clips_per_file}")
                        else:
                            print(f"DEBUG: All files have single clips, using clip-level evaluation")
                elif 'file_paths' in metadata:
                    print(f"DEBUG: Found file_paths in metadata: {len(metadata['file_paths'])} files")
                    print(f"DEBUG: File paths content: {metadata['file_paths']}")
                    
                    # Check if we have parquet files that suggest file-level evaluation
                    clips_df = eval_state.get("eval_clips_df")
                    files_df = eval_state.get("eval_files_df")
                    
                    if clips_df is not None and files_df is not None:
                        num_clips = len(clips_df)
                        num_files = len(files_df)
                        print(f"DEBUG: Database shows {num_files} files with {num_clips} clips")
                        
                        if num_clips > num_files:
                            file_level_evaluation = True
                            print(f"DEBUG: File-level evaluation enabled - more clips ({num_clips}) than files ({num_files})")
                        else:
                            print(f"DEBUG: Clip-level evaluation - clips match files")
                    else:
                        print(f"DEBUG: Using file_paths for clip-level evaluation")
                else:
                    print(f"DEBUG: No 'clips' or 'file_paths' section found in metadata")
            else:
                print(f"DEBUG: No metadata file found at {metadata_path}")
        
        print(f"DEBUG: File-level evaluation: {file_level_evaluation}")
        
        # Perform file-level aggregation if needed
        if file_level_evaluation:
            print(f"DEBUG: Performing file-level aggregation...")
            
            # Group clips by original file
            file_groups = {}
            clip_to_file = {}
            
            # Try to use clips metadata first
            if metadata and 'clips' in metadata:
                print(f"DEBUG: Using clips metadata for file grouping")
                for i, clip_info in enumerate(metadata['clips']):
                    # Try multiple possible file name fields
                    original_file = (clip_info.get('original_file') or 
                                   clip_info.get('file_path') or 
                                   clip_info.get('filename') or 
                                   clip_info.get('file_name') or
                                   f'unknown_file_{i}')
                    
                    # Extract just the filename from full paths
                    if '/' in original_file or '\\' in original_file:
                        original_file = Path(original_file).name
                    
                    if original_file not in file_groups:
                        file_groups[original_file] = []
                    file_groups[original_file].append(i)
                    clip_to_file[i] = original_file
            else:
                # Use parquet files for file grouping
                clips_df = eval_state.get("eval_clips_df")
                files_df = eval_state.get("eval_files_df")
                
                if clips_df is not None and files_df is not None:
                    print(f"DEBUG: Using parquet files for file grouping")
                    
                    # Get file names directly from files.parquet in order
                    try:
                        file_names_list = files_df["file_name"].to_list()
                        print(f"DEBUG: File names from files.parquet: {file_names_list}")
                        
                        # Create file groups - each embedding corresponds to one file
                        for i, file_name in enumerate(file_names_list):
                            if i >= len(predictions):
                                break
                                
                            # Extract just filename from path if needed
                            if file_name and ('/' in str(file_name) or '\\' in str(file_name)):
                                clean_file_name = Path(str(file_name)).name
                            else:
                                clean_file_name = str(file_name) if file_name else f"file_{i}"
                            
                            # Each file corresponds to one prediction in file-level evaluation
                            file_groups[clean_file_name] = [i]
                            clip_to_file[i] = clean_file_name
                            
                            print(f"DEBUG: File {i}: {clean_file_name} -> prediction index {i}")
                        
                        print(f"DEBUG: Final file groups: {file_groups}")
                        
                    except Exception as e:
                        print(f"DEBUG: Error grouping with parquet files: {e}")
                        # Fallback: create one group per prediction
                        for i in range(len(predictions)):
                            file_name = f"file_{i}"
                            file_groups[file_name] = [i]
                            clip_to_file[i] = file_name
                else:
                    print(f"DEBUG: No grouping information available, treating each sample as separate file")
                    for i in range(len(predictions)):
                        file_name = f"file_{i}"
                        file_groups[file_name] = [i]
                        clip_to_file[i] = file_name
            
            print(f"DEBUG: Found {len(file_groups)} unique files:")
            for file_name, clip_indices in file_groups.items():
                print(f"DEBUG: - {file_name}: {len(clip_indices)} clips (indices: {clip_indices})")
            
            # Aggregate predictions and labels at file level
            file_names = list(file_groups.keys())
            num_files = len(file_names)
            
            # Create file-level predictions (max across clips for each class)
            file_predictions = np.zeros((num_files, predictions.shape[1]))
            file_labels = np.zeros((num_files, true_labels.shape[1]))
            
            for file_idx, (file_name, clip_indices) in enumerate(file_groups.items()):
                # Get max prediction across all clips of this file for each class
                file_clip_predictions = predictions[clip_indices]
                file_predictions[file_idx] = np.max(file_clip_predictions, axis=0)
                
                # For labels, take the max (if any clip is positive, file is positive)
                file_clip_labels = true_labels[clip_indices]
                file_labels[file_idx] = np.max(file_clip_labels, axis=0)
                
                # Store detailed prediction info for display
                detailed_predictions.append({
                    "file_name": file_name,
                    "predictions": file_predictions[file_idx].tolist(),
                    "labels": file_labels[file_idx].tolist(),
                    "num_clips": len(clip_indices)
                })
                
                print(f"DEBUG: File '{file_name}': max predictions {file_predictions[file_idx]}, labels {file_labels[file_idx]}")
            
            # Use file-level data for evaluation
            predictions = file_predictions
            true_labels = file_labels
            
            print(f"DEBUG: Aggregated to {num_files} files for evaluation")
            print(f"DEBUG: New predictions shape: {predictions.shape}")
            print(f"DEBUG: New true labels shape: {true_labels.shape}")
        else:
            # Clip-level evaluation - create detailed predictions from metadata if available
            if metadata:
                if 'clips' in metadata:
                    print(f"DEBUG: Using metadata clips for clip-level evaluation")
                    print(f"DEBUG: Found {len(metadata['clips'])} clips in metadata, {len(predictions)} predictions")
                    
                    if len(metadata['clips']) == len(predictions):
                        for i, clip_info in enumerate(metadata['clips']):
                            # Try different possible file name fields
                            clip_name = (clip_info.get('filename') or 
                                       clip_info.get('original_file') or 
                                       clip_info.get('file_path') or 
                                       clip_info.get('file_name') or
                                       f'clip_{i}')
                            
                            # If we have a path, extract just the filename
                            if '/' in clip_name or '\\' in clip_name:
                                clip_name = Path(clip_name).name
                            
                            print(f"DEBUG: Clip {i}: using name '{clip_name}' from {list(clip_info.keys())}")
                            detailed_predictions.append({
                                "file_name": clip_name,
                                "predictions": predictions[i].tolist(),
                                "labels": true_labels[i].tolist(),
                                "num_clips": 1
                            })
                    else:
                        print(f"DEBUG: Metadata clips count mismatch: {len(metadata['clips'])} != {len(predictions)}")
                elif 'file_paths' in metadata:
                    print(f"DEBUG: Using file_paths from metadata for clip-level evaluation")
                    print(f"DEBUG: Found {len(metadata['file_paths'])} file paths, {len(predictions)} predictions")
                    
                    if len(metadata['file_paths']) == len(predictions):
                        for i, file_path in enumerate(metadata['file_paths']):
                            # Extract just the filename from the path
                            filename = Path(file_path).name
                            print(f"DEBUG: Sample {i}: using filename '{filename}' from path '{file_path}'")
                            detailed_predictions.append({
                                "file_name": filename,
                                "predictions": predictions[i].tolist(),
                                "labels": true_labels[i].tolist(),
                                "num_clips": 1
                            })
                    else:
                        print(f"DEBUG: File paths count mismatch: {len(metadata['file_paths'])} != {len(predictions)}")
                else:
                    print(f"DEBUG: No clips or file_paths found in metadata")
            
            # Try using file names from clips dataframe as fallback
            if not detailed_predictions:
                clips_df = eval_state.get("eval_clips_df")
                files_df = eval_state.get("eval_files_df")
                
                if clips_df is not None and files_df is not None and len(clips_df) == len(predictions):
                    print(f"DEBUG: Using clips and files dataframes for file names")
                    print(f"DEBUG: Clips dataframe has {len(clips_df)} rows, predictions has {len(predictions)} samples")
                    
                    # Join clips with files to get file names
                    try:
                        clips_with_files = clips_df.join(files_df, on="file_id", how="left")
                        print(f"DEBUG: Joined dataframe columns: {clips_with_files.columns}")
                        
                        # Extract file names (try different possible column names)
                        filename_column = None
                        for col in ["file_name", "filename", "name", "path", "file_path"]:
                            if col in clips_with_files.columns:
                                filename_column = col
                                break
                        
                        if filename_column:
                            print(f"DEBUG: Using column '{filename_column}' for file names")
                            file_names = clips_with_files[filename_column].to_list()
                            
                            for i, filename in enumerate(file_names):
                                # Extract just filename from path if needed
                                if filename and ('/' in str(filename) or '\\' in str(filename)):
                                    filename = Path(str(filename)).name
                                elif not filename:
                                    filename = f"unknown_file_{i}"
                                    
                                detailed_predictions.append({
                                    "file_name": str(filename),
                                    "predictions": predictions[i].tolist(),
                                    "labels": true_labels[i].tolist(),
                                    "num_clips": 1
                                })
                                print(f"DEBUG: Added detailed prediction {i}: {str(filename)} -> {predictions[i].tolist()}")
                        else:
                            print(f"DEBUG: No suitable filename column found in joined dataframe")
                            
                    except Exception as e:
                        print(f"DEBUG: Error joining dataframes: {e}")
                        
                # Try using file names from embeddings data as another fallback
                if not detailed_predictions:
                    file_names_from_embeddings = eval_state.get("eval_file_names")
                    if file_names_from_embeddings and len(file_names_from_embeddings) == len(predictions):
                        print(f"DEBUG: Using file names from embeddings data")
                        for i, filename in enumerate(file_names_from_embeddings):
                            # Extract just filename from path if needed
                            if '/' in filename or '\\' in filename:
                                filename = Path(filename).name
                            detailed_predictions.append({
                                "file_name": filename,
                                "predictions": predictions[i].tolist(),
                                "labels": true_labels[i].tolist(),
                                "num_clips": 1
                            })
                    else:
                        print(f"DEBUG: Using fallback naming for {len(predictions)} predictions")
                        if file_names_from_embeddings:
                            print(f"DEBUG: File names count mismatch: {len(file_names_from_embeddings)} != {len(predictions)}")
                        for i in range(len(predictions)):
                            detailed_predictions.append({
                                "file_name": f"sample_{i}",
                                "predictions": predictions[i].tolist(),
                                "labels": true_labels[i].tolist(),
                                "num_clips": 1
                            })
        
        # Determine if single class or multiclass
        num_classes = true_labels.shape[1]
        num_prediction_classes = predictions.shape[1]
        is_single_class = num_classes == 1
        
        print(f"DEBUG: Dataset has {num_classes} classes, model predicts {num_prediction_classes} classes")
        
        # Check for class count mismatch
        if num_classes != num_prediction_classes:
            error_msg = f"Class count mismatch: Dataset has {num_classes} classes but model predicts {num_prediction_classes} classes"
            print(f"DEBUG: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        results = {
            "is_single_class": is_single_class,
            "num_classes": num_classes,
            "evaluation_level": "file" if file_level_evaluation else "clip",
            "num_samples": len(predictions)
        }
        
        if is_single_class:
            # Single class evaluation
            true_labels_1d = true_labels[:, 0]
            predictions_1d = predictions[:, 0]
            
            # Calculate AUC
            auc_result = tc.get_AUC(true_labels, predictions, threshold)
            results["auc"] = auc_result["individual"][0] if len(auc_result["individual"]) > 0 else None
            
            # Calculate Average Precision
            ap_result = tc.cmap(true_labels, predictions, threshold)
            results["average_precision"] = ap_result["individual"][0] if len(ap_result["individual"]) > 0 else None
            
            # Calculate confusion matrix (binary)
            predicted_binary = (predictions_1d > 0.5).astype(int)
            cm = confusion_matrix(true_labels_1d, predicted_binary)
            results["confusion_matrix"] = cm.tolist()
            
        else:
            # Multiclass evaluation
            # Calculate macro AUC
            auc_result = tc.get_AUC(true_labels, predictions, threshold)
            results["macro_auc"] = auc_result["macro"]
            results["class_aucs"] = auc_result["individual"]
            results["auc_mask"] = auc_result["mask"].tolist() if "mask" in auc_result else None
            
            # Calculate mean Average Precision
            ap_result = tc.cmap(true_labels, predictions, threshold)
            results["mean_ap"] = ap_result["macro"]
            results["class_aps"] = ap_result["individual"]
            results["ap_mask"] = ap_result["mask"].tolist() if "mask" in ap_result else None
            
            # Generate class names from metadata if available
            if eval_state["eval_class_map"]:
                # Sort class names by their numeric values
                class_items = sorted(eval_state["eval_class_map"].items(), key=lambda x: x[1])
                results["class_names"] = [name for name, _ in class_items]
            else:
                results["class_names"] = [f"Class_{i}" for i in range(num_classes)]
            
            # Calculate multiclass confusion matrix
            print(f"DEBUG CM: predictions shape: {predictions.shape}")
            print(f"DEBUG CM: true_labels shape: {true_labels.shape}")
            print(f"DEBUG CM: predictions sample: {predictions[:2]}")
            print(f"DEBUG CM: true_labels sample: {true_labels[:2]}")
            
            # Check if this is multi-label (binary) or multi-class (mutually exclusive)
            # Multi-label: each sample can have multiple classes (0s and 1s)
            # Multi-class: each sample has exactly one class (one-hot or single index)
            
            is_multilabel = np.any((true_labels > 0) & (true_labels < 1)) == False and np.any(np.sum(true_labels, axis=1) > 1)
            print(f"DEBUG CM: is_multilabel: {is_multilabel}")
            
            if is_multilabel:
                # For multi-label, create separate binary confusion matrices for each class
                # This is more appropriate than a single multiclass matrix
                print(f"DEBUG CM: Creating multi-label confusion matrices")
                cms = []
                for class_idx in range(num_classes):
                    # Binary predictions for this class (threshold at 0.5)
                    pred_binary = (predictions[:, class_idx] > 0.5).astype(int)
                    true_binary = true_labels[:, class_idx].astype(int)
                    cm_binary = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
                    cms.append(cm_binary.tolist())
                results["confusion_matrix"] = cms
                results["confusion_matrix_type"] = "multilabel"
            else:
                # For true multi-class, use argmax approach
                print(f"DEBUG CM: Creating multi-class confusion matrix")
                predicted_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(true_labels, axis=1)
                cm = confusion_matrix(true_classes, predicted_classes, labels=range(num_classes))
                results["confusion_matrix"] = cm.tolist()
                results["confusion_matrix_type"] = "multiclass"
        
        # Add detailed predictions to results
        results["detailed_predictions"] = detailed_predictions
        
        # Clean NaN values before returning
        cleaned_results = clean_nan_values(results)
        
        return {
            "status": "success", 
            "message": "Evaluation completed successfully",
            "results": cleaned_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/export-metrics-csv")
async def export_metrics_csv(export_path: str, filename: str):
    """Export evaluation metrics as CSV with metrics summary"""
    try:
        # Check if we have evaluation results
        if (eval_state["eval_embeddings"] is None or eval_state["eval_labels"] is None or 
            eval_state["eval_classifier"] is None):
            raise HTTPException(status_code=400, detail="No evaluation results available")
        
        # Get the current evaluation results by calling run_evaluation with default threshold
        # This ensures we have the latest results
        response = await run_evaluation(threshold=0)
        results = response["results"]
        
        import csv
        import os
        
        # Ensure export path exists
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create full file path
        csv_filename = f"{filename}.csv"
        full_path = export_path / csv_filename
        
        # Write CSV file
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Metric', 'Macro']
            if 'class_names' in results:
                header.extend(results['class_names'])
            else:
                # Fallback class names
                num_classes = len(results.get('class_aucs', []))
                header.extend([f'Class_{i}' for i in range(num_classes)])
            writer.writerow(header)
            
            # Write AUC row
            if results['is_single_class']:
                auc_row = ['AUC', results.get('auc', 'N/A'), results.get('auc', 'N/A')]
            else:
                auc_row = ['AUC', results.get('macro_auc', 'N/A')]
                auc_row.extend([str(auc) if auc is not None else 'N/A' for auc in results.get('class_aucs', [])])
            writer.writerow(auc_row)
            
            # Write AP row
            if results['is_single_class']:
                ap_row = ['AP', results.get('average_precision', 'N/A'), results.get('average_precision', 'N/A')]
            else:
                ap_row = ['AP', results.get('mean_ap', 'N/A')]
                ap_row.extend([str(ap) if ap is not None else 'N/A' for ap in results.get('class_aps', [])])
            writer.writerow(ap_row)
        
        return {
            "status": "success",
            "message": f"Metrics CSV exported successfully to {full_path}",
            "file_path": str(full_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/export-predictions-csv")
async def export_predictions_csv(export_path: str, filename: str):
    """Export detailed predictions as CSV with file/clip names and class scores"""
    try:
        # Check if we have evaluation results
        if (eval_state["eval_embeddings"] is None or eval_state["eval_labels"] is None or 
            eval_state["eval_classifier"] is None):
            raise HTTPException(status_code=400, detail="No evaluation results available")
        
        # Get the current evaluation results
        response = await run_evaluation(threshold=0)
        results = response["results"]
        
        import csv
        import os
        
        # Ensure export path exists
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create full file path
        csv_filename = f"{filename}.csv"
        full_path = export_path / csv_filename
        
        # Write CSV file
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            sample_type = "File" if results.get('evaluation_level') == 'file' else "Clip"
            header = [f'{sample_type}_Name']
            if 'class_names' in results:
                header.extend(results['class_names'])
            else:
                # Fallback class names
                if 'detailed_predictions' in results and len(results['detailed_predictions']) > 0:
                    num_classes = len(results['detailed_predictions'][0]['predictions'])
                    header.extend([f'Class_{i}' for i in range(num_classes)])
            writer.writerow(header)
            
            # Write prediction rows
            if 'detailed_predictions' in results:
                for item in results['detailed_predictions']:
                    row = [item['file_name']]
                    row.extend([f"{pred:.6f}" for pred in item['predictions']])
                    writer.writerow(row)
        
        return {
            "status": "success",
            "message": f"Predictions CSV exported successfully to {full_path}",
            "file_path": str(full_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def training_thread(config: ModelTrainingConfig):
    """Thread function to train model in background"""
    try:
        training_state["status"] = "training"
        training_state["message"] = "Loading metadata and audio files..."
        training_state["logs"] = []
        training_state["stop_requested"] = False
        
        # Load metadata
        metadata_path = Path(config.metadata_path)
        if not metadata_path.exists():
            training_state["status"] = "error"
            training_state["message"] = "Metadata file not found"
            return
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        class_map = metadata.get("class_map", {})
        backend_model = metadata.get("dataset_info", {}).get("backend_model", "PERCH")
        
        # Set backend configuration
        os.environ["BACKEND"] = backend_model
        
        # Find training audio files
        audio_folder = Path(config.training_audio_folder)
        if not audio_folder.exists():
            training_state["status"] = "error"
            training_state["message"] = "Training audio folder not found"
            return
        
        files = list(audio_folder.glob("**/*.wav"))
        files.extend(list(audio_folder.glob("**/*.mp3")))
        files.extend(list(audio_folder.glob("**/*.WAV")))
        files.extend(list(audio_folder.glob("**/*.MP3")))
        files = [str(f) for f in files]
        
        if not files:
            training_state["status"] = "error"
            training_state["message"] = "No audio files found in training folder"
            return
        
        training_state["message"] = f"Processing {len(files)} audio files..."
        
        # Check if this is an exported dataset with enhanced metadata
        try:
            file_paths, labels, label_strengths, export_metadata = u.load_training_data_with_strength(
                str(audio_folder), class_map
            )
            print(f"Using enhanced label loading with strength information")
            use_label_strength = True
            
            # Generate embeddings for the labeled files
            embeddings = u.load_and_preprocess(file_paths)
            
        except (ValueError, KeyError) as e:
            # Fallback to legacy loading if enhanced loading fails
            print(f"Enhanced loading failed: {e}")
            print("Falling back to legacy label extraction from filenames")
            use_label_strength = False
            
            # Generate embeddings for training files
            embeddings = u.load_and_preprocess(files)
            
            # Extract labels from filenames (legacy method)
            labels = []
            for file_path in files:
                file_label = u.get_label(class_map, file_path)
                labels.append(file_label)
            labels = np.array(labels)
            label_strengths = np.ones_like(labels)  # Assume all strong labels
        
        if training_state["stop_requested"]:
            training_state["status"] = "error"
            training_state["message"] = "Training stopped by user"
            return
        
        # Prepare test data based on mode
        embeddings_array = np.array(embeddings).squeeze()
        
        if config.test_data_mode == "split":
            # Create train/test split (including label strengths)
            if use_label_strength:
                X_train, X_test, y_train, y_test, strength_train, strength_test = train_test_split(
                    embeddings_array, labels, label_strengths,
                    test_size=config.test_split, 
                    random_state=config.random_state,
                    stratify=labels if labels.shape[1] == 1 else None
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    embeddings_array, labels, 
                    test_size=config.test_split, 
                    random_state=config.random_state,
                    stratify=labels if labels.shape[1] == 1 else None
                )
                # Create default strong labels for strength arrays
                strength_train = np.ones_like(y_train)
                strength_test = np.ones_like(y_test)
        else:  # test_data_mode == "folder"
            # Use all training data for training
            X_train = embeddings_array
            y_train = labels
            if use_label_strength:
                strength_train = label_strengths
            else:
                strength_train = np.ones_like(y_train)
            
            # Load test data from separate folder
            training_state["message"] = "Loading test data from separate folder..."
            
            test_folder = Path(config.test_audio_folder)
            if not test_folder.exists():
                training_state["status"] = "error"
                training_state["message"] = "Test audio folder not found"
                return
            
            test_files = list(test_folder.glob("**/*.wav"))
            test_files.extend(list(test_folder.glob("**/*.mp3")))
            test_files.extend(list(test_folder.glob("**/*.WAV")))
            test_files.extend(list(test_folder.glob("**/*.MP3")))
            test_files = [str(f) for f in test_files]
            
            if not test_files:
                training_state["status"] = "error"
                training_state["message"] = "No audio files found in test folder"
                return
            
            # Try to load test data with enhanced metadata
            try:
                test_file_paths, test_labels, test_label_strengths, test_export_metadata = u.load_training_data_with_strength(
                    str(test_folder), class_map
                )
                print(f"Using enhanced test data loading with strength information")
                
                # Generate embeddings for the labeled test files
                test_embeddings = u.load_and_preprocess(test_file_paths)
                strength_test = test_label_strengths
                
            except (ValueError, KeyError) as e:
                # Fallback to legacy loading for test data
                print(f"Enhanced test data loading failed: {e}")
                print("Falling back to legacy test label extraction")
                
                # Generate embeddings for test files
                test_embeddings = u.load_and_preprocess(test_files)
                
                # Extract labels for test files (legacy method)
                test_labels = []
                for file_path in test_files:
                    file_label = u.get_label(class_map, file_path)
                    test_labels.append(file_label)
                test_labels = np.array(test_labels)
                strength_test = np.ones_like(test_labels)  # Assume all strong labels
            
            X_test = np.array(test_embeddings).squeeze()
            y_test = test_labels
        
        training_state["message"] = "Starting model training..."
        
        
        
        # Call fit_w_tape function with full save path
        full_save_path = config.model_save_path

        # Call fit_w_tape with label strength support and optional early stopping/lr reduction
        classifier_model, train_losses, val_losses, cmaps, aucs, geomeans = tc.fit_w_tape(
            X_train,
            y_train,
            X_test,
            y_test,
            config.training_params.n_steps,
            config.training_params.batch_size,
            config.training_params.learning_rate,
            config.training_params.model_type,
            full_save_path,
            config.training_params.verbose,
            label_strength=strength_train,
            eval_label_strength=strength_test,
            weak_neg_weight=config.training_params.weak_neg_weight,
            enable_early_stopping=config.training_params.enable_early_stopping,
            enable_lr_reduction=config.training_params.enable_lr_reduction,
            lr_redux=config.training_params.lr_redux,
            patience=config.training_params.patience,
            lr_reduce_patience=config.training_params.lr_reduce_patience,
            metric_for_tracking=config.training_params.metric_for_tracking
        )
        
        # Add final metrics to logs
        try:
            final_loss = train_losses[-1] if train_losses and len(train_losses) > 0 else None
            best_cmap = max(cmaps) if cmaps and len(cmaps) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in cmaps) else None
            best_auc = max(aucs) if aucs and len(aucs) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in aucs) else None
            best_geomean = max(geomeans) if geomeans and len(geomeans) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in geomeans) else None
            min_val_loss = min(val_losses) if val_losses and len(val_losses) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in val_losses) else None

            training_state["logs"].append(f"Final loss: {final_loss}")
            training_state["logs"].append(f"Macro cMAP of best fit: {best_cmap}")
            training_state["logs"].append(f"Macro AUC of best fit: {best_auc}")
            training_state["logs"].append(f"Geometric Mean of best fit: {best_geomean}")
            training_state["logs"].append(f"Minimum validation loss: {min_val_loss}")

            if config.training_params.metric_for_tracking == 'auc':
                training_state["logs"].append(f"Model tracking metric: Macro AUC")
            elif config.training_params.metric_for_tracking == 'geomean':
                training_state["logs"].append(f"Model tracking metric: Geometric Mean (AUC × cMAP)")
            elif config.training_params.metric_for_tracking == 'loss':
                training_state["logs"].append(f"Model tracking metric: Test Data Loss (minimize)")
            else:
                training_state["logs"].append(f"Model tracking metric: Macro cMAP")
            
            if config.training_params.verbose:
                training_state["logs"].append("Training completed with verbose output to console")
            else:
                training_state["logs"].append("Training completed")
                
        except (ValueError, TypeError, IndexError) as e:
            training_state["logs"].append(f"Warning: Error calculating final metrics: {e}")
        
        if training_state["stop_requested"]:
            training_state["status"] = "error"
            training_state["message"] = "Training stopped by user"
            return
        
        # Model is already saved to the correct location by fit_w_tape
        actual_model_path = full_save_path
        training_state["logs"].append(f"Model saved to: {actual_model_path}")
        
        # Training completed successfully
        training_state["status"] = "completed"
        training_state["message"] = "Model training completed successfully"
        
        # Safely calculate final statistics
        try:
            final_loss = float(train_losses[-1]) if train_losses and len(train_losses) > 0 else None
            best_cmap = float(max(cmaps)) if cmaps and len(cmaps) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in cmaps) else None
            total_steps = len(train_losses) if train_losses else 0
        except (ValueError, TypeError, IndexError) as e:
            training_state["logs"].append(f"Warning: Error calculating final statistics: {e}")
            final_loss = None
            best_cmap = None
            total_steps = 0
        
        training_state["results"] = {
            "final_loss": final_loss,
            "best_cmap": best_cmap,
            "total_steps": total_steps,
            "model_path": actual_model_path,
            "batch_size": config.training_params.batch_size,
            "learning_rate": config.training_params.learning_rate,
            "model_type": config.training_params.model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
    except Exception as e:
        training_state["status"] = "error"
        training_state["message"] = f"Training failed: {str(e)}"
        training_state["logs"].append(f"ERROR: {str(e)}")

# Model Training endpoints
@app.post("/api/model-training/start")
async def start_model_training(config: ModelTrainingConfig):
    """Start model training in background"""
    try:
        # Check if already training
        if training_state["status"] == "training":
            raise HTTPException(status_code=400, detail="Model training already in progress")
        
        # Reset training state
        training_state.update({
            "status": "training",
            "message": "Starting model training...",
            "logs": [],
            "results": None,
            "stop_requested": False
        })
        
        # Start training in background thread
        thread = threading.Thread(target=training_thread, args=(config,))
        thread.daemon = True
        thread.start()
        
        return {
            "status": "started",
            "message": "Model training started in background"
        }
        
    except Exception as e:
        training_state["status"] = "error"
        training_state["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "status": training_state["status"],
        "message": training_state["message"],
        "logs": training_state["logs"],
        "results": training_state["results"]
    }

@app.post("/api/model-training/stop")
async def stop_model_training():
    """Stop current model training"""
    if training_state["status"] == "training":
        training_state["stop_requested"] = True
        training_state["status"] = "stopping"
        return {"status": "success", "message": "Stop signal sent to training process"}
    else:
        return {"status": "error", "message": "No training in progress"}

@app.post("/api/model-training/preview-data")
async def preview_training_data(training_audio_folder: str, metadata_path: str):
    """Preview training data - files, labels, and label strengths"""
    try:
        # Load metadata
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise HTTPException(status_code=400, detail="Metadata file not found")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        class_map = metadata.get("class_map", {})
        backend_model = metadata.get("dataset_info", {}).get("backend_model", "PERCH")
        
        # Find training audio files
        audio_folder = Path(training_audio_folder)
        if not audio_folder.exists():
            raise HTTPException(status_code=400, detail="Training audio folder not found")
        
        files = list(audio_folder.glob("**/*.wav"))
        files.extend(list(audio_folder.glob("**/*.mp3")))
        files.extend(list(audio_folder.glob("**/*.WAV")))
        files.extend(list(audio_folder.glob("**/*.MP3")))
        files = [str(f) for f in files]
        
        if not files:
            raise HTTPException(status_code=400, detail="No audio files found in training folder")
        
        # Check if this is an exported dataset with enhanced metadata
        preview_data = []
        use_label_strength = False
        
        try:
            print(f"DEBUG: Attempting enhanced loading for {len(files)} files")
            file_paths, labels, label_strengths, export_metadata = u.load_training_data_with_strength(
                str(audio_folder), class_map
            )
            use_label_strength = True
            print(f"DEBUG: Enhanced loading successful - got {len(file_paths)} files, {len(labels)} labels, {len(label_strengths)} strengths")
            
            # Create preview data with enhanced metadata
            for i, file_path in enumerate(file_paths):
                file_name = os.path.basename(file_path)
                label_vector = labels[i] if len(labels) > i else None
                strength_vector = label_strengths[i] if len(label_strengths) > i else None
                
                print(f"DEBUG: File {i}: {file_name}")
                print(f"DEBUG: Label vector: {label_vector}")
                print(f"DEBUG: Strength vector: {strength_vector}")
                
                # Convert numpy arrays to lists and create readable class labels
                class_labels = {}
                class_strengths = {}
                
                if label_vector is not None and strength_vector is not None:
                    for class_name, class_idx in class_map.items():
                        if class_idx < len(label_vector):
                            label_value = int(label_vector[class_idx])
                            strength_value = float(strength_vector[class_idx])
                            
                            class_labels[class_name] = "Present" if label_value == 1 else "Not Present"
                            class_strengths[class_name] = "Strong" if strength_value >= 0.8 else "Weak"
                
                preview_data.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "class_labels": class_labels,
                    "class_strengths": class_strengths,
                    "raw_label_vector": label_vector.tolist() if label_vector is not None else None,
                    "raw_strength_vector": strength_vector.tolist() if strength_vector is not None else None
                })
                
        except (ValueError, KeyError) as e:
            # Fallback to legacy loading if enhanced loading fails
            print(f"DEBUG: Enhanced loading failed: {e}")
            print("DEBUG: Falling back to legacy label extraction from filenames")
            use_label_strength = False
            
            # Extract labels from filenames (legacy method)  
            for file_path in files:
                file_name = os.path.basename(file_path)
                file_label = u.get_label(class_map, file_path)
                
                print(f"DEBUG: Legacy - File: {file_name}")
                print(f"DEBUG: Legacy - Label from get_label: {file_label}")
                print(f"DEBUG: Legacy - Class map: {class_map}")
                
                # Convert label vector to readable class labels
                class_labels = {}
                class_strengths = {}
                
                if file_label is not None:
                    for class_name, class_idx in class_map.items():
                        if class_idx < len(file_label):
                            label_value = int(file_label[class_idx])
                            class_labels[class_name] = "Present" if label_value == 1 else "Not Present" 
                            class_strengths[class_name] = "Strong"  # Legacy assumes all strong
                
                preview_data.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "class_labels": class_labels,
                    "class_strengths": class_strengths,
                    "raw_label_vector": file_label.tolist() if file_label is not None else None,
                    "raw_strength_vector": None  # Not available in legacy mode
                })
        
        return {
            "status": "success",
            "data": {
                "total_files": len(preview_data),
                "class_map": class_map,
                "backend_model": backend_model,
                "use_label_strength": use_label_strength,
                "files": preview_data[:100]  # Limit to first 100 for performance
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Validation System Endpoints
@app.post("/api/validation/load-predictions")
async def load_validation_predictions(request: Request):
    """Load prediction data for validation workflow"""
    import asyncio

    body = await request.json()
    predictions_path = body.get("predictions_path")
    audio_directory = body.get("audio_directory")
    model_name = body.get("model_name")
    format_type = body.get("format_type", "auto")
    recursive = body.get("recursive", True)
    replace_existing = body.get("replace_existing", True)
    save_location = body.get("save_location")  # User's specified save location

    if not predictions_path or not model_name:
        raise HTTPException(status_code=400, detail="predictions_path and model_name are required")

    try:
        print(f"DEBUG: Loading predictions from: {predictions_path}")
        print(f"DEBUG: Model name: {model_name}")
        print(f"DEBUG: Audio directory: {audio_directory}")

        # Initialize validation database if not exists
        if app_state["validation_db"] is None:
            print("DEBUG: Initializing validation database")
            app_state["validation_db"] = vdb.ValidationDB()
        elif replace_existing:
            # Reuse existing instance, but wait for pending saves
            print("DEBUG: Reusing validation database instance, clearing data")
            app_state["validation_db"].wait_for_pending_save()

        validation_db = app_state["validation_db"]

        # Load predictions from CSV file(s) using standard format
        # Run blocking I/O operation in thread pool to avoid blocking event loop
        print(f"DEBUG: Using standard CSV loader with format: {format_type}")
        result = await asyncio.to_thread(
            validation_db.load_predictions_from_csv,
            file_path=predictions_path,
            format_type=format_type,
            model_name=model_name,
            audio_directory=audio_directory,
            replace_existing=False  # We already cleared at the DB level if needed
        )

        print(f"DEBUG: Load result status: {result.get('status')}")
        if result.get('status') == 'error':
            print(f"ERROR: {result.get('message')}")
            return result

        # Set project path for future auto-saves if save location is provided
        if save_location and save_location.strip():
            validation_db.project_base_path = save_location.strip()
            # Don't save yet - wait for strata creation
            print(f"DEBUG: Set project save location to {save_location}. Saving deferred until strata creation.")
        else:
            # Warn user that work won't be saved
            result['no_save_location'] = True
            print("WARNING: No save location specified - validations will NOT be saved!")

        return result

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"ERROR: Exception in load_validation_predictions: {error_msg}")
        print(f"ERROR: Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/validation/load-unvalidated-clips")
async def load_unvalidated_clips(request: Request):
    """Load unvalidated clips by subdividing audio files into fixed-length windows"""
    import asyncio

    body = await request.json()
    audio_directory = body.get("audio_directory")
    clip_window_length = body.get("clip_window_length", 3.0)
    target_classes = body.get("target_classes", [])
    strata_file = body.get("strata_file")  # Changed from strata_column to strata_file
    use_filename_as_strata = body.get("use_filename_as_strata", False)
    replace_existing = body.get("replace_existing", True)
    save_location = body.get("save_location")  # User's specified save location

    if not audio_directory:
        raise HTTPException(status_code=400, detail="audio_directory is required")

    if not target_classes:
        raise HTTPException(status_code=400, detail="target_classes is required")

    try:
        # Initialize validation database if not exists
        if app_state["validation_db"] is None:
            print("DEBUG: Initializing validation database")
            app_state["validation_db"] = vdb.ValidationDB()
        elif replace_existing:
            # Reuse existing instance, but wait for pending saves and clear data
            print("DEBUG: Reusing validation database instance, clearing data")
            app_state["validation_db"].wait_for_pending_save()

        validation_db = app_state["validation_db"]

        # Run blocking I/O operation in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(
            validation_db.load_unvalidated_clips,
            audio_directory=audio_directory,
            clip_window_length=clip_window_length,
            target_classes=target_classes,
            strata_file=strata_file,  # Changed from strata_column to strata_file
            use_filename_as_strata=use_filename_as_strata,
            replace_existing=False  # We already cleared at the DB level if needed
        )

        if result.get('status') == 'error':
            return result

        # Set project path for future auto-saves if save location is provided
        if save_location and save_location.strip():
            validation_db.project_base_path = save_location.strip()
            # Don't save yet - wait for strata creation
            print(f"DEBUG: Set project save location to {save_location}. Saving deferred until strata creation.")
        else:
            # Warn user that work won't be saved
            result['no_save_location'] = True
            print("WARNING: No save location specified - validations will NOT be saved!")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validation/create-strata")
async def create_validation_strata(request: Request):
    """Create validation strata using user-provided strata column"""
    import asyncio

    try:
        # Optional JSON body parsing (for future extensibility)
        try:
            body = await request.json()
        except:
            body = {}

        if app_state["validation_db"] is None:
            raise HTTPException(status_code=400, detail="No prediction data loaded")

        validation_db = app_state["validation_db"]

        # Get confidence threshold from request body if provided
        confidence_threshold = body.get('confidence_threshold', 0.0)

        # Run potentially blocking DataFrame operations in thread pool
        result = await asyncio.to_thread(validation_db.create_strata, confidence_threshold=confidence_threshold)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/strata")
async def get_validation_strata():
    """Get list of available validation strata"""
    import asyncio

    try:
        if app_state["validation_db"] is None:
            return {"strata": []}

        validation_db = app_state["validation_db"]

        # Run potentially blocking DataFrame operations in thread pool
        strata_summary = await asyncio.to_thread(validation_db.get_strata_summary)

        return {"strata": strata_summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/strata/{strata_id}/species")
async def get_strata_species(strata_id: str):
    """Get species available in a specific strata"""
    import asyncio

    try:
        if app_state["validation_db"] is None:
            return {"species": []}

        validation_db = app_state["validation_db"]

        # Run potentially blocking DataFrame operations in thread pool
        def get_species_data():
            species_data = validation_db.validation_progress_df.filter(
                pl.col("strata_id") == strata_id
            ).select([
                "species_name",
                "total_clips",
                "confirmed_clips"
            ])
            return species_data.to_dicts()

        species = await asyncio.to_thread(get_species_data)

        return {"species": species}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_overall_validation_stats():
    """Helper to calculate overall strata-level progress and total annotations"""
    if app_state["validation_db"] is None:
        return None

    db = app_state["validation_db"]
    progress_df = db.validation_progress_df

    if len(progress_df) == 0:
        return {
            "total_strata_species": 0,
            "completed_strata_species": 0,
            "total_validated": len(db.validation_annotations_df)
        }

    # A combination is completed if confirmed_clips >= target_confirmations
    # OR if validated_clips == total_clips (all reviewed)
    completed = progress_df.filter(
        (pl.col("confirmed_clips") >= pl.col("target_confirmations")) |
        (pl.col("validated_clips") == pl.col("total_clips"))
    )

    return {
        "total_strata_species": len(progress_df),
        "completed_strata_species": len(completed),
        "total_validated": len(db.validation_annotations_df)
    }

@app.post("/api/validation/start-session")
async def start_validation_session(request: Request):
    """Start a validation session for a specific strata and species"""
    print("DEBUG: start_validation_session called")
    body = await request.json()
    strata_id = body.get("strata_id")
    species_name = body.get("species_name")
    validation_rules = body.get("validation_rules", {})
    review_mode = body.get("review_mode", False)
    selection_strategy = body.get("selection_strategy", "top_down")
    validation_status_filter = body.get("validation_status_filter", None)  # e.g., ['confirmed', 'rejected']
    print(f"DEBUG: Request params - strata_id={strata_id}, species={species_name}, rules={validation_rules}, review_mode={review_mode}, strategy={selection_strategy}, status_filter={validation_status_filter}")

    if not strata_id or not species_name:
        raise HTTPException(status_code=400, detail="strata_id and species_name are required")

    try:
        if app_state["validation_db"] is None:
            print("DEBUG: validation_db is None!")
            raise HTTPException(status_code=400, detail="No validation data loaded")

        validation_db = app_state["validation_db"]

        # Wait for any pending saves before starting new session
        print("INFO: Ensuring database is saved before starting new validation session")
        validation_db.wait_for_pending_save()

        print(f"DEBUG: validation_db found, predictions_df has {len(validation_db.predictions_df)} rows")

        # Get predictions for this strata and species
        predictions = validation_db.predictions_df.filter(
            (pl.col("strata_id") == strata_id) &
            (pl.col("species_name") == species_name)
        )
        print(f"DEBUG: Found {len(predictions)} total predictions for strata_id={strata_id}, species={species_name}")

        # Check for duplicates by filename and time
        duplicate_check = predictions.group_by(['filename', 'start_time', 'end_time']).agg([
            pl.count().alias('count')
        ]).filter(pl.col('count') > 1)

        if len(duplicate_check) > 0:
            print(f"WARNING: Found {len(duplicate_check)} duplicate clips (same filename + time):")
            print(duplicate_check.head(5))
            print(f"Total duplicate instances: {duplicate_check['count'].sum()}")

        # Filter by confidence threshold if specified (only for new validation)
        confidence_threshold = validation_rules.get("confidence_threshold", 0.0)
        
        # Only apply filter if threshold is meaningfully above 0 (avoid floating point precision issues)
        if confidence_threshold > 1e-6 and not review_mode:
            before_filter = len(predictions)
            predictions = predictions.filter(pl.col("confidence") >= confidence_threshold)
            print(f"DEBUG: After confidence filter: {len(predictions)} (was {before_filter})")

        # Join with annotations to include annotation status
        annotations = validation_db.validation_annotations_df
        if len(annotations) > 0:
            # Check which column name is used (validation_state or validation_label)
            annotation_col = 'validation_state' if 'validation_state' in annotations.columns else 'validation_label'
            print(f"DEBUG: Using annotation column: {annotation_col}")

            # Select relevant annotation fields and deduplicate to keep latest status
            relevant_annotations = annotations.sort("validated_at", descending=True).unique(
                subset=["prediction_id"], 
                keep="first"
            ).select([
                'prediction_id',
                pl.col(annotation_col).alias('annotation_status'),
                pl.col('validated_at').alias('annotation_timestamp')
            ])

            # Left join to include annotation status for all predictions
            predictions = predictions.join(
                relevant_annotations,
                on='prediction_id',
                how='left'
            )
            print(f"DEBUG: Joined with annotations table")
        else:
            # No annotations yet, add null columns
            predictions = predictions.with_columns([
                pl.lit(None).alias('annotation_status'),
                pl.lit(None).alias('annotation_timestamp')
            ])

        # Get already validated prediction IDs
        validated_annotations = validation_db.validation_annotations_df.filter(
            (pl.col("strata_id") == strata_id) &
            (pl.col("species_name") == species_name)
        )
        validated_ids = validated_annotations["prediction_id"].to_list()
        print(f"DEBUG: Found {len(validated_ids)} already validated predictions")
        if len(validated_annotations) > 0:
            print(f"DEBUG: Sample validated annotation: {validated_annotations.head(1).to_dicts()}")

        # Apply validation status filter if specified
        if validation_status_filter is not None and len(validation_status_filter) > 0:
            before_filter = len(predictions)
            if 'unvalidated' in validation_status_filter:
                # Special case: include clips without annotations OR with null status
                other_statuses = [s for s in validation_status_filter if s != 'unvalidated']
                if other_statuses:
                    # Include both unvalidated and specific statuses
                    predictions = predictions.filter(
                        pl.col("annotation_status").is_null() |
                        pl.col("annotation_status").is_in(other_statuses)
                    )
                else:
                    # Only unvalidated
                    predictions = predictions.filter(pl.col("annotation_status").is_null())
            else:
                # Filter for specific validation statuses
                predictions = predictions.filter(pl.col("annotation_status").is_in(validation_status_filter))
            print(f"DEBUG: After validation status filter {validation_status_filter}: {len(predictions)} (was {before_filter})")
        elif review_mode:
            # Review Mode: Filter FOR validated clips (legacy behavior if no status filter)
            before_filter = len(predictions)
            if len(validated_ids) == 0:
                print("WARNING: Review mode requested but no validated clips found for this strata/species")
            predictions = predictions.filter(pl.col("prediction_id").is_in(validated_ids))
            print(f"DEBUG: Review Mode - After filtering for validated: {len(predictions)} (was {before_filter})")

            # Sort by timestamp (newest first) for review
            if "annotation_timestamp" in predictions.columns:
                predictions = predictions.sort("annotation_timestamp", descending=True)
        else:
            # Validation Mode: Filter OUT validated clips (legacy behavior if no status filter)
            if validated_ids:
                before_filter = len(predictions)
                predictions = predictions.filter(~pl.col("prediction_id").is_in(validated_ids))
                print(f"DEBUG: Validation Mode - After excluding validated: {len(predictions)} (was {before_filter})")
            
            # Apply Selection Strategy
            if selection_strategy == "random":
                # Shuffle the predictions
                # Polars sample with n=len(df) and shuffle=True is effectively a shuffle
                predictions = predictions.sample(n=len(predictions), shuffle=True, seed=None)
                print("DEBUG: Applied RANDOM selection strategy")
            elif selection_strategy == "bottom_up":
                # Sort by confidence (lowest first)
                predictions = predictions.sort("confidence", descending=False)
                print("DEBUG: Applied BOTTOM-UP selection strategy")
            elif selection_strategy == "sequential":
                # Sort by filename and start_time for sequential clip ordering
                predictions = predictions.sort(["filename", "start_time"])
                print("DEBUG: Applied SEQUENTIAL selection strategy")
            else:
                # Default: Top-down (highest confidence first)
                predictions = predictions.sort("confidence", descending=True)
                print("DEBUG: Applied TOP-DOWN selection strategy")

        # Debug: Show confidence values of final results
        if len(predictions) > 0:
            confidences = predictions["confidence"].to_list()
            print(f"DEBUG: Final predictions count: {len(predictions)}")
            print(f"DEBUG: Confidence range: {min(confidences)} to {max(confidences)}")
            print(f"DEBUG: First 5 confidences: {confidences[:5]}")
        else:
            print("DEBUG: No predictions found for validation session!")

        # Get current strata progress (for validation counts)
        current_strata_progress = validation_db.validation_progress_df.filter(
            (pl.col("strata_id") == strata_id) &
            (pl.col("species_name") == species_name)
        )

        # Get progress across ALL strata for this species (for progress bar)
        species_progress = validation_db.validation_progress_df.filter(
            pl.col("species_name") == species_name
        )

        if len(current_strata_progress) > 0 and len(species_progress) > 0:
            # Current strata data (for validation counts)
            current_strata_row = current_strata_progress.row(0, named=True)

            # Add species-wide aggregated data (for progress bar)
            progress_row = {
                # Current strata data (used for counts display)
                "strata_id": strata_id,
                "strata_name": current_strata_row["strata_name"],
                "species_name": species_name,
                "total_clips": current_strata_row["total_clips"],
                "validated_clips": current_strata_row["validated_clips"],
                "confirmed_clips": current_strata_row["confirmed_clips"],
                "rejected_clips": current_strata_row["rejected_clips"],
                "uncertain_clips": current_strata_row["uncertain_clips"],
                "skipped_clips": current_strata_row["skipped_clips"],
                "target_confirmations": current_strata_row["target_confirmations"],
                "is_completed": current_strata_row["is_completed"],
                "last_updated": current_strata_row["last_updated"],
                # Species-wide aggregated data (used for progress bar)
                "total_strata": len(species_progress),
                "completed_strata": int(species_progress["is_completed"].sum()),
                "species_total_clips": int(species_progress["total_clips"].sum()),
                "species_validated_clips": int(species_progress["validated_clips"].sum()),
            }
        else:
            progress_row = None

        return {
            "status": "success",
            "validation_queue": predictions.to_dicts(),
            "session_progress": progress_row,
            "overall_progress": await get_overall_validation_stats()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validation/submit-annotation")
async def submit_validation_annotation(request: Request):
    """Submit a validation annotation"""
    import time
    start_time = time.time()

    body = await request.json()
    prediction_id = body.get("prediction_id")
    validation_state = body.get("validation_state")
    validation_confidence = body.get("validation_confidence", 3)
    notes = body.get("notes", "")
    strata_id = body.get("strata_id")
    species_name = body.get("species_name")

    print(f"PERF: submit_annotation - request parsed in {(time.time() - start_time)*1000:.1f}ms")

    if not prediction_id or not validation_state:
        raise HTTPException(status_code=400, detail="prediction_id and validation_state are required")

    try:
        if app_state["validation_db"] is None:
            raise HTTPException(status_code=400, detail="No validation data loaded")

        validation_db = app_state["validation_db"]

        # Acquire lock to prevent dataframe access during save
        lock_acquired = False
        try:
            validation_db.acquire_lock()
            lock_acquired = True

            # Get the original prediction
            t1 = time.time()
            prediction = validation_db.predictions_df.filter(
                pl.col("prediction_id") == prediction_id
            ).row(0, named=True)
            print(f"PERF: submit_annotation - get prediction in {(time.time() - t1)*1000:.1f}ms")

            # Check for existing annotation to handle updates
            t1 = time.time()
            existing_annotation = validation_db.validation_annotations_df.filter(
                pl.col("prediction_id") == prediction_id
            )
            print(f"PERF: submit_annotation - check existing in {(time.time() - t1)*1000:.1f}ms")

            previous_state = None
            if len(existing_annotation) > 0:
                # Get previous state for stats update
                previous_state = existing_annotation["validation_state"][0]
                print(f"DEBUG: Updating existing annotation. Previous state: {previous_state}")

                # Remove old annotation
                validation_db.validation_annotations_df = validation_db.validation_annotations_df.filter(
                    pl.col("prediction_id") != prediction_id
                )

            # Create new annotation
            import uuid
            annotation_id = str(uuid.uuid4())
            current_time = datetime.now()

            new_annotation = pl.DataFrame({
                "annotation_id": [annotation_id],
                "prediction_id": [prediction_id],
                "filename": [prediction["filename"]],
                "start_time": [float(prediction["start_time"])],
                "end_time": [float(prediction["end_time"])],
                "species_name": [prediction["species_name"]],
                "original_confidence": [float(prediction["confidence"])],
                "validation_state": [validation_state],
                "validation_confidence": [validation_confidence],
                "annotator_id": ["user"],  # Could be made dynamic
                "validated_at": [current_time],
                "strata_id": [prediction["strata_id"]],
                "notes": [notes]
            }, schema={
                "annotation_id": pl.Utf8,
                "prediction_id": pl.Utf8,
                "filename": pl.Utf8,
                "start_time": pl.Float32,
                "end_time": pl.Float32,
                "species_name": pl.Utf8,
                "original_confidence": pl.Float32,
                "validation_state": pl.Utf8,
                "validation_confidence": pl.Int32,
                "annotator_id": pl.Utf8,
                "validated_at": pl.Datetime,
                "strata_id": pl.Utf8,
                "notes": pl.Utf8
            })

            # Add to annotations database
            t1 = time.time()
            validation_db.validation_annotations_df = pl.concat([
                validation_db.validation_annotations_df, new_annotation
            ])
            print(f"PERF: submit_annotation - concat annotation in {(time.time() - t1)*1000:.1f}ms")

            # Update progress tracking
            t1 = time.time()
            if strata_id and species_name:
                # Get current progress
                current_progress = validation_db.validation_progress_df.filter(
                    (pl.col("strata_id") == strata_id) &
                    (pl.col("species_name") == species_name)
                )

                if len(current_progress) > 0:
                    # Update counts based on validation state
                    updates = {"last_updated": current_time}

                    # First, revert previous state counts if updating
                    if previous_state:
                        if previous_state == "confirmed":
                            updates["confirmed_clips"] = current_progress["confirmed_clips"].item() - 1
                        elif previous_state == "rejected":
                            updates["rejected_clips"] = current_progress["rejected_clips"].item() - 1
                        elif previous_state == "uncertain":
                            updates["uncertain_clips"] = current_progress["uncertain_clips"].item() - 1
                        elif previous_state == "skipped":
                            updates["skipped_clips"] = current_progress["skipped_clips"].item() - 1

                        updates["validated_clips"] = current_progress["validated_clips"].item() - 1

                        # Apply decrements to current_progress reference for the increments below
                        # (Note: This is a simplified in-memory update for the logic below,
                        # actual DB update happens at the end)
                        for key, val in updates.items():
                            if key != "last_updated":
                                # We need to manually update the item in our logic reference
                                # But since we are building an 'updates' dict, we can just use the values from 'updates'
                                # if they exist, or fall back to current_progress.
                                # However, 'updates' accumulates changes.
                                pass

                    # Calculate base values for incrementing (handle if they were just decremented)
                    def get_base_value(key):
                        if key in updates:
                            return updates[key]
                        return current_progress[key].item()

                    # Now increment for new state
                    if validation_state == "confirmed":
                        updates["confirmed_clips"] = get_base_value("confirmed_clips") + 1
                    elif validation_state == "rejected":
                        updates["rejected_clips"] = get_base_value("rejected_clips") + 1
                    elif validation_state == "uncertain":
                        updates["uncertain_clips"] = get_base_value("uncertain_clips") + 1
                    elif validation_state == "skipped":
                        updates["skipped_clips"] = get_base_value("skipped_clips") + 1

                    updates["validated_clips"] = get_base_value("validated_clips") + 1

                    # Check if target met
                    target_confirmations = current_progress["target_confirmations"].item()
                    target_met = updates.get("confirmed_clips", current_progress["confirmed_clips"].item()) >= target_confirmations

                    # Update the progress record
                    mask = (pl.col("strata_id") == strata_id) & (pl.col("species_name") == species_name)
                    for field, value in updates.items():
                        # Cast integer values to Int32 to match schema
                        if isinstance(value, int):
                            lit_value = pl.lit(value, dtype=pl.Int32)
                        else:
                            lit_value = pl.lit(value)
                        validation_db.validation_progress_df = validation_db.validation_progress_df.with_columns(
                            pl.when(mask).then(lit_value).otherwise(pl.col(field)).alias(field)
                        )

                    # Get updated progress for current strata and species-wide
                    t2 = time.time()
                    try:
                        current_strata_progress = validation_db.validation_progress_df.filter(mask)
                        species_progress = validation_db.validation_progress_df.filter(
                            pl.col("species_name") == species_name
                        )

                        if len(current_strata_progress) > 0 and len(species_progress) > 0:
                            # Current strata data
                            current_strata_row = current_strata_progress.row(0, named=True)

                            # Combine current strata data with species-wide aggregation
                            updated_progress = {
                                # Current strata data (used for counts display)
                                "strata_id": strata_id,
                                "strata_name": current_strata_row["strata_name"],
                                "species_name": species_name,
                                "total_clips": int(current_strata_row["total_clips"]),
                                "validated_clips": int(current_strata_row["validated_clips"]),
                                "confirmed_clips": int(current_strata_row["confirmed_clips"]),
                                "rejected_clips": int(current_strata_row["rejected_clips"]),
                                "uncertain_clips": int(current_strata_row["uncertain_clips"]),
                                "skipped_clips": int(current_strata_row["skipped_clips"]),
                                "target_confirmations": int(current_strata_row["target_confirmations"]),
                                "is_completed": bool(current_strata_row["is_completed"]),
                                "last_updated": current_strata_row["last_updated"],
                                # Species-wide aggregated data (used for progress bar)
                                "total_strata": len(species_progress),
                                "completed_strata": int(species_progress["is_completed"].sum()),
                                "species_total_clips": int(species_progress["total_clips"].sum()),
                                "species_validated_clips": int(species_progress["validated_clips"].sum()),
                            }
                        elif len(current_strata_progress) > 0:
                            # Fallback: just current strata data, add empty species-wide fields
                            current_strata_row = current_strata_progress.row(0, named=True)
                            updated_progress = {
                                "strata_id": strata_id,
                                "strata_name": current_strata_row["strata_name"],
                                "species_name": species_name,
                                "total_clips": int(current_strata_row["total_clips"]),
                                "validated_clips": int(current_strata_row["validated_clips"]),
                                "confirmed_clips": int(current_strata_row["confirmed_clips"]),
                                "rejected_clips": int(current_strata_row["rejected_clips"]),
                                "uncertain_clips": int(current_strata_row["uncertain_clips"]),
                                "skipped_clips": int(current_strata_row["skipped_clips"]),
                                "target_confirmations": int(current_strata_row["target_confirmations"]),
                                "is_completed": bool(current_strata_row["is_completed"]),
                                "last_updated": current_strata_row["last_updated"],
                                "total_strata": 1,
                                "completed_strata": 0,
                                "species_total_clips": int(current_strata_row["total_clips"]),
                                "species_validated_clips": int(current_strata_row["validated_clips"]),
                            }
                        else:
                            updated_progress = None
                            print("WARNING: No current_strata_progress found!")
                    except Exception as e:
                        print(f"ERROR building updated_progress: {e}")
                        import traceback
                        traceback.print_exc()
                        updated_progress = None

                    print(f"PERF: submit_annotation - progress aggregation in {(time.time() - t2)*1000:.1f}ms")
                    print(f"PERF: submit_annotation - total progress update in {(time.time() - t1)*1000:.1f}ms")
                    print(f"PERF: submit_annotation - TOTAL TIME: {(time.time() - start_time)*1000:.1f}ms")

                    # Store updated progress and target_met before releasing lock
                    has_progress_tracking = True
                    stored_updated_progress = updated_progress
                    stored_target_met = target_met
                else:
                    # No progress tracking
                    has_progress_tracking = False

        finally:
            # Ensure lock is always released if we acquired it
            if lock_acquired:
                try:
                    validation_db.release_lock()
                except Exception as e:
                    print(f"ERROR: Failed to release lock: {e}")

        # Now that lock is released, get overall stats (can access dataframes safely)
        t_stats = time.time()
        overall_stats = await get_overall_validation_stats()
        print(f"PERF: submit_annotation - get_overall_stats in {(time.time() - t_stats)*1000:.1f}ms")

        # Request auto-save via smart queue (non-blocking, outside lock)
        validation_db.auto_save()

        # Build result based on whether we had progress tracking
        if has_progress_tracking:
            result = {
                "status": "success",
                "session_progress": stored_updated_progress,
                "overall_progress": overall_stats,
                "target_met": stored_target_met
            }
        else:
            result = {
                "status": "success",
                "overall_progress": overall_stats
            }

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in submit_annotation: {e}")
        print(f"ERROR traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/summary")
async def get_validation_summary():
    """Get overall validation summary statistics"""
    import asyncio

    try:
        if app_state["validation_db"] is None:
            return {
                "total_strata": 0,
                "total_species": 0,
                "total_predictions": 0,
                "total_annotations": 0,
                "confirmed_count": 0,
                "rejected_count": 0,
                "uncertain_count": 0,
                "skipped_count": 0,
                "completion_percentage": 0
            }

        validation_db = app_state["validation_db"]

        # Run potentially blocking DataFrame operations in thread pool
        def compute_summary():
            # Calculate summary statistics
            total_predictions = len(validation_db.predictions_df)
            total_annotations = len(validation_db.validation_annotations_df)
            total_strata = validation_db.strata_definitions_df["strata_id"].n_unique()
            total_species = validation_db.predictions_df["species_name"].n_unique()

            # Count by validation state
            if total_annotations > 0:
                validation_counts = validation_db.validation_annotations_df.group_by("validation_state").agg(
                    pl.count().alias("count")
                ).to_dict(as_series=False)

                state_counts = {
                    state: count
                    for state, count in zip(validation_counts["validation_state"], validation_counts["count"])
                }
            else:
                state_counts = {}

            completion_percentage = (total_annotations / total_predictions * 100) if total_predictions > 0 else 0

            return {
                "total_strata": total_strata,
                "total_species": total_species,
                "total_predictions": total_predictions,
                "total_annotations": total_annotations,
                "confirmed_count": state_counts.get("confirmed", 0),
                "rejected_count": state_counts.get("rejected", 0),
                "uncertain_count": state_counts.get("uncertain", 0),
                "skipped_count": state_counts.get("skipped", 0),
                "completion_percentage": completion_percentage
            }

        summary = await asyncio.to_thread(compute_summary)
        return summary

    except Exception as e:
        import traceback
        print(f"ERROR in validation summary: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/strata-progress")
async def get_strata_progress():
    """Get detailed progress for all strata"""
    import asyncio

    try:
        if app_state["validation_db"] is None:
            return {"strata_progress": []}

        validation_db = app_state["validation_db"]

        # Run potentially blocking DataFrame operations in thread pool
        def compute_progress():
            # Join progress data with strata definitions to get confidence_threshold
            progress_df = validation_db.validation_progress_df.join(
                validation_db.strata_definitions_df.select(['strata_id', 'confidence_threshold']),
                on='strata_id',
                how='left'
            )
            progress_data = progress_df.to_dicts()

            # Add completion_status to each record based on target_confirmations
            for record in progress_data:
                confirmed = record.get('confirmed_clips', 0)
                target = record.get('target_confirmations', 1)
                validated = record.get('validated_clips', 0)
                total = record.get('total_clips', 0)

                # If no clips above threshold, mark as target_met (nothing to validate)
                if total == 0:
                    record['completion_status'] = 'target_met'
                elif confirmed >= target:
                    record['completion_status'] = 'target_met'
                elif validated > 0:
                    record['completion_status'] = 'in_progress'
                else:
                    record['completion_status'] = 'not_started'

            return progress_data

        progress_data = await asyncio.to_thread(compute_progress)

        return {"strata_progress": progress_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validation/toggle-strata-completion")
async def toggle_strata_completion(request: Request):
    """Toggle completion status for a strata/species combination"""
    import asyncio

    try:
        body = await request.json()
        strata_id = body.get("strata_id")
        species_name = body.get("species_name")
        is_completed = body.get("is_completed", False)

        if not strata_id or not species_name:
            raise HTTPException(status_code=400, detail="strata_id and species_name are required")

        if app_state["validation_db"] is None:
            raise HTTPException(status_code=400, detail="No validation database loaded")

        validation_db = app_state["validation_db"]

        # Run potentially blocking DataFrame operations in thread pool
        result = await asyncio.to_thread(validation_db.toggle_strata_completion, strata_id, species_name, is_completed)

        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])

        # Request auto-save and WAIT for completion when toggling strata completion
        validation_db.auto_save()
        print("INFO: Waiting for save to complete after toggling strata completion")
        validation_db.wait_for_pending_save()

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/annotations")
async def get_validation_annotations(
    strata_id: Optional[str] = None,
    species_name: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    sort_field: str = "validated_at",
    sort_direction: str = "desc"
):
    """Get detailed validation annotations with filtering and pagination"""
    try:
        if app_state["validation_db"] is None:
            return {"annotations": [], "total_count": 0}

        validation_db = app_state["validation_db"]

        # Join annotations with predictions to get confidence scores
        annotations = validation_db.validation_annotations_df
        predictions = validation_db.predictions_df
        strata_defs = validation_db.strata_definitions_df

        # Join on prediction_id to get original confidence
        joined = annotations.join(
            predictions.select(['prediction_id', 'confidence']),
            on='prediction_id',
            how='left'
        )

        # Join with strata definitions to get strata name
        if len(strata_defs) > 0:
            joined = joined.join(
                strata_defs.select(['strata_id', 'strata_name']),
                on='strata_id',
                how='left'
            )

        # Apply filters
        if strata_id:
            joined = joined.filter(pl.col("strata_id") == strata_id)
        if species_name:
            joined = joined.filter(pl.col("species_name") == species_name)

        total_count = len(joined)

        # Apply sorting - validate sort field exists
        if sort_field in joined.columns:
            descending = sort_direction == "desc"
            joined = joined.sort(sort_field, descending=descending)
        else:
            # Default to validated_at if sort field doesn't exist
            print(f"WARNING: Sort field '{sort_field}' not found in columns: {joined.columns}")
            if 'validated_at' in joined.columns:
                joined = joined.sort('validated_at', descending=True)

        # Apply pagination
        offset = (page - 1) * limit
        joined = joined.slice(offset, limit)

        # Debug: Check what columns we have
        print(f"DEBUG: Joined columns: {joined.columns}")
        if len(joined) > 0:
            print(f"DEBUG: First row sample: {joined.head(1).to_dicts()}")

        # Convert to dicts and rename fields for frontend compatibility
        result_annotations = []
        for row in joined.to_dicts():
            # Check if validation_state already exists (from loaded project),
            # otherwise fall back to validation_label (from new annotations)
            if 'validation_state' not in row and 'validation_label' in row:
                row['validation_state'] = row['validation_label']
            elif 'validation_state' not in row:
                row['validation_state'] = 'unknown'

            # Set original_confidence if not already present
            if 'original_confidence' not in row and 'confidence' in row:
                row['original_confidence'] = row['confidence']

            result_annotations.append(row)

        return {
            "annotations": result_annotations,
            "total_count": total_count,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        print(f"ERROR in get_validation_annotations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/export/{format}")
async def export_validation_results(
    format: str,
    strata_id: Optional[str] = None,
    species_name: Optional[str] = None
):
    """Export validation results in specified format"""
    try:
        if app_state["validation_db"] is None:
            raise HTTPException(status_code=400, detail="No validation data available")

        validation_db = app_state["validation_db"]

        if format.lower() == "csv":
            # Export annotations as CSV
            annotations = validation_db.validation_annotations_df

            # Apply filters
            if strata_id:
                annotations = annotations.filter(pl.col("strata_id") == strata_id)
            if species_name:
                annotations = annotations.filter(pl.col("species_name") == species_name)

            # Convert to CSV
            import io
            csv_buffer = io.StringIO()
            annotations.write_csv(csv_buffer)

            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=validation_results.csv"}
            )

        elif format.lower() == "json":
            # Export as JSON
            annotations = validation_db.validation_annotations_df

            # Apply filters
            if strata_id:
                annotations = annotations.filter(pl.col("strata_id") == strata_id)
            if species_name:
                annotations = annotations.filter(pl.col("species_name") == species_name)

            import io
            json_data = annotations.to_dicts()

            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                io.BytesIO(json.dumps(json_data, indent=2, default=str).encode()),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=validation_results.json"}
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== VALIDATION PROJECT PERSISTENCE =====

@app.post("/api/validation/save-project")
async def save_validation_project(request: Request):
    """Save validation project to disk for later loading"""
    import asyncio

    try:
        body = await request.json()
    except:
        body = {}

    base_path = body.get("base_path")
    project_name = body.get("project_name")

    try:
        if app_state["validation_db"] is None:
            raise HTTPException(status_code=400, detail="No validation data to save")

        validation_db = app_state["validation_db"]

        # Use stored project path if no path provided (for auto-save)
        if not base_path:
            if validation_db.project_base_path:
                base_path = validation_db.project_base_path
                project_name = validation_db.project_name
            else:
                # Default to current working directory
                base_path = "."

        # Run blocking I/O operation in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(validation_db.save_validation_database, base_path, project_name)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validation/load-project")
async def load_validation_project(request: Request):
    """Load validation project from disk"""
    import asyncio

    body = await request.json()
    project_path = body.get("project_path")

    if not project_path:
        raise HTTPException(status_code=400, detail="project_path is required")

    try:
        from modules.validation_db import ValidationDB

        # Create or reuse validation database instance
        if app_state["validation_db"] is None:
            print("INFO: Creating new validation database instance")
            app_state["validation_db"] = ValidationDB()
        else:
            # Reuse existing instance - just wait for any pending saves
            print("INFO: Reusing existing validation database instance")
            app_state["validation_db"].wait_for_pending_save()

        validation_db = app_state["validation_db"]

        # Run blocking I/O operation in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(validation_db.load_validation_database, project_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/list-projects")
async def list_validation_projects(base_path: str):
    """List available validation projects in a directory"""
    import asyncio

    if not base_path:
        raise HTTPException(status_code=400, detail="base_path is required")

    try:
        from modules.validation_db import ValidationDB
        temp_db = ValidationDB()  # Create temporary instance for listing

        # Run blocking I/O operation in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(temp_db.list_validation_projects, base_path)

        # Shutdown the temporary worker thread to prevent accumulation
        temp_db.shutdown_save_worker()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/diagnostic")
async def validation_diagnostic():
    """Diagnostic endpoint to check validation database contents"""
    try:
        if app_state["validation_db"] is None:
            return {
                "status": "error",
                "message": "No validation database loaded"
            }

        validation_db = app_state["validation_db"]

        # Get basic counts
        total_predictions = len(validation_db.predictions_df)
        total_annotations = len(validation_db.validation_annotations_df)
        total_strata = len(validation_db.strata_definitions_df)

        # Get unique values
        unique_strata_ids = validation_db.predictions_df["strata_id"].unique().to_list() if total_predictions > 0 else []
        unique_species = validation_db.predictions_df["species_name"].unique().to_list() if total_predictions > 0 else []

        # Get strata details
        strata_details = []
        for strata_row in validation_db.strata_definitions_df.iter_rows(named=True):
            strata_id = strata_row['strata_id']
            strata_name = strata_row['strata_name']

            # Count predictions for this strata
            strata_preds = validation_db.predictions_df.filter(
                validation_db.predictions_df['strata_id'] == strata_id
            )

            # Get species in this strata
            species_in_strata = {}
            for species in unique_species:
                species_preds = strata_preds.filter(
                    strata_preds['species_name'] == species
                )
                if len(species_preds) > 0:
                    confidences = species_preds['confidence'].to_list()
                    species_in_strata[species] = {
                        'total_clips': len(species_preds),
                        'min_confidence': float(min(confidences)),
                        'max_confidence': float(max(confidences)),
                        'avg_confidence': float(sum(confidences) / len(confidences))
                    }

            strata_details.append({
                'strata_id': strata_id,
                'strata_name': strata_name,
                'total_predictions': len(strata_preds),
                'species': species_in_strata
            })

        return {
            "status": "success",
            "total_predictions": total_predictions,
            "total_annotations": total_annotations,
            "total_strata": total_strata,
            "unique_strata_ids": unique_strata_ids[:10],  # First 10
            "unique_species": unique_species[:20],  # First 20
            "strata_details": strata_details
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/api/validation/save-status")
async def validation_save_status():
    """Get status of smart queue save system"""
    try:
        if app_state["validation_db"] is None:
            return {
                "status": "no_database",
                "message": "No validation database loaded"
            }

        validation_db = app_state["validation_db"]
        save_status = validation_db.get_save_status()

        return {
            "status": "success",
            **save_status
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/audio/{file_path:path}")
def get_audio_clip(file_path: str, clip_start: float = None, clip_end: float = None):
    """Get audio clip with disk caching for extracted clips (shared by all interfaces)"""
    from urllib.parse import unquote
    from pathlib import Path
    import soundfile as sf
    import io
    import hashlib
    import os
    from fastapi.responses import StreamingResponse, FileResponse

    # Cache directory setup (cross-platform)
    import tempfile
    CACHE_DIR = Path(tempfile.gettempdir()) / "bioacoustic_audio_cache"
    MAX_CACHE_SIZE_GB = 2
    MAX_CACHE_SIZE_BYTES = MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024

    def get_cache_key(file_path: str, start: float, end: float) -> str:
        """Generate unique cache key for a clip"""
        cache_string = f"{file_path}_{start}_{end}"
        return hashlib.md5(cache_string.encode()).hexdigest() + ".wav"

    def get_cache_size() -> int:
        """Get total size of cache directory in bytes"""
        total_size = 0
        if CACHE_DIR.exists():
            for file in CACHE_DIR.iterdir():
                if file.is_file():
                    total_size += file.stat().st_size
        return total_size

    def cleanup_old_cache_files():
        """Remove oldest cache files if cache exceeds size limit"""
        if not CACHE_DIR.exists():
            return

        current_size = get_cache_size()
        if current_size <= MAX_CACHE_SIZE_BYTES:
            return

        # Get all cache files sorted by access time (oldest first)
        cache_files = [(f, f.stat().st_atime) for f in CACHE_DIR.iterdir() if f.is_file()]
        cache_files.sort(key=lambda x: x[1])

        # Remove oldest files until under limit
        for file_path, _ in cache_files:
            if current_size <= MAX_CACHE_SIZE_BYTES * 0.8:  # Keep at 80% to avoid frequent cleanup
                break
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                current_size -= file_size
                print(f"Cache cleanup: Removed {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"Cache cleanup error: {e}")

    try:
        # Decode URL-encoded file path
        decoded_path = unquote(file_path)

        # Handle path reconstruction (cross-platform)
        # FastAPI strips leading slashes, so we need to restore them for Unix paths
        # but NOT for Windows paths (which start with drive letters like C:)
        file_path_obj = Path(decoded_path)
        if not file_path_obj.is_absolute():
            # If not absolute, assume it's a Unix path missing the leading slash
            decoded_path = '/' + decoded_path
            file_path_obj = Path(decoded_path)

        # Check if file exists
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {decoded_path}")

        # Optimization: If no specific clip range is requested, OR if we want to rely on browser seeking,
        # serve the file directly using FileResponse. This enables Range requests (efficient seeking),
        # OS-level caching, and avoids expensive decoding/encoding in Python.
        if clip_start is None and clip_end is None:
            return FileResponse(
                decoded_path,
                media_type="audio/wav",
                headers={
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "Accept-Ranges": "bytes"
                }
            )

        # Legacy/Specific Clip Logic: If specific start/end times are requested via query params,
        # we extract that specific segment with disk caching for performance.

        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check cache first
        cache_key = get_cache_key(decoded_path, clip_start, clip_end)
        cache_file_path = CACHE_DIR / cache_key

        if cache_file_path.exists():
            # Serve from cache (instant!)
            print(f"Cache HIT: Serving {cache_key} from cache")
            # Update access time for LRU cleanup
            cache_file_path.touch()
            return FileResponse(
                cache_file_path,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=clip.wav",
                    "Cache-Control": "public, max-age=31536000, immutable"
                }
            )

        # Cache MISS: Extract clip from source file
        print(f"Cache MISS: Extracting clip from {decoded_path} ({clip_start}-{clip_end}s)")

        # Efficiently read and process audio
        with sf.SoundFile(decoded_path) as f:
            original_sr = f.samplerate

            start_frame = int(clip_start * original_sr)
            end_frame = int(clip_end * original_sr)
            # Ensure we don't seek past end
            if start_frame < f.frames:
                f.seek(start_frame)
                frames_to_read = min(end_frame - start_frame, f.frames - start_frame)
                audio = f.read(frames_to_read)
            else:
                audio = np.array([])

        # Save to cache file
        try:
            sf.write(cache_file_path, audio, original_sr, format='WAV')
            print(f"Cache SAVE: Saved {cache_key} ({cache_file_path.stat().st_size / 1024:.2f} KB)")

            # Cleanup old cache if needed
            cleanup_old_cache_files()
        except Exception as e:
            print(f"Cache save error (non-fatal): {e}")

        # Serve the cached file
        return FileResponse(
            cache_file_path,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=clip.wav",
                "Cache-Control": "public, max-age=31536000, immutable"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Failed to serve audio: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")

@app.get("/api/validation/cache/stats")
def get_cache_stats():
    """Get audio cache statistics"""
    from pathlib import Path

    CACHE_DIR = Path("/tmp/bioacoustic_audio_cache")

    try:
        if not CACHE_DIR.exists():
            return {
                "status": "success",
                "cache_enabled": True,
                "cache_size_bytes": 0,
                "cache_size_mb": 0,
                "file_count": 0,
                "max_size_gb": 2
            }

        # Count files and calculate total size
        total_size = 0
        file_count = 0
        for file in CACHE_DIR.iterdir():
            if file.is_file():
                total_size += file.stat().st_size
                file_count += 1

        return {
            "status": "success",
            "cache_enabled": True,
            "cache_size_bytes": total_size,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "max_size_gb": 2
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/validation/cache/clear")
def clear_cache():
    """Clear the audio cache"""
    from pathlib import Path
    import shutil

    CACHE_DIR = Path("/tmp/bioacoustic_audio_cache")

    try:
        if not CACHE_DIR.exists():
            return {
                "status": "success",
                "message": "Cache directory does not exist (already empty)"
            }

        # Get stats before clearing
        file_count = len(list(CACHE_DIR.iterdir()))
        total_size = sum(f.stat().st_size for f in CACHE_DIR.iterdir() if f.is_file())

        # Remove all cache files
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        return {
            "status": "success",
            "message": f"Cache cleared: {file_count} files ({total_size / (1024 * 1024):.2f} MB) removed"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    import subprocess
    import signal

    # Clean up any existing process on port 8000
    try:
        result = subprocess.run(['lsof', '-ti:8000'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"INFO: Killing existing process {pid} on port 8000")
                    os.kill(int(pid), signal.SIGKILL)
                    time.sleep(0.5)  # Give it a moment to release the port
    except Exception as e:
        print(f"INFO: Port cleanup check: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)