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
    "review_mode": "random"  # Default review mode
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
    color_mode: str = "viridis"  # viridis, gray_r

class TrainingParams(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    n_steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 0.001
    model_type: int = 2
    verbose: bool = True
    weak_neg_weight: float = 0.05

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
        num_classes = len(config.class_map)
        audio_db = db.Audio_DB(num_classes=num_classes)
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
    valid_modes = ["random", "top_down", "top_10+score_quantiles"]
    
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
        num_classes = len(metadata.get("class_map", {})) if metadata else 1
        audio_db = db.Audio_DB(num_classes=num_classes)
        
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
        
        # Get filtered clips using the annotation interface
        filtered_df = annotation_interface.get_filtered_clips(
            score_min=filter_config.score_min,
            score_max=filter_config.score_max,
            annotation_filter=filter_config.annotation_filter
        )
        
        # Get the next clip based on review mode
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
        annotation_map = {0: "not_present", 1: "present", 2: "uncertain", 4: "not_reviewed"}
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
        annotation_map = {0: "not_present", 1: "present", 2: "uncertain", 4: "not_reviewed"}
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

@app.post("/api/active-learning/export-clips")
async def export_clips(export_path: str, annotation_slug: str = "export"):
    """Export annotated clips as WAV files with enhanced metadata"""
    if app_state["audio_db"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    try:
        # Pass class_map to the database for metadata export
        if app_state["class_map"]:
            app_state["audio_db"].class_map = app_state["class_map"]
            
        positive_count, negative_count, uncertain_count = app_state["audio_db"].export_wav_clips(
            export_path, annotation_slug
        )
        
        total_count = positive_count + negative_count + uncertain_count
        
        return {
            "status": "success",
            "positive_clips": positive_count,
            "negative_clips": negative_count,
            "uncertain_clips": uncertain_count,
            "total_clips": total_count,
            "message": f"Exported {total_count} clips ({positive_count} positive, {negative_count} negative, {uncertain_count} uncertain) with enhanced metadata"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spectrogram")
async def generate_spectrogram(request: SpectrogramRequest):
    """Generate spectrogram data for a clip"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
        import io
        import base64
        
        # Load audio with buffer
        buffer_s = 1.0  # 1 second buffer
        buffered_start = max(0, request.clip_start - buffer_s)
        
        import soundfile as sf
        f = sf.SoundFile(request.file_path)
        file_duration = f.frames / f.samplerate
        buffered_end = min(file_duration, request.clip_end + buffer_s)
        
        # Load buffered audio
        audio = u.load_audio(request.file_path, (buffered_start, buffered_end, f.samplerate))
        
        # Create spectrogram
        plt.figure(figsize=(12, 6))
        nyquist = cfg.MODEL_SR // 2
        fmax = min(cfg.MAX_FREQ, nyquist)
        
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=cfg.MODEL_SR,
            n_mels=256,
            fmax=fmax,
            hop_length=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Display
        # Validate color mode and use it directly
        valid_cmaps = ['viridis', 'plasma', 'inferno', 'gray_r', 'magma', 'cividis']
        cmap = request.color_mode if request.color_mode in valid_cmaps else 'viridis'
        print(f"DEBUG: Using colormap: {cmap}")
        librosa.display.specshow(
            S_dB, 
            sr=cfg.MODEL_SR,
            x_axis='time', 
            y_axis='mel', 
            fmax=fmax,
            x_coords=np.linspace(buffered_start, buffered_end, S.shape[1]),
            cmap=cmap
        )
        
        plt.colorbar(format='%+2.0f dB')
        
        # Add clip boundaries
        plt.axvline(x=request.clip_start, color='r', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=request.clip_end, color='r', linestyle='-', linewidth=2, alpha=0.7)
        
        plt.xlabel("Time (seconds)")
        plt.title(f'Clip: {request.clip_start:.1f}s - {request.clip_end:.1f}s')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"spectrogram": f"data:image/png;base64,{image_data}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{file_path:path}")
async def get_audio_clip(file_path: str, clip_start: float, clip_end: float):
    """Extract and return audio clip"""
    try:
        # Load the specific clip
        audio = u.load_audio(file_path, None)
        start_idx = int(clip_start * cfg.TARGET_SR)
        end_idx = int(clip_end * cfg.TARGET_SR)
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(audio) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(audio)))
        
        clip_audio = audio[start_idx:end_idx]
        
        # Convert to bytes for streaming
        import soundfile as sf
        import io
        
        buffer = io.BytesIO()
        sf.write(buffer, clip_audio, cfg.TARGET_SR, format='WAV')
        buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=clip.wav"}
        )
        
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
        
        # Call fit_w_tape with label strength support and weak_neg_weight=0.05
        classifier_model, train_losses, val_losses, cmaps = tc.fit_w_tape(
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
            weak_neg_weight=config.training_params.weak_neg_weight
        )
        
        # Add final metrics to logs
        try:
            final_loss = train_losses[-1] if train_losses and len(train_losses) > 0 else None
            best_cmap = max(cmaps) if cmaps and len(cmaps) > 0 and all(isinstance(x, (int, float)) and not np.isnan(x) for x in cmaps) else None
            training_state["logs"].append(f"Final loss: {final_loss}")
            training_state["logs"].append(f"Macro cMAP of best fit: {best_cmap}")
            
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)